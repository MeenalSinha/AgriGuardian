"""Circuit Breaker pattern implementation for external service resilience in AgriGuardian (Phase 12)."""

import time
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple
from backend.app.core.logging import get_logger

logger = get_logger("CircuitBreaker")


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation; requests pass through
    OPEN = "OPEN"           # Tripped; requests fail immediately without calling provider
    HALF_OPEN = "HALF_OPEN" # Testing recovery; a limited number of probe calls are permitted


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    def __init__(self, service_name: str, retry_after_seconds: float):
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker for service '{service_name}' is OPEN. "
            f"Failing fast to preserve system stability. Try again in {retry_after_seconds:.1f}s."
        )


class CircuitBreaker:
    """Thread-safe circuit breaker protecting the system against cascading failures from external dependencies."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_probes: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_probes = half_open_probes

        self._lock = Lock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        self._last_failure_time: Optional[float] = None
        self._last_error: Optional[str] = None

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            self._evaluate_state_transition()
            return self._state

    def _evaluate_state_transition(self) -> None:
        """Evaluate if state should transition from OPEN to HALF_OPEN after recovery timeout."""
        now = time.time()
        if self._state == CircuitBreakerState.OPEN:
            if now - self._last_state_change >= self.recovery_timeout:
                logger.info(f"CircuitBreaker[{self.name}]: Transitioning from OPEN to HALF_OPEN for probing.")
                self._state = CircuitBreakerState.HALF_OPEN
                self._last_state_change = now
                self._success_count = 0

    def before_call(self) -> None:
        """Check whether execution is allowed before initiating call."""
        with self._lock:
            self._evaluate_state_transition()
            if self._state == CircuitBreakerState.OPEN:
                remaining = max(0.1, self.recovery_timeout - (time.time() - self._last_state_change))
                raise CircuitBreakerOpenException(self.name, remaining)

    def record_success(self) -> None:
        """Record a successful provider call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_probes:
                    logger.info(f"CircuitBreaker[{self.name}]: Recovery probes succeeded. Transitioning to CLOSED.")
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    self._last_state_change = time.time()
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count = 0

    def record_failure(self, error: Exception) -> None:
        """Record a failed provider call."""
        with self._lock:
            now = time.time()
            self._last_failure_time = now
            self._last_error = str(error)

            if self._state == CircuitBreakerState.HALF_OPEN:
                logger.warning(f"CircuitBreaker[{self.name}]: Probe call failed ({error}). Tripping back to OPEN.")
                self._state = CircuitBreakerState.OPEN
                self._last_state_change = now
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    logger.error(
                        f"CircuitBreaker[{self.name}]: Failure threshold ({self.failure_threshold}) exceeded. "
                        f"Tripping circuit to OPEN for {self.recovery_timeout}s."
                    )
                    self._state = CircuitBreakerState.OPEN
                    self._last_state_change = now

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronously execute a callable protected by this circuit breaker."""
        self.before_call()
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of the circuit breaker's status."""
        with self._lock:
            self._evaluate_state_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_error": self._last_error,
                "last_failure_time": self._last_failure_time,
                "recovery_timeout_seconds": self.recovery_timeout,
            }


# Registry of circuit breakers across external services
_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = Lock()


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Retrieve or create named circuit breaker."""
    with _registry_lock:
        if service_name not in _breakers:
            _breakers[service_name] = CircuitBreaker(name=service_name)
        return _breakers[service_name]

