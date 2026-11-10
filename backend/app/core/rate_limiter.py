"""In-memory thread-safe rate limiter and brute-force protection for AgriGuardian (Phase 12)."""

import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List, Optional, Tuple


class LoginRateLimiter:
    """Tracks failed authentication attempts per username/IP to protect against credential stuffing and brute-force attacks."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300, lockout_seconds: int = 600):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self._lock = Lock()
        self._failures: Dict[str, List[float]] = defaultdict(list)
        self._lockouts: Dict[str, float] = {}

    def is_locked_out(self, key: str) -> Tuple[bool, int]:
        """Check if a given key (e.g. username or IP) is locked out.
        
        Returns (is_locked_out, remaining_lockout_seconds).
        """
        now = time.time()
        with self._lock:
            # Check active lockout
            if key in self._lockouts:
                remaining = int(self._lockouts[key] - now)
                if remaining > 0:
                    return True, remaining
                else:
                    del self._lockouts[key]
                    self._failures[key] = []

            # Clean expired failures outside sliding window
            cutoff = now - self.window_seconds
            self._failures[key] = [t for t in self._failures[key] if t > cutoff]

            if len(self._failures[key]) >= self.max_attempts:
                self._lockouts[key] = now + self.lockout_seconds
                return True, self.lockout_seconds

            return False, 0

    def record_failure(self, key: str) -> None:
        """Record a failed login attempt."""
        now = time.time()
        with self._lock:
            self._failures[key].append(now)
            if len(self._failures[key]) >= self.max_attempts:
                self._lockouts[key] = now + self.lockout_seconds

    def record_success(self, key: str) -> None:
        """Clear failures on successful login."""
        with self._lock:
            self._failures.pop(key, None)
            self._lockouts.pop(key, None)


class SimpleAPIRateLimiter:
    """Sliding-window request rate limiter per client IP."""

    def __init__(self, max_requests: int = 120, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        """Check whether client IP is within rate limits.
        
        Returns (is_allowed, retry_after_seconds).
        """
        now = time.time()
        with self._lock:
            cutoff = now - self.window_seconds
            self._requests[client_ip] = [t for t in self._requests[client_ip] if t > cutoff]
            if len(self._requests[client_ip]) >= self.max_requests:
                oldest = self._requests[client_ip][0]
                retry_after = max(1, int(oldest + self.window_seconds - now))
                return False, retry_after
            self._requests[client_ip].append(now)
            return True, 0


# Global singletons
_login_limiter: Optional[LoginRateLimiter] = None
_api_limiter: Optional[SimpleAPIRateLimiter] = None


def get_login_rate_limiter() -> LoginRateLimiter:
    """Get singleton login rate limiter."""
    global _login_limiter
    if _login_limiter is None:
        _login_limiter = LoginRateLimiter(max_attempts=5, window_seconds=300, lockout_seconds=600)
    return _login_limiter


def get_api_rate_limiter() -> SimpleAPIRateLimiter:
    """Get singleton API rate limiter."""
    global _api_limiter
    if _api_limiter is None:
        _api_limiter = SimpleAPIRateLimiter(max_requests=120, window_seconds=60)
    return _api_limiter

