"""Tests for Circuit Breaker pattern, Health Check endpoints, and Telemetry metrics (Phase 12)."""

import time
import pytest
from backend.app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    CircuitBreakerState,
)
from backend.app.core.metrics import get_metrics_collector
from backend.app.main import (
    health_liveness,
    health_readiness,
    health_dependencies,
    get_prometheus_metrics,
)


def test_circuit_breaker_lifecycle():
    """Verify state transitions: CLOSED -> trips to OPEN -> recovery timeout -> HALF_OPEN -> CLOSED."""
    cb = CircuitBreaker(
        name="test_service",
        failure_threshold=3,
        recovery_timeout=0.2,  # 200ms recovery timeout for fast testing
        half_open_probes=2,
    )

    # 1. Initial State: CLOSED
    assert cb.state == CircuitBreakerState.CLOSED

    # 2. Record 2 failures: still CLOSED
    cb.record_failure(Exception("Network error 1"))
    cb.record_failure(Exception("Network error 2"))
    assert cb.state == CircuitBreakerState.CLOSED

    # 3. Record 3rd failure: trips to OPEN
    cb.record_failure(Exception("Network error 3"))
    assert cb.state == CircuitBreakerState.OPEN

    # 4. Immediate execution attempt while OPEN must fast-fail
    with pytest.raises(CircuitBreakerOpenException, match="OPEN"):
        cb.before_call()

    # 5. Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.25)
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 6. First probe succeeds
    cb.record_success()
    assert cb.state == CircuitBreakerState.HALF_OPEN

    # 7. Second probe succeeds: transitions back to CLOSED
    cb.record_success()
    assert cb.state == CircuitBreakerState.CLOSED


def test_api_health_liveness():
    """Verify /api/v1/health/live returns healthy status, uptime, and pid."""
    data = health_liveness()
    assert data["status"] == "healthy"
    assert data["probe"] == "liveness"
    assert "uptime_seconds" in data
    assert "pid" in data


def test_api_health_readiness():
    """Verify /api/v1/health/ready validates database and storage availability."""
    data = health_readiness()
    assert data["status"] == "ready"
    assert data["database_connected"] is True
    assert data["storage_available"] is True


def test_api_health_dependencies():
    """Verify /api/v1/health/deps inspects database and external circuit breakers."""
    data = health_dependencies()
    assert "dependencies" in data
    assert "database" in data["dependencies"]
    assert "open_meteo_weather" in data["dependencies"]
    assert "gnani_speech_ai" in data["dependencies"]
    assert "circuit_breaker" in data["dependencies"]["open_meteo_weather"]


def test_api_metrics_endpoint():
    """Verify /api/v1/metrics exports Prometheus text format and JSON summary."""
    # 1. Prometheus text format
    resp_text = get_prometheus_metrics(format="text")
    body_text = resp_text.body.decode("utf-8") if hasattr(resp_text, "body") else str(resp_text)
    assert "agriguardian_http_requests_total" in body_text
    assert "agriguardian_uptime_seconds" in body_text

    # 2. JSON format
    resp_json = get_prometheus_metrics(format="json")
    assert "uptime_seconds" in resp_json
    assert "total_requests" in resp_json
    assert "average_latency_ms" in resp_json

