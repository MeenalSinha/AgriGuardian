# AGRIGUARDIAN — OBSERVABILITY & OPERATIONS GUIDE
**Phase 12 Telemetry, Logging, Health Monitoring, Circuit Breakers & Prometheus Metrics**  
**Author:** AgriGuardian Reliability & SRE Engineering Team  
**Date:** November 10, 2026  
**Status:** Operational & Validated in Production  

---

## 1. Observability Architecture Overview

AgriGuardian implements a modern telemetry and observability stack designed for production containerized environments:

```
                  +-------------------------------------------------+
                  |              Inbound HTTP Traffic               |
                  +-----------------------+-------------------------+
                                          |
                                          v
                  +-------------------------------------------------+
                  |          CorrelationIdMiddleware (ASGI)         |
                  |  - Assigns/Propagates X-Request-ID              |
                  |  - Sets contextvars request correlation ID      |
                  |  - Records authentic latency & status to Metrics|
                  +-----------------------+-------------------------+
                                          |
                                          v
       +----------------------------------+----------------------------------+
       |                                                                     |
       v                                                                     v
+-------------------------------+                             +-------------------------------+
|     Structured JSON Logs      |                             |   Prometheus Live Telemetry   |
| - ISO-8601 UTC Timestamps     |                             | - Counter: requests_total     |
| - Correlation ID embedding    |                             | - Gauge: uptime_seconds       |
| - Sensitive credential filter |                             | - Gauge: latency_p95_ms       |
| - Streamed to stdout / file   |                             | - Gauge: error_rate_pct       |
+-------------------------------+                             +-------------------------------+
                                                                             ^
                                                                             |
                                      +--------------------------------------+
                                      |
       +------------------------------+------------------------------+
       |                                                             |
       v                                                             v
+------------------------------------+             +------------------------------------+
|   Open-Meteo Circuit Breaker       |             |     Gnani AI Circuit Breaker       |
|   State: CLOSED/OPEN/HALF_OPEN     |             |   State: CLOSED/OPEN/HALF_OPEN     |
|   Fallback: Climatological normals |             |   Fallback: Plain text UI advisory |
+------------------------------------+             +------------------------------------+
```

---

## 2. Structured JSON Logging

Implemented in [`backend/app/core/logging.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/core/logging.py):

### 2.1 Schema of Log Records

Every production log output is emitted as a single-line JSON record:

```json
{
  "timestamp": "2026-11-10T14:15:37.123456+00:00",
  "level": "INFO",
  "logger": "AgriGuardian.AuthService",
  "message": "User authenticated successfully: 'kisan_patel'",
  "correlation_id": "req_8a3f91bc7d20",
  "module": "auth_service",
  "line": 142
}
```

### 2.2 Sensitive Data Redaction Filter

To ensure privacy compliance and prevent credential leakage into centralized log collectors (e.g. Datadog, CloudWatch, Loki):
- Automatically intercepts all log messages via `SensitiveDataFilter`.
- Replaces authorization tokens, passwords, bearer headers, and API keys with `[REDACTED]`.
- Enforces strict zero-leakage policy for credentials.

---

## 3. Real Operational Metrics (Prometheus Exposition)

Implemented in [`backend/app/core/metrics.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/core/metrics.py):

### 3.1 Endpoint

- **Path:** `GET /api/v1/metrics`
- **Formats:**
  - `?format=text` (Standard Prometheus exposition format, `text/plain; version=0.0.4`)
  - `?format=json` (Structured JSON representation for internal dashboards)

### 3.2 Metrics Catalog

```prometheus
# HELP agriguardian_http_requests_total Total HTTP requests served by endpoint and status.
# TYPE agriguardian_http_requests_total counter
agriguardian_http_requests_total{endpoint="/api/v1/farms",status="201"} 14
agriguardian_http_requests_total{endpoint="/api/v1/economics/expenses",status="200"} 85

# HELP agriguardian_uptime_seconds Application process uptime in seconds.
# TYPE agriguardian_uptime_seconds gauge
agriguardian_uptime_seconds 4521.8

# HELP agriguardian_error_rate_pct Percentage of requests returning HTTP 5xx errors.
# TYPE agriguardian_error_rate_pct gauge
agriguardian_error_rate_pct 0.00

# HELP agriguardian_latency_average_ms Average request duration in milliseconds.
# TYPE agriguardian_latency_average_ms gauge
agriguardian_latency_average_ms 14.2

# HELP agriguardian_latency_p95_ms 95th percentile request duration in milliseconds.
# TYPE agriguardian_latency_p95_ms gauge
agriguardian_latency_p95_ms 42.1

# HELP agriguardian_circuit_breaker_state Circuit breaker state (0=closed, 1=half_open, 2=open).
# TYPE agriguardian_circuit_breaker_state gauge
agriguardian_circuit_breaker_state{name="open_meteo"} 0
agriguardian_circuit_breaker_state{name="gnani_ai"} 0
```

---

## 4. Multi-Tiered Health Probes

Three dedicated health check endpoints in [`backend/app/main.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/main.py) provide Kubernetes and Docker orchestrators with accurate lifecycle signals:

### 4.1 Liveness Probe (`GET /api/v1/health/live`)

- **Purpose:** Verifies that the FastAPI process is alive, the event loop is executing, and the process has not deadlocked.
- **Payload:**
  ```json
  {
    "status": "healthy",
    "probe": "liveness",
    "timestamp": "2026-11-10T14:20:00.000000+00:00",
    "uptime_seconds": 4521.8,
    "version": "2.12.0",
    "pid": 12840
  }
  ```
- **HTTP Code:** Always returns `200 OK` unless the process is completely hung.

### 4.2 Readiness Probe (`GET /api/v1/health/ready`)

- **Purpose:** Verifies that the application is ready to accept production farmer traffic.
- **Checks:**
  1. Executes `SELECT 1 AS ready` against the active SQLite/PostgreSQL database.
  2. Asserts that the local storage data directory exists and is accessible.
- **Behavior:** Returns `200 OK` when ready; returns `503 Service Unavailable` during startup or database downtime.

### 4.3 Dependency & Diagnostics Probe (`GET /api/v1/health/deps`)

- **Purpose:** Deep diagnostic probe for operations, monitoring, and administrative consoles.
- **Checks:**
  - Database connectivity status.
  - Circuit breaker states for Open-Meteo weather API and Gnani AI speech services.
  - Real-time request and error summary.

---

## 5. Circuit Breaker Architecture & Resilient Degradation

Implemented in [`backend/app/core/circuit_breaker.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/core/circuit_breaker.py):

### 5.1 State Machine

- **CLOSED (Normal):** All external requests execute normally. Failures increment an internal sliding failure counter.
- **OPEN (Tripped):** When 5 consecutive failures occur within 60 seconds, the circuit trips to `OPEN`. Outgoing calls immediately fast-fail without network roundtrip, preventing worker thread exhaustion.
- **HALF_OPEN (Probing):** After a 30-second recovery timeout, the circuit enters `HALF_OPEN` and permits 2 canary probe requests. If both probes succeed, the circuit resets to `CLOSED`; if any probe fails, it trips back to `OPEN`.

### 5.2 Graceful Degradation Behaviors

- **Open-Meteo Weather Degraded:** When the Open-Meteo circuit trips, AgriGuardian falls back to cached regional seasonal historical normals, allowing crop stage advisory calculations to proceed safely.
- **Gnani Speech AI Degraded:** When the Gnani circuit trips, AgriGuardian falls back to text-based agricultural intelligence in the farmer's preferred language, preventing platform unavailability.

---

## 6. Operational Alerting Thresholds

| Metric | Warning Threshold | Critical Threshold | Recommended Operational Action |
|---|---|---|---|
| `agriguardian_error_rate_pct` | > 2.0% (5m) | > 5.0% (2m) | Inspect structured logs for 500 error stack traces and database locks. |
| `agriguardian_latency_p95_ms` | > 1,000 ms | > 2,500 ms | Check database indexing (`PRAGMA compile_options`), query plans, or CPU limits. |
| `circuit_breaker_state` | State = 1 (HALF_OPEN) | State = 2 (OPEN) | Check upstream provider status pages (Open-Meteo / Gnani AI). |
| Disk Usage (`data/`) | > 75% | > 90% | Run backup rotation or archive historical sync attachments to S3/GCS. |
