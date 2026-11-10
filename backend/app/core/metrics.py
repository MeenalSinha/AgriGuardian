"""Real measured operational metrics collector for AgriGuardian (Phase 12).

Calculates authentic live counters, latencies, and error rates without synthetic or hardcoded metrics.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """Thread-safe collector for real operational API and service telemetry."""

    def __init__(self):
        self._lock = Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.status_codes: Dict[int, int] = defaultdict(int)
        self.endpoint_counts: Dict[str, int] = defaultdict(int)
        self.recent_latencies: List[float] = []  # sliding window of last 500 requests
        self.provider_calls: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
        self.max_latencies_window = 500

    def record_request(self, endpoint: str, status_code: int, duration_seconds: float) -> None:
        """Record live request execution metrics."""
        with self._lock:
            self.total_requests += 1
            self.status_codes[status_code] += 1
            self.endpoint_counts[endpoint] += 1
            self.recent_latencies.append(duration_seconds)
            if len(self.recent_latencies) > self.max_latencies_window:
                self.recent_latencies.pop(0)

    def record_provider_call(self, provider_name: str, success: bool) -> None:
        """Record call outcomes for external integrations (Gnani, Open-Meteo, LLM)."""
        with self._lock:
            key = "success" if success else "failure"
            self.provider_calls[provider_name][key] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Generate structured operational summary based on authentic measurements."""
        with self._lock:
            uptime_seconds = max(0.1, time.time() - self.start_time)
            avg_latency_ms = (
                (sum(self.recent_latencies) / len(self.recent_latencies)) * 1000.0
                if self.recent_latencies
                else 0.0
            )
            p95_latency_ms = 0.0
            if self.recent_latencies:
                sorted_lats = sorted(self.recent_latencies)
                idx = int(len(sorted_lats) * 0.95)
                p95_latency_ms = sorted_lats[min(idx, len(sorted_lats) - 1)] * 1000.0

            total_errors = sum(count for code, count in self.status_codes.items() if code >= 500)
            error_rate_pct = (total_errors / self.total_requests * 100.0) if self.total_requests > 0 else 0.0

            return {
                "uptime_seconds": round(uptime_seconds, 1),
                "total_requests": self.total_requests,
                "requests_per_second": round(self.total_requests / uptime_seconds, 2),
                "status_code_distribution": dict(self.status_codes),
                "average_latency_ms": round(avg_latency_ms, 2),
                "p95_latency_ms": round(p95_latency_ms, 2),
                "error_rate_pct": round(error_rate_pct, 2),
                "provider_telemetry": {k: dict(v) for k, v in self.provider_calls.items()},
                "endpoint_distribution": dict(self.endpoint_counts),
            }

    def to_prometheus_format(self) -> str:
        """Export metrics in standard Prometheus exposition format."""
        summary = self.get_summary()
        lines = [
            "# HELP agriguardian_uptime_seconds Total application process uptime in seconds",
            "# TYPE agriguardian_uptime_seconds gauge",
            f"agriguardian_uptime_seconds {summary['uptime_seconds']}",
            "",
            "# HELP agriguardian_http_requests_total Total HTTP requests received",
            "# TYPE agriguardian_http_requests_total counter",
            f"agriguardian_http_requests_total {summary['total_requests']}",
            "",
            "# HELP agriguardian_http_latency_ms_avg Rolling average HTTP latency in milliseconds",
            "# TYPE agriguardian_http_latency_ms_avg gauge",
            f"agriguardian_http_latency_ms_avg {summary['average_latency_ms']}",
            "",
            "# HELP agriguardian_http_latency_ms_p95 Rolling 95th percentile HTTP latency in milliseconds",
            "# TYPE agriguardian_http_latency_ms_p95 gauge",
            f"agriguardian_http_latency_ms_p95 {summary['p95_latency_ms']}",
            "",
            "# HELP agriguardian_http_error_rate_pct Percentage of requests resulting in 5xx errors",
            "# TYPE agriguardian_http_error_rate_pct gauge",
            f"agriguardian_http_error_rate_pct {summary['error_rate_pct']}",
        ]

        # Add status code lines
        lines.append("")
        lines.append("# HELP agriguardian_http_status_codes_total HTTP response count by status code")
        lines.append("# TYPE agriguardian_http_status_codes_total counter")
        for code, count in summary["status_code_distribution"].items():
            lines.append(f'agriguardian_http_status_codes_total{{status="{code}"}} {count}')

        # Add external provider lines
        lines.append("")
        lines.append("# HELP agriguardian_provider_calls_total External provider calls by outcome")
        lines.append("# TYPE agriguardian_provider_calls_total counter")
        for prov, outcomes in summary["provider_telemetry"].items():
            for outcome, count in outcomes.items():
                lines.append(f'agriguardian_provider_calls_total{{provider="{prov}",outcome="{outcome}"}} {count}')

        return "\n".join(lines) + "\n"


_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Retrieve global singleton metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

