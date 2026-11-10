"""Centralized structured logging with sensitive data redaction and correlation IDs for AgriGuardian (Phase 12)."""

import contextvars
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Thread/asyncio context variable for request correlation ID
correlation_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current async context."""
    correlation_id_ctx.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Retrieve correlation ID for current context."""
    return correlation_id_ctx.get()


# Sensitive patterns to redact in logs
SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|password|bearer|token)\s*[:=]\s*["\']?([^"\'\s,;]+)'),
    re.compile(r'(?i)(X-API-Key-ID)\s*[:=]\s*["\']?([^"\'\s,;]+)'),
    re.compile(r'(?i)"(password|access_token|refresh_token|jwt_secret_key|secret)"\s*:\s*"[^"]+"'),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts credentials, passwords, and sensitive keys from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in SECRET_PATTERNS:
                record.msg = pattern.sub(r'\1: [REDACTED]', record.msg)
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production log ingestion and observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


_initialized = False


def setup_logging(log_level: str = "INFO", logs_dir: str = "logs", json_format: bool = False) -> None:
    """Initialize structured application-wide logging."""
    global _initialized
    if _initialized:
        return

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    log_file_path = os.path.join(logs_dir, f"agriguardian_{today_str}.log")

    level = getattr(logging, log_level.upper(), logging.INFO)

    sensitive_filter = SensitiveDataFilter()

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Root AgriGuardian logger
    root_logger = logging.getLogger("AgriGuardian")
    root_logger.setLevel(level)

    # Avoid duplicate handlers if re-called
    if not root_logger.handlers:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        root_logger.addHandler(console_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get or create a named sub-logger under AgriGuardian namespace."""
    if not name.startswith("AgriGuardian"):
        logger_name = f"AgriGuardian.{name}"
    else:
        logger_name = name
    return logging.getLogger(logger_name)
