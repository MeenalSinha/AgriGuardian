"""Centralized structured logging for AgriGuardian."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Sensitive patterns to redact in logs
SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|password|bearer|token)\s*[:=]\s*["\']?([^"\'\s,;]+)'),
    re.compile(r'(?i)(X-API-Key-ID)\s*[:=]\s*["\']?([^"\'\s,;]+)'),
]


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts API keys and sensitive credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in SECRET_PATTERNS:
                record.msg = pattern.sub(r'\1: [REDACTED]', record.msg)
        return True


_initialized = False


def setup_logging(log_level: str = "INFO", logs_dir: str = "logs") -> None:
    """Initialize structured application-wide logging."""
    global _initialized
    if _initialized:
        return

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    log_file_path = os.path.join(logs_dir, f"agriguardian_{today_str}.log")

    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sensitive_filter = SensitiveDataFilter()

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

