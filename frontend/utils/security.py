"""Frontend XSS mitigation and HTML sanitization utilities for AgriGuardian (Phase 12)."""

import html
from typing import Any, Optional


def safe_text(val: Any) -> str:
    """Escape untrusted user or dynamic database content before interpolation in HTML templates."""
    if val is None:
        return ""
    return html.escape(str(val), quote=True)

