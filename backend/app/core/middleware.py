"""Security, correlation ID, and payload size ASGI middlewares for AgriGuardian (Phase 12).

Implemented as pure ASGI middleware to eliminate Starlette BaseHTTPMiddleware
anyio taskgroup deadlock hazards under Python 3.12/3.14.
"""

import json
import time
import uuid

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, set_correlation_id
from backend.app.core.metrics import get_metrics_collector

logger = get_logger("Middleware")


class CorrelationIdMiddleware:
    """Pure ASGI middleware assigning or propagating request correlation IDs and recording live telemetry."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract X-Request-ID from scope headers
        req_id = None
        for k, v in scope.get("headers", []):
            if k.lower() == b"x-request-id":
                req_id = v.decode("latin1", errors="replace")
                break
        if not req_id:
            req_id = f"req_{uuid.uuid4().hex[:12]}"

        set_correlation_id(req_id)
        start_time = time.time()
        status_code = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", req_id.encode("latin1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            get_metrics_collector().record_request(scope.get("path", "/"), status_code[0], duration)


class SecurityHeadersMiddleware:
    """Pure ASGI middleware enforcing essential defensive HTTP security headers (OWASP guidelines)."""

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                headers.append((b"permissions-policy", b"camera=(self), microphone=(self), geolocation=(self)"))
                headers.append((b"content-security-policy", b"default-src 'self'; img-src 'self' data: blob: https:; media-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self' https:;"))
                env = (self.settings.app_env or "development").strip().lower()
                if env in ("production", "prod", "staging"):
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains; preload"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware enforcing maximum allowed request body size to mitigate DoS attacks."""

    def __init__(self, app, max_bytes: int = 15 * 1024 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for k, v in scope.get("headers", []):
                if k.lower() == b"content-length":
                    try:
                        length = int(v.decode("latin1", errors="replace"))
                        if length > self.max_bytes:
                            response_body = json.dumps({
                                "detail": f"Payload Too Large: Request body ({length} bytes) exceeds the maximum allowed limit ({self.max_bytes} bytes)."
                            }).encode("utf-8")
                            await send({
                                "type": "http.response.start",
                                "status": 413,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(response_body)).encode("ascii")),
                                ],
                            })
                            await send({
                                "type": "http.response.body",
                                "body": response_body,
                            })
                            return
                    except ValueError:
                        pass
        await self.app(scope, receive, send)
