"""Base HTTP client for Gnani AI communication with retries, timeout, and auth."""

import time
from typing import Any, Dict, Optional, Tuple, Union
import httpx
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import (
    GnaniAPIError,
    GnaniAuthError,
    GnaniRateLimitError,
    GnaniTimeoutError,
)
from backend.app.core.logging import get_logger

logger = get_logger("GnaniClient")


class GnaniBaseClient:
    """Base client managing HTTP communication with Gnani AI APIs."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.gnani_api_base_url.rstrip("/")
        self.api_key = self.settings.gnani_api_key

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct required Gnani authentication and request headers."""
        if not self.settings.is_gnani_configured:
            raise GnaniAuthError(
                "Gnani API key is not configured. Please set GNANI_API_KEY in your .env file or environment.",
                status_code=401,
            )

        headers = {
            "X-API-Key-ID": self.api_key,
        }
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def _execute_request(
        self,
        method: str,
        endpoint: str,
        timeout: float,
        max_retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute HTTP request with timeout, exponential backoff retries, and error mapping."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self._get_headers(headers)

        last_exception = None
        backoff_delay = 1.0

        for attempt in range(1, max_retries + 2):
            try:
                logger.debug(f"Calling Gnani API: {method} {url} (Attempt {attempt}/{max_retries + 1})")
                with httpx.Client(timeout=timeout) as client:
                    response = client.request(
                        method=method,
                        url=url,
                        headers=request_headers,
                        data=data,
                        files=files,
                        json=json_body,
                    )

                # Check HTTP error status codes
                if response.status_code in (401, 403):
                    logger.error(f"Gnani authentication failed: HTTP {response.status_code}")
                    raise GnaniAuthError(
                        f"Gnani authentication failed (HTTP {response.status_code}). Verify GNANI_API_KEY.",
                        status_code=response.status_code,
                        details={"response_text": response.text[:200]},
                    )

                if response.status_code == 429:
                    logger.warning(f"Gnani rate limit reached: HTTP 429")
                    if attempt <= max_retries:
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    raise GnaniRateLimitError(
                        "Gnani API rate limit exceeded. Please try again shortly.",
                        status_code=429,
                    )

                if response.status_code >= 500:
                    logger.warning(f"Gnani server error HTTP {response.status_code} on attempt {attempt}")
                    if attempt <= max_retries:
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    raise GnaniAPIError(
                        f"Gnani server error (HTTP {response.status_code}): {response.text[:200]}",
                        status_code=response.status_code,
                    )

                if response.status_code >= 400:
                    logger.error(f"Gnani client error HTTP {response.status_code}: {response.text[:200]}")
                    raise GnaniAPIError(
                        f"Gnani request error (HTTP {response.status_code}): {response.text[:200]}",
                        status_code=response.status_code,
                    )

                return response

            except httpx.TimeoutException as e:
                logger.warning(f"Gnani request timed out on attempt {attempt}: {e}")
                last_exception = GnaniTimeoutError(
                    f"Gnani request timed out after {timeout}s.",
                    details={"attempt": attempt},
                )
                if attempt <= max_retries:
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                    continue
                raise last_exception

            except httpx.RequestError as e:
                logger.warning(f"Gnani network connection error on attempt {attempt}: {e}")
                last_exception = GnaniAPIError(
                    f"Failed to connect to Gnani API ({url}): {e}",
                    details={"attempt": attempt},
                )
                if attempt <= max_retries:
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                    continue
                raise last_exception

        if last_exception:
            raise last_exception
        raise GnaniAPIError("Unexpected failure executing Gnani request.")

