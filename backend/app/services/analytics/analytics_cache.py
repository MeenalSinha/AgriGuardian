"""Thread-safe in-memory caching for dashboard and analytics metrics."""

import threading
import time
from typing import Any, Dict, Optional, Tuple


class AnalyticsCache:
    """In-memory cache with TTL for aggregated metrics and dashboard overviews."""

    def __init__(self, default_ttl_seconds: int = 180):
        self.default_ttl = default_ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value if not expired."""
        with self._lock:
            if key not in self._store:
                return None
            expiry, value = self._store[key]
            if time.time() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store value with expiration."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl
        with self._lock:
            self._store[key] = (expiry, value)

    def delete(self, key: str) -> None:
        """Invalidate a specific key."""
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidate all keys starting with prefix (e.g. 'farmer:', 'officer:')."""
        with self._lock:
            keys_to_remove = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Flush the entire cache."""
        with self._lock:
            self._store.clear()
