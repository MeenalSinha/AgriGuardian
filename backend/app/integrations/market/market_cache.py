"""In-memory TTL cache for market price observations and analytical payloads."""

import time
from typing import Any, Dict, Optional, Tuple


class MarketCache:
    """Thread-safe TTL caching for market observations, mandi comparisons, and forecasts."""

    def __init__(self, default_ttl_seconds: int = 1800):
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached payload if valid and not expired."""
        if key in self._cache:
            expire_at, value = self._cache[key]
            if time.time() <= expire_at:
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store payload with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expire_at = time.time() + ttl
        self._cache[key] = (expire_at, value)

    def invalidate(self, prefix: Optional[str] = None) -> int:
        """Clear cache entries, optionally matching a key prefix."""
        if prefix is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        keys_to_del = [k for k in self._cache.keys() if k.startswith(prefix)]
        for k in keys_to_del:
            del self._cache[k]
        return len(keys_to_del)

    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        hit_ratio = (self._hits / total) if total > 0 else 0.0
        return {
            "entries_count": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_ratio": round(hit_ratio, 3),
        }


# Global singleton instance for application-wide market caching
market_cache = MarketCache(default_ttl_seconds=1800)
