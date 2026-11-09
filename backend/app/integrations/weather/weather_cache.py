"""In-memory TTL weather cache with spatial grid discretization."""

import math
import time
from typing import Any, Dict, Optional, Tuple


class WeatherCache:
    """Thread-safe in-memory cache for weather observations, forecasts, and reanalysis.
    
    Coordinates are rounded to a spatial grid (~0.05 degrees ~= 5.5 km) to prevent
    redundant network requests for slight GPS fluctuations while respecting local microclimates.
    """

    def __init__(self, default_ttl_seconds: int = 1800, spatial_grid_deg: float = 0.05):
        """Initialize weather cache.
        
        Args:
            default_ttl_seconds: Cache duration in seconds (default 30 mins).
            spatial_grid_deg: Coordinate rounding increment (default 0.05 deg).
        """
        self._cache: Dict[Tuple[float, float, str], Tuple[float, Any]] = {}
        self.default_ttl = default_ttl_seconds
        self.spatial_grid = spatial_grid_deg
        self._hits = 0
        self._misses = 0

    def _discretize_coordinate(self, coord: float) -> float:
        """Snap coordinate to nearest grid point."""
        return round(coord / self.spatial_grid) * self.spatial_grid

    def _make_key(self, lat: float, lon: float, cache_type: str) -> Tuple[float, float, str]:
        """Generate normalized cache key."""
        d_lat = round(self._discretize_coordinate(lat), 4)
        d_lon = round(self._discretize_coordinate(lon), 4)
        return (d_lat, d_lon, cache_type)

    def get(self, lat: float, lon: float, cache_type: str = "forecast") -> Optional[Any]:
        """Retrieve cached entry if not expired."""
        key = self._make_key(lat, lon, cache_type)
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() <= ts:
                self._hits += 1
                return val
            else:
                # Expired
                del self._cache[key]
        self._misses += 1
        return None

    def set(
        self,
        lat: float,
        lon: float,
        data: Any,
        cache_type: str = "forecast",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store entry with TTL."""
        key = self._make_key(lat, lon, cache_type)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expire_at = time.time() + ttl
        self._cache[key] = (expire_at, data)

    def invalidate(self, lat: Optional[float] = None, lon: Optional[float] = None) -> int:
        """Clear cache entries, optionally filtered by coordinate."""
        if lat is None or lon is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        
        d_lat = round(self._discretize_coordinate(lat), 4)
        d_lon = round(self._discretize_coordinate(lon), 4)
        keys_to_del = [k for k in self._cache.keys() if k[0] == d_lat and k[1] == d_lon]
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


# Global singleton instance for app-wide weather caching
weather_cache = WeatherCache(default_ttl_seconds=1800, spatial_grid_deg=0.05)
