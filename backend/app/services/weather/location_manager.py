"""Location management service with district presets, coordinate validation, and privacy consent handling."""

import math
from typing import Any, Dict, List, Optional
from backend.app.core.logging import get_logger
from backend.app.schemas.location import (
    PRESET_AGRICULTURAL_DISTRICTS,
    LocationPrecision,
    LocationRecord,
)

logger = get_logger("LocationManager")


class LocationManager:
    """Manages farmer geographical context, presets, map clicks, and GPS consent."""

    def __init__(self):
        self.presets = PRESET_AGRICULTURAL_DISTRICTS

    def get_supported_states(self) -> List[str]:
        """Return list of supported Indian agricultural states."""
        return sorted(list(self.presets.keys()))

    def get_districts_for_state(self, state: str) -> List[str]:
        """Return list of supported districts in a given state."""
        if state not in self.presets:
            return []
        return sorted(list(self.presets[state].keys()))

    def resolve_preset_district(
        self,
        state: str,
        district: str,
        is_confirmed: bool = True,
    ) -> LocationRecord:
        """Resolve a preset agricultural district into a LocationRecord.
        
        Args:
            state: Name of Indian state.
            district: Name of district.
            is_confirmed: User confirmation status.
            
        Returns:
            LocationRecord centered on district centroid.
        """
        if state not in self.presets or district not in self.presets[state]:
            raise ValueError(f"District '{district}' in state '{state}' is not in the preset agricultural database.")

        meta = self.presets[state][district]
        slug = f"{state[:3].upper()}-{district[:3].upper()}"
        return LocationRecord(
            location_id=f"LOC-PRESET-{slug}",
            display_name=f"{district}, {state}",
            state=state,
            district=district,
            latitude=meta["lat"],
            longitude=meta["lon"],
            elevation_m=meta.get("elevation", 100.0),
            precision=LocationPrecision.DISTRICT_CENTROID.value,
            source="preset",
            is_confirmed=is_confirmed,
        )

    def resolve_coordinates(
        self,
        latitude: float,
        longitude: float,
        source: str = "manual_pin",
        consent_granted: bool = False,
        label_override: Optional[str] = None,
    ) -> LocationRecord:
        """Resolve arbitrary geographic coordinates from map click or GPS.
        
        Args:
            latitude: Latitude coordinate (-90 to +90).
            longitude: Longitude coordinate (-180 to +180).
            source: Source mechanism ('map_click', 'gps', 'manual').
            consent_granted: Explicit GPS consent flag (mandatory for 'gps' source).
            label_override: Optional custom display label.
            
        Returns:
            LocationRecord with appropriate precision classification.
        """
        if source == "gps" and not consent_granted:
            logger.warning("GPS source requested without explicit user consent. Defaulting to regional district centroid.")
            return self.get_default_location()

        # Find closest preset district within radius to provide contextual naming
        closest_state, closest_district, min_dist_km = self._find_nearest_preset(latitude, longitude)

        precision = LocationPrecision.EXACT_GPS.value if source == "gps" else LocationPrecision.MANUAL_PIN.value
        loc_id = f"LOC-{source.upper()}-{abs(hash((latitude, longitude))) % 100000:05d}"
        
        if label_override:
            display_name = label_override
        elif min_dist_km < 35.0:
            display_name = f"Near {closest_district}, {closest_state} (~{min_dist_km:.1f} km)"
        else:
            display_name = f"Coordinates ({latitude:.3f}°N, {longitude:.3f}°E)"

        return LocationRecord(
            location_id=loc_id,
            display_name=display_name,
            state=closest_state if closest_state else "Unknown State",
            district=closest_district if closest_district else "Custom Area",
            latitude=latitude,
            longitude=longitude,
            elevation_m=self.presets.get(closest_state, {}).get(closest_district, {}).get("elevation", 100.0),
            precision=precision,
            source=source,
            is_confirmed=True,
        )

    def get_default_location(self) -> LocationRecord:
        """Return the default baseline location (Varanasi, UP) with explicit approximate label."""
        return self.resolve_preset_district(state="Uttar Pradesh", district="Varanasi", is_confirmed=False)

    def _find_nearest_preset(self, lat: float, lon: float) -> (Optional[str], Optional[str], float):
        """Find the nearest district centroid using haversine formula."""
        best_state = None
        best_district = None
        min_distance = float("inf")

        for state, districts in self.presets.items():
            for dist_name, coords in districts.items():
                d = self._haversine_km(lat, lon, coords["lat"], coords["lon"])
                if d < min_distance:
                    min_distance = d
                    best_state = state
                    best_district = dist_name

        return best_state, best_district, min_distance

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in km."""
        r = 6371.0  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return 2.0 * r * math.asin(math.sqrt(a))
