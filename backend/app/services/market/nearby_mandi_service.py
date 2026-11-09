"""Location-aware nearby APMC mandi discovery and spatial price comparison service."""

from typing import List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.market.agmarknet_provider import (
    AgmarknetProvider,
    haversine_distance_km,
)
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import NearbyMandiPrice
from backend.app.services.weather.location_manager import LocationManager

logger = get_logger("NearbyMandiService")


class NearbyMandiService:
    """Discovers surrounding APMC markets and compares trading rates using verified coordinates."""

    def __init__(self, provider: Optional[AgmarknetProvider] = None):
        self.provider = provider or AgmarknetProvider()
        self.location_manager = LocationManager()

    def discover_nearby_prices(
        self,
        commodity_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        market_id: Optional[str] = None,
        location_query: Optional[str] = None,
        radius_km: float = 120.0,
    ) -> List[NearbyMandiPrice]:
        """Discover APMC mandis trading a commodity within radius_km.
        
        Args:
            commodity_id: Commodity identifier or alias.
            latitude: Farmer GPS latitude.
            longitude: Farmer GPS longitude.
            market_id: APMC market ID to center search around.
            location_query: District or city query string if GPS not provided.
            radius_km: Maximum search radius in straight-line kilometers.
            
        Returns:
            List of NearbyMandiPrice sorted primarily by distance.
        """
        # Resolve coordinates if not provided directly
        lat = latitude
        lon = longitude

        if (lat is None or lon is None) and market_id:
            mandi = market_catalog.get_mandi(market_id)
            if mandi:
                lat, lon = mandi.latitude, mandi.longitude

        if (lat is None or lon is None) and location_query:
            # Check market catalog first for exact mandi/district match
            for m in market_catalog.get_all_mandis():
                if location_query.lower() in m.market_name.lower() or location_query.lower() in m.district.lower():
                    lat, lon = m.latitude, m.longitude
                    break

            if lat is None or lon is None:
                for state_name, dists in self.location_manager.presets.items():
                    for dist_name, meta in dists.items():
                        if dist_name.lower() in location_query.lower() or location_query.lower() in dist_name.lower():
                            lat, lon = meta["lat"], meta["lon"]
                            break
                    if lat is not None:
                        break

        if lat is None or lon is None:
            # Fallback to default agricultural location
            default_loc = self.location_manager.get_default_location()
            lat, lon = default_loc.latitude, default_loc.longitude

        # Query provider
        nearby = self.provider.get_nearby_market_prices(
            commodity_id=commodity_id,
            latitude=lat,
            longitude=lon,
            max_distance_km=radius_km,
        )

        return nearby
