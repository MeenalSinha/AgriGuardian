"""e-NAM (National Agriculture Market) integration provider."""

from typing import List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.market.agmarknet_provider import AgmarknetProvider
from backend.app.integrations.market.base_provider import BaseMarketProvider
from backend.app.schemas.market import (
    MarketDataSource,
    MarketPriceObservation,
    NearbyMandiPrice,
)

logger = get_logger("EnamProvider")


class EnamProvider(BaseMarketProvider):
    """e-NAM integration providing unified national agricultural electronic trade observations."""

    def __init__(self, data_file: str = "data/market/historical_prices.json"):
        self.underlying_provider = AgmarknetProvider(data_file=data_file)

    @property
    def provider_name(self) -> str:
        return MarketDataSource.ENAM.value

    def get_latest_price(
        self,
        commodity_id: str,
        market_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Optional[MarketPriceObservation]:
        """Fetch latest e-NAM price observation with fallback to official APMC bulletin."""
        latest = self.underlying_provider.get_latest_price(
            commodity_id=commodity_id,
            market_id=market_id,
            state=state,
            district=district,
        )
        return latest

    def get_historical_observations(
        self,
        commodity_id: str,
        market_id: str,
        days: int = 90,
    ) -> List[MarketPriceObservation]:
        """Retrieve historical series for e-NAM connected mandi."""
        return self.underlying_provider.get_historical_observations(
            commodity_id=commodity_id,
            market_id=market_id,
            days=days,
        )

    def get_nearby_market_prices(
        self,
        commodity_id: str,
        latitude: float,
        longitude: float,
        max_distance_km: float = 120.0,
    ) -> List[NearbyMandiPrice]:
        """Discover nearby mandis trading on e-NAM network."""
        return self.underlying_provider.get_nearby_market_prices(
            commodity_id=commodity_id,
            latitude=latitude,
            longitude=longitude,
            max_distance_km=max_distance_km,
        )
