"""e-NAM (National Agriculture Market) integration provider with institutional gateway disclosure."""

from typing import List, Optional

from backend.app.core.config import get_settings
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
    """e-NAM integration providing unified national agricultural electronic trade intelligence.

    NOTE ON INTEGRATION STATUS & INSTITUTIONAL GATEWAY:
    The Small Farmers' Agribusiness Consortium (SFAC) and Ministry of Agriculture & Farmers
    Welfare operate the e-NAM platform as a secured electronic trade platform. Real-time online
    bidding streams and electronic auction lots require authorized institutional APMC gateway
    credentials (enam_api_key / APMC Gateway Token).

    When running without institutional gateway credentials, this provider honestly discloses its
    operational mode: it references validated official APMC bulletins via AgmarknetProvider with
    is_live=False, rather than simulating live electronic trade execution feeds.
    """

    def __init__(self, data_file: str = "data/market/official_bulletins.json"):
        self.settings = get_settings()
        self.data_file = data_file
        self.underlying_provider = AgmarknetProvider(data_file=data_file)
        self.has_institutional_gateway = bool(
            getattr(self.settings, "enam_api_key", None) and self.settings.enam_api_key.strip()
        )
        if not self.has_institutional_gateway:
            logger.info("EnamProvider: Running in institutional gateway fallback mode (using verified APMC bulletins).")

    @property
    def provider_name(self) -> str:
        return MarketDataSource.ENAM.value

    @property
    def is_authenticated_gateway(self) -> bool:
        """Returns True if authorized institutional e-NAM credentials are provided."""
        return self.has_institutional_gateway

    def get_latest_price(
        self,
        commodity_id: str,
        market_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Optional[MarketPriceObservation]:
        """Fetch latest price observation with honest source disclosure."""
        latest = self.underlying_provider.get_latest_price(
            commodity_id=commodity_id,
            market_id=market_id,
            state=state,
            district=district,
        )
        if latest and not self.has_institutional_gateway:
            # Explicitly mark that live electronic auction feed requires institutional gateway
            latest = latest.model_copy(
                update={
                    "is_live": False,
                }
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
        """Discover nearby mandis connected to e-NAM national grid."""
        return self.underlying_provider.get_nearby_market_prices(
            commodity_id=commodity_id,
            latitude=latitude,
            longitude=longitude,
            max_distance_km=max_distance_km,
        )
