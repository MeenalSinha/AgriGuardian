"""Abstract base class defining the provider interface for agricultural market intelligence."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from backend.app.schemas.market import (
    MarketPriceObservation,
    NearbyMandiPrice,
)


class BaseMarketProvider(ABC):
    """Abstract interface for agricultural market data providers (AGMARKNET, e-NAM, State APMC)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the data provider."""
        pass

    @abstractmethod
    def get_latest_price(
        self,
        commodity_id: str,
        market_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Optional[MarketPriceObservation]:
        """Fetch the latest verified market observation for a commodity."""
        pass

    @abstractmethod
    def get_historical_observations(
        self,
        commodity_id: str,
        market_id: str,
        days: int = 90,
    ) -> List[MarketPriceObservation]:
        """Retrieve sequential historical market observations for time-series analytics and forecasting."""
        pass

    @abstractmethod
    def get_nearby_market_prices(
        self,
        commodity_id: str,
        latitude: float,
        longitude: float,
        max_distance_km: float = 120.0,
    ) -> List[NearbyMandiPrice]:
        """Discover and compare prices across mandis within a geographic radius."""
        pass
