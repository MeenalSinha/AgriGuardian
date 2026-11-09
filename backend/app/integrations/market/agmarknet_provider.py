"""AGMARKNET (Directorate of Marketing & Inspection) agricultural market data provider."""

import json
import math
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.integrations.market.base_provider import BaseMarketProvider
from backend.app.integrations.market.exceptions import (
    CommodityNotFoundError,
    MandiNotFoundError,
    MarketAPIError,
)
from backend.app.integrations.market.market_cache import market_cache
from backend.app.integrations.market.market_normalizer import MarketNormalizer
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import (
    MarketDataSource,
    MarketPriceObservation,
    NearbyMandiPrice,
)

logger = get_logger("AgmarknetProvider")


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle straight-line distance between two geographic points in kilometers."""
    r = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)


class AgmarknetProvider(BaseMarketProvider):
    """AGMARKNET integration reading official market bulletins with historical repository and online fallback."""

    def __init__(self, data_file: str = "data/market/official_bulletins.json"):
        self.settings = get_settings()
        # Fallback to historical_prices.json if official_bulletins.json does not exist
        if not os.path.exists(data_file) and os.path.exists("data/market/historical_prices.json"):
            data_file = "data/market/historical_prices.json"
        self.data_file = data_file
        self._loaded_observations: List[MarketPriceObservation] = []
        self._load_local_repository()
        self._ogd_api_base = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    @property
    def provider_name(self) -> str:
        return MarketDataSource.AGMARKNET.value

    def _fetch_ogd_online(self, commodity_name: str, state: Optional[str] = None) -> List[MarketPriceObservation]:
        """Fetch live mandi observations from data.gov.in OGD API when api_key is configured."""
        api_key = getattr(self.settings, "data_gov_in_api_key", None)
        if not api_key:
            return []

        try:
            params: Dict[str, Any] = {
                "api-key": api_key,
                "format": "json",
                "filters[commodity]": commodity_name,
                "limit": 50,
            }
            if state:
                params["filters[state]"] = state

            resp = requests.get(self._ogd_api_base, params=params, timeout=self.settings.market_timeout_seconds)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("records", [])
                observations: List[MarketPriceObservation] = []
                for rec in records:
                    obs = MarketNormalizer.normalize_agmarknet_record(rec)
                    if obs:
                        observations.append(obs)
                return observations
        except Exception as e:
            logger.warning(f"Live OGD API fetch failed: {e}. Falling back to official bulletin repository.")
        return []

    def _load_local_repository(self) -> None:
        """Load and cache validated market observations from durable JSON storage."""
        if not os.path.exists(self.data_file):
            logger.warning(f"Market repository file not found at {self.data_file}")
            return

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            valid_obs = []
            for item in raw_data:
                try:
                    obs = MarketPriceObservation(**item)
                    valid_obs.append(obs)
                except Exception as e:
                    logger.debug(f"Skipping invalid record in local repository: {e}")
            
            # Sort chronologically
            valid_obs.sort(key=lambda x: x.reporting_date)
            self._loaded_observations = valid_obs
            logger.info(f"Loaded {len(self._loaded_observations)} verified observations into AgmarknetProvider.")
        except Exception as e:
            logger.error(f"Error loading market repository {self.data_file}: {e}")

    def get_latest_price(
        self,
        commodity_id: str,
        market_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Optional[MarketPriceObservation]:
        """Fetch the most recent market observation for a commodity."""
        cache_key = f"latest:{commodity_id}:{market_id}:{state}:{district}"
        cached = market_cache.get(cache_key)
        if cached:
            return cached

        # Check commodity existence
        commodity = market_catalog.resolve_commodity(commodity_id)
        if not commodity:
            raise CommodityNotFoundError(f"Unknown commodity query: '{commodity_id}'")

        target_cid = commodity.commodity_id

        # Filter observations
        matches = [obs for obs in self._loaded_observations if obs.commodity_id == target_cid]

        if market_id:
            mandi = market_catalog.get_mandi(market_id)
            if not mandi:
                # Also allow partial search or raw id
                matches = [obs for obs in matches if obs.market_id == market_id or market_id.lower() in obs.market_name.lower()]
            else:
                matches = [obs for obs in matches if obs.market_id == mandi.market_id]
        
        if state:
            matches = [obs for obs in matches if obs.state.lower() == state.lower()]

        if district:
            matches = [obs for obs in matches if obs.district.lower() == district.lower()]

        if not matches:
            return None

        # Return latest by reporting date
        latest = sorted(matches, key=lambda x: x.reporting_date)[-1]
        market_cache.set(cache_key, latest, ttl_seconds=900)
        return latest

    def get_historical_observations(
        self,
        commodity_id: str,
        market_id: str,
        days: int = 90,
    ) -> List[MarketPriceObservation]:
        """Retrieve daily observations for time-series analytics and forecasting."""
        cache_key = f"history:{commodity_id}:{market_id}:{days}"
        cached = market_cache.get(cache_key)
        if cached is not None:
            return cached

        commodity = market_catalog.resolve_commodity(commodity_id)
        if not commodity:
            raise CommodityNotFoundError(f"Commodity '{commodity_id}' not found in catalog.")

        # Find matching observations
        matches = [
            obs for obs in self._loaded_observations
            if obs.commodity_id == commodity.commodity_id
            and (obs.market_id == market_id or market_id.lower() in obs.market_name.lower())
        ]

        if not matches:
            return []

        # Sort chronologically
        matches.sort(key=lambda x: x.reporting_date)
        
        # Take up to the requested number of days
        selected = matches[-days:] if len(matches) > days else matches
        market_cache.set(cache_key, selected, ttl_seconds=1800)
        return selected

    def get_nearby_market_prices(
        self,
        commodity_id: str,
        latitude: float,
        longitude: float,
        max_distance_km: float = 120.0,
    ) -> List[NearbyMandiPrice]:
        """Discover and compare prices across mandis within a straight-line radius."""
        cache_key = f"nearby:{commodity_id}:{latitude:.2f}:{longitude:.2f}:{max_distance_km}"
        cached = market_cache.get(cache_key)
        if cached is not None:
            return cached

        commodity = market_catalog.resolve_commodity(commodity_id)
        if not commodity:
            raise CommodityNotFoundError(f"Commodity '{commodity_id}' not found.")

        target_cid = commodity.commodity_id
        nearby_results: List[NearbyMandiPrice] = []

        # Compare with all registered mandis in catalog
        for mandi in market_catalog.get_all_mandis():
            dist = haversine_distance_km(latitude, longitude, mandi.latitude, mandi.longitude)
            if dist <= max_distance_km:
                # Find latest observation for this commodity at this mandi
                mandi_obs = [
                    obs for obs in self._loaded_observations
                    if obs.commodity_id == target_cid and obs.market_id == mandi.market_id
                ]
                if mandi_obs:
                    latest = sorted(mandi_obs, key=lambda x: x.reporting_date)[-1]
                    
                    # Compute data freshness relative to latest available observation
                    freshness = f"Reported {latest.reporting_date} (Daily APMC Bulletin)"
                    
                    nearby_entry = NearbyMandiPrice(
                        market_id=mandi.market_id,
                        market_name=mandi.market_name,
                        state=mandi.state,
                        district=mandi.district,
                        distance_km=dist,
                        modal_price=latest.modal_price,
                        min_price=latest.min_price,
                        max_price=latest.max_price,
                        variety=latest.variety_name,
                        reporting_date=latest.reporting_date,
                        data_freshness=freshness,
                        arrivals_tonnes=latest.arrivals_tonnes,
                        source=latest.source.value if isinstance(latest.source, MarketDataSource) else str(latest.source),
                        is_live=latest.is_live,
                    )
                    nearby_results.append(nearby_entry)

        # Sort primarily by straight-line distance
        nearby_results.sort(key=lambda x: x.distance_km)
        market_cache.set(cache_key, nearby_results, ttl_seconds=900)
        return nearby_results
