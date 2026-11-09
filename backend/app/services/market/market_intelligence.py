"""High-level agricultural market intelligence orchestration service."""

from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.market.agmarknet_provider import AgmarknetProvider
from backend.app.integrations.market.msp_provider import MSPProvider
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import (
    MarketPriceObservation,
    MarketTrendAnalysis,
    NearbyMandiPrice,
    PriceForecastResult,
)
from backend.app.services.market.market_alert_manager import MarketAlertManager
from backend.app.services.market.market_analytics import MarketAnalyticsService
from backend.app.services.market.nearby_mandi_service import NearbyMandiService
from backend.app.services.market.net_realization_calculator import NetRealizationCalculator
from backend.app.services.market.price_forecaster import PriceForecaster

logger = get_logger("MarketIntelligenceService")


class MarketIntelligenceService:
    """Unified service providing verified prices, trends, MSP comparisons, and forecasts."""

    def __init__(self):
        self.catalog = market_catalog
        self.provider = AgmarknetProvider()
        self.msp_provider = MSPProvider()
        self.nearby_service = NearbyMandiService(provider=self.provider)
        self.forecaster = PriceForecaster()
        self.alert_manager = MarketAlertManager()

    def get_market_summary(
        self,
        commodity_query: str,
        market_id: Optional[str] = None,
        location_query: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        history_days: int = 90,
        forecast_horizon_days: int = 14,
    ) -> Dict[str, Any]:
        """Generate comprehensive market intelligence summary for a commodity and region."""
        commodity = self.catalog.resolve_commodity(commodity_query)
        if not commodity:
            return {
                "is_available": False,
                "error": f"Commodity '{commodity_query}' not recognized in agricultural market catalog.",
                "commodity_query": commodity_query,
            }

        # 1. Nearby Mandi Discovery
        nearby_mandis = self.nearby_service.discover_nearby_prices(
            commodity_id=commodity.commodity_id,
            latitude=latitude,
            longitude=longitude,
            market_id=market_id,
            location_query=location_query,
            radius_km=120.0,
        )

        # 2. Select target mandi (explicit market_id, or nearest discovered mandi, or default)
        selected_mandi_id = market_id
        if not selected_mandi_id and nearby_mandis:
            selected_mandi_id = nearby_mandis[0].market_id

        # 3. Latest verified observation
        latest_obs = self.provider.get_latest_price(
            commodity_id=commodity.commodity_id,
            market_id=selected_mandi_id,
        )

        # 4. Historical observations & Trend Analysis
        historical_obs = []
        trend_analysis: Optional[MarketTrendAnalysis] = None
        if latest_obs:
            historical_obs = self.provider.get_historical_observations(
                commodity_id=commodity.commodity_id,
                market_id=latest_obs.market_id,
                days=history_days,
            )
            if historical_obs:
                trend_analysis = MarketAnalyticsService.analyze_trend(
                    observations=historical_obs,
                    window_days=min(len(historical_obs), 30),
                )

        # 5. MSP Comparison
        msp_comparison = None
        if latest_obs:
            msp_comparison = self.msp_provider.compare_price_with_msp(
                commodity_query=commodity.commodity_id,
                current_modal_price=latest_obs.modal_price,
            )

        # 6. Price Forecast
        forecast_result: Optional[PriceForecastResult] = None
        if len(historical_obs) >= PriceForecaster.MIN_OBSERVATIONS_REQUIRED:
            try:
                forecast_result = self.forecaster.forecast_prices(
                    observations=historical_obs,
                    horizon_days=forecast_horizon_days,
                )
            except Exception as e:
                logger.warning(f"Forecasting skipped: {e}")

        # 7. Evaluate and retrieve alerts
        if latest_obs:
            msp_val = msp_comparison.get("msp_price") if msp_comparison and msp_comparison.get("has_msp") else None
            self.alert_manager.evaluate_observation(
                latest_obs=latest_obs,
                trend=trend_analysis,
                msp_price=msp_val,
            )

        active_alerts = self.alert_manager.get_active_alerts(
            commodity_id=commodity.commodity_id,
            market_id=latest_obs.market_id if latest_obs else None,
        )

        return {
            "is_available": latest_obs is not None,
            "commodity": commodity.model_dump(),
            "latest_observation": latest_obs.model_dump() if latest_obs else None,
            "msp_comparison": msp_comparison,
            "trend_analysis": trend_analysis.model_dump() if trend_analysis else None,
            "nearby_mandis": [m.model_dump() for m in nearby_mandis],
            "forecast": forecast_result.model_dump() if forecast_result else None,
            "active_alerts": [a.model_dump() for a in active_alerts],
            "data_source_disclosure": (
                "Market prices retrieved from official AGMARKNET daily bulletins & e-NAM trade records. "
                "MSP policy benchmarks cited from Ministry of Agriculture & Farmers Welfare notifications."
            ),
        }

    def get_market_context_for_advisory(
        self,
        crop_name: str,
        location_query: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Lightweight context generator consumed by Advisory Orchestrator and Conversation Engine."""
        commodity = self.catalog.resolve_commodity(crop_name)
        if not commodity:
            return {
                "is_available": False,
                "crop": crop_name,
                "summary": f"No official APMC market data available for '{crop_name}'.",
            }

        summary = self.get_market_summary(
            commodity_query=commodity.commodity_id,
            location_query=location_query,
            latitude=latitude,
            longitude=longitude,
            history_days=30,
            forecast_horizon_days=7,
        )

        if not summary.get("is_available") or not summary.get("latest_observation"):
            return {
                "is_available": False,
                "crop": commodity.canonical_name,
                "summary": f"APMC market data for {commodity.canonical_name} currently unavailable in selected district.",
            }

        latest = summary["latest_observation"]
        msp_info = summary.get("msp_comparison")
        trend = summary.get("trend_analysis")
        nearby = summary.get("nearby_mandis", [])

        # Construct clear natural language summary for voice and chat
        parts = [
            f"{latest['commodity_name']} is trading at a modal price of ₹{latest['modal_price']:,.2f}/quintal "
            f"in {latest['market_name']} (Range: ₹{latest['min_price']:,.2f} - ₹{latest['max_price']:,.2f}/qtl, "
            f"Reported {latest['reporting_date']})."
        ]

        if trend and trend.get("trend_direction"):
            td_val = trend["trend_direction"]
            td_str = (td_val.value if hasattr(td_val, "value") else str(td_val)).lower().replace("pricetrenddirection.", "")
            parts.append(f"Price trend over last 30 days is {td_str}. {trend.get('trend_explanation', '')}")

        if msp_info and msp_info.get("has_msp"):
            parts.append(msp_info.get("advisory", ""))
        elif msp_info and not msp_info.get("has_msp") and msp_info.get("message"):
            parts.append(msp_info.get("message", ""))

        if nearby and len(nearby) > 1:
            alt = nearby[1]
            diff = alt['modal_price'] - latest['modal_price']
            if diff > 0:
                parts.append(
                    f"Nearby alternative: {alt['market_name']} ({alt['distance_km']} km away) is quoting ₹{alt['modal_price']:,.2f}/quintal (+₹{diff:,.2f}/qtl higher)."
                )

        trend_clean = None
        if trend and trend.get("trend_direction"):
            t_val = trend["trend_direction"]
            trend_clean = (t_val.value if hasattr(t_val, "value") else str(t_val)).lower().replace("pricetrenddirection.", "")

        return {
            "is_available": True,
            "crop": commodity.canonical_name,
            "market_name": latest["market_name"],
            "state": latest["state"],
            "modal_price": latest["modal_price"],
            "min_price": latest["min_price"],
            "max_price": latest["max_price"],
            "reporting_date": latest["reporting_date"],
            "source": latest["source"],
            "has_msp": msp_info.get("has_msp", False) if msp_info else False,
            "msp_price": msp_info.get("msp_price") if msp_info else None,
            "trend_direction": trend_clean,
            "summary": " ".join(parts),
            "nearby_options_count": len(nearby),
        }
