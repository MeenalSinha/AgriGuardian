"""Agricultural Market Intelligence Service with verified APMC grounding and source integrity."""

from datetime import datetime
from typing import Any, Dict, Optional

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.advisory import MarketContext
from backend.app.services.market.market_intelligence import MarketIntelligenceService

logger = get_logger("MarketService")


class MarketService:
    """Service providing transparent agricultural market pricing intelligence backed by verified APMC bulletins."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.intelligence_service = MarketIntelligenceService()

    def get_market_data(
        self,
        crop: str,
        state: Optional[str] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> MarketContext:
        """Retrieve verified APMC market prices for specified crop with strict source attribution.

        Args:
            crop: Crop name (e.g. 'Wheat', 'Tomato', 'Paddy').
            state: Target agricultural state.
            location: District or city query string.
            latitude: GPS latitude if available.
            longitude: GPS longitude if available.

        Returns:
            MarketContext with verified prices, trends, and authoritative source attribution.
        """
        # Never silently attribute missing location to a hardcoded city like 'Meerut'
        if not location and not state and (latitude is None or longitude is None):
            logger.info(f"Market data requested without location for crop='{crop}'")
            return MarketContext(
                crop=crop.capitalize() if crop else "General",
                current_price=None,
                trend="location_required",
                min_price=None,
                max_price=None,
                last_week_price=None,
                price_change=None,
                change_percent=None,
                forecast=None,
                demand=None,
                insight="Location required for local APMC mandi price discovery. Please select or specify your district/state.",
                summary="Farmer location is required to discover local APMC mandi prices and transportation logistics.",
                source="AGMARKNET / e-NAM",
                is_live=False,
                is_available=False,
                timestamp=datetime.now().isoformat(),
            )

        location_query = location or state or ""
        market_ctx = self.intelligence_service.get_market_context_for_advisory(
            crop_name=crop,
            location_query=location_query,
            latitude=latitude,
            longitude=longitude,
        )

        if not market_ctx.get("is_available"):
            logger.info(f"Market data unavailable for crop='{crop}', location='{location_query}'")
            return MarketContext(
                crop=crop.capitalize() if crop else "General",
                current_price=None,
                trend="insufficient_data",
                min_price=None,
                max_price=None,
                last_week_price=None,
                price_change=None,
                change_percent=None,
                forecast=None,
                demand=None,
                insight="No verified APMC market price bulletin currently available for this commodity in the selected region.",
                summary=f"Market data temporarily unavailable for {crop}.",
                source="AGMARKNET / e-NAM",
                is_live=False,
                is_available=False,
                timestamp=datetime.now().isoformat(),
            )

        modal_p = market_ctx["modal_price"]
        min_p = market_ctx["min_price"]
        max_p = market_ctx["max_price"]
        trend_val = market_ctx.get("trend_direction")
        trend_str = (trend_val.value if hasattr(trend_val, "value") else str(trend_val or "stable")).lower()
        if "pricetrenddirection." in trend_str:
            trend_str = trend_str.replace("pricetrenddirection.", "")

        insight = (
            f"Trading at ₹{modal_p:,.2f}/quintal in {market_ctx['market_name']} "
            f"(Range: ₹{min_p:,.2f} - ₹{max_p:,.2f}/qtl)."
        )
        if market_ctx.get("has_msp") and market_ctx.get("msp_price"):
            insight += f" Statutory MSP benchmark: ₹{market_ctx['msp_price']:,.2f}/qtl."

        return MarketContext(
            crop=market_ctx["crop"],
            current_price=modal_p,
            trend=trend_str,
            min_price=min_p,
            max_price=max_p,
            last_week_price=None,
            price_change=None,
            change_percent=None,
            forecast=trend_str,
            demand="active",
            insight=insight,
            summary=market_ctx["summary"],
            source=market_ctx["source"],
            is_live=False,
            is_available=True,
            mandi_name=market_ctx["market_name"],
            has_msp=market_ctx.get("has_msp", False),
            msp_price=market_ctx.get("msp_price"),
            timestamp=datetime.now().isoformat(),
        )
