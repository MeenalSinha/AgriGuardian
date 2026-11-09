"""Agricultural Market Intelligence Service with source attribution and integrity."""

from datetime import datetime
from typing import Dict, Optional, Any
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.advisory import MarketContext

logger = get_logger("MarketIntegration")

# Indicative wholesale reference prices (Source: Agmarknet Mandi Benchmarks)
INDICATIVE_MARKET_BENCHMARKS = {
    "wheat": {"current": 2150.0, "trend": "stable", "min": 2000.0, "max": 2250.0, "last_week": 2100.0, "forecast": "stable", "demand": "high"},
    "rice": {"current": 2850.0, "trend": "rising", "min": 2600.0, "max": 3050.0, "last_week": 2750.0, "forecast": "rising", "demand": "very_high"},
    "cotton": {"current": 6600.0, "trend": "falling", "min": 6100.0, "max": 7100.0, "last_week": 6800.0, "forecast": "stable", "demand": "medium"},
    "sugarcane": {"current": 3150.0, "trend": "stable", "min": 2950.0, "max": 3250.0, "last_week": 3100.0, "forecast": "stable", "demand": "high"},
    "potato": {"current": 1850.0, "trend": "rising", "min": 1550.0, "max": 2050.0, "last_week": 1700.0, "forecast": "rising", "demand": "high"},
    "tomato": {"current": 2600.0, "trend": "rising", "min": 2100.0, "max": 3100.0, "last_week": 2300.0, "forecast": "volatile", "demand": "very_high"},
    "onion": {"current": 2250.0, "trend": "stable", "min": 2050.0, "max": 2550.0, "last_week": 2200.0, "forecast": "stable", "demand": "high"},
    "maize": {"current": 1950.0, "trend": "rising", "min": 1850.0, "max": 2150.0, "last_week": 1900.0, "forecast": "rising", "demand": "high"},
}


class MarketService:
    """Service providing transparent agricultural market pricing intelligence."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def get_market_data(self, crop: str, state: str = "Delhi") -> MarketContext:
        """Retrieve market prices for specified crop with clear source attribution.

        Args:
            crop: Crop name (e.g. 'Wheat', 'Tomato').
            state: Target agricultural state or mandi region.

        Returns:
            MarketContext with prices, trends, and explicit benchmark attribution.
        """
        crop_clean = crop.strip().lower() if crop else "wheat"
        benchmark = INDICATIVE_MARKET_BENCHMARKS.get(
            crop_clean,
            {
                "current": 2000.0,
                "trend": "stable",
                "min": 1800.0,
                "max": 2200.0,
                "last_week": 1950.0,
                "forecast": "stable",
                "demand": "medium",
            },
        )

        current = benchmark["current"]
        last_week = benchmark.get("last_week", current)
        price_change = current - last_week
        change_pct = (price_change / last_week * 100) if last_week else 0.0

        if benchmark["trend"] == "rising":
            insight = "Favorable mandi trend. Demand is active; observe local arrivals before trading."
        elif benchmark["trend"] == "falling":
            insight = "Prices are softening due to heavy arrivals. Verify storage conditions if holding."
        else:
            insight = "Stable regional pricing. Focus on quality grading to obtain top mandi rates."

        logger.info(f"Retrieved market benchmark for {crop}: ₹{current}/quintal ({benchmark['trend']})")

        return MarketContext(
            crop=crop.capitalize() if crop else "General",
            current_price=current,
            trend=benchmark["trend"],
            min_price=benchmark["min"],
            max_price=benchmark["max"],
            last_week_price=last_week,
            price_change=price_change,
            change_percent=round(change_pct, 1),
            forecast=benchmark.get("forecast", "stable"),
            demand=benchmark.get("demand", "medium"),
            insight=insight,
            summary=f"₹{current:.0f}/quintal ({benchmark['trend']}, {change_pct:+.1f}%)",
            source="Agmarknet Indicative Benchmark Model",
            is_live=False,  # Explicitly transparent
            timestamp=datetime.now().isoformat(),
        )

