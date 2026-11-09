"""Unit tests for transparent market intelligence."""

import pytest
from backend.app.integrations.market.market_service import MarketService
from backend.app.schemas.advisory import MarketContext


def test_market_service_known_crop(test_settings):
    """Verify market pricing and transparent source attribution for known crops when location is provided."""
    service = MarketService(settings=test_settings)
    ctx = service.get_market_data(crop="Wheat", location="Punjab")

    assert isinstance(ctx, MarketContext)
    assert ctx.crop == "Wheat"
    assert ctx.current_price is not None and ctx.current_price > 0
    assert ctx.is_available is True
    assert ctx.is_live is False  # Must be explicitly transparent
    assert "AGMARKNET" in ctx.source
    assert ctx.trend in ("rising", "falling", "stable", "volatile")
    assert ctx.has_msp is True
    assert ctx.msp_price == 2275.0 or ctx.msp_price == 2425.0


def test_market_service_missing_location(test_settings):
    """Verify market service returns explicit unavailable state when location is missing (no silent fallback)."""
    service = MarketService(settings=test_settings)
    ctx = service.get_market_data(crop="Wheat")

    assert isinstance(ctx, MarketContext)
    assert ctx.is_available is False
    assert ctx.trend == "location_required"
    assert "location required" in ctx.insight.lower()


def test_market_service_unknown_crop(test_settings):
    """Verify market service returns explicit unavailable state without synthetic price fallbacks."""
    service = MarketService(settings=test_settings)
    ctx = service.get_market_data(crop="Dragonfruit", location="Delhi")

    assert isinstance(ctx, MarketContext)
    assert ctx.crop == "Dragonfruit"
    assert ctx.is_available is False
    assert ctx.current_price is None
    assert "unavailable" in ctx.summary.lower()

