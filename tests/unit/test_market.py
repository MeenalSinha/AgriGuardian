"""Unit tests for transparent market intelligence."""

import pytest
from backend.app.integrations.market.market_service import MarketService
from backend.app.schemas.advisory import MarketContext


def test_market_service_known_crop(test_settings):
    """Verify market pricing and transparent source attribution for known crops."""
    service = MarketService(settings=test_settings)
    ctx = service.get_market_data(crop="Wheat")

    assert isinstance(ctx, MarketContext)
    assert ctx.crop == "Wheat"
    assert ctx.current_price > 0
    assert ctx.is_live is False  # Must be explicitly transparent
    assert "Agmarknet Indicative" in ctx.source
    assert ctx.trend in ("rising", "falling", "stable")


def test_market_service_unknown_crop(test_settings):
    """Verify market service returns sensible baseline for unlisted crop."""
    service = MarketService(settings=test_settings)
    ctx = service.get_market_data(crop="Dragonfruit")

    assert isinstance(ctx, MarketContext)
    assert ctx.crop == "Dragonfruit"
    assert ctx.current_price == 2000.0
    assert ctx.is_live is False

