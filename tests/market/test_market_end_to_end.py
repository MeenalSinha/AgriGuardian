"""End-to-end integration tests for agricultural market intelligence service and REST API."""

import pytest
from backend.app.integrations.market.market_cache import market_cache
from backend.app.main import (
    NetRealizationRequest,
    calculate_net_realization,
    discover_nearby_mandis,
    get_latest_market_price,
    get_market_history_and_trend,
    get_msp_benchmark,
    get_price_forecast,
    list_commodities,
    list_mandis,
)
from backend.app.services.market.market_intelligence import MarketIntelligenceService


@pytest.fixture(autouse=True)
def clear_market_cache():
    """Ensure clean cache before every test."""
    market_cache.invalidate()
    yield
    market_cache.invalidate()


@pytest.fixture
def market_service():
    return MarketIntelligenceService()


def test_market_intelligence_service_full_flow(market_service):
    """Verify unified market intelligence summary generation for Wheat in Meerut."""
    summary = market_service.get_market_summary(
        commodity_query="Wheat",
        market_id="MKT-UP-MEERUT",
        history_days=90,
        forecast_horizon_days=14,
    )

    assert summary["is_available"] is True
    assert summary["commodity"]["commodity_id"] == "CMD-WHEAT"
    assert summary["latest_observation"] is not None
    assert summary["latest_observation"]["market_id"] == "MKT-UP-MEERUT"
    assert summary["latest_observation"]["modal_price"] > 0
    assert summary["trend_analysis"] is not None
    assert summary["trend_analysis"]["observations_count"] > 14
    assert summary["msp_comparison"] is not None
    assert summary["msp_comparison"]["has_msp"] is True
    assert summary["forecast"] is not None
    assert len(summary["forecast"]["forecast_points"]) == 14
    assert len(summary["nearby_mandis"]) >= 1


def test_market_context_for_advisory(market_service):
    """Verify conversational advisory context generation."""
    ctx = market_service.get_market_context_for_advisory(
        crop_name="Wheat",
        location_query="Meerut",
    )

    assert ctx["is_available"] is True
    assert ctx["crop"] == "Wheat"
    assert ctx["modal_price"] > 0
    assert "₹" in ctx["summary"]


def test_api_list_commodities():
    """Verify GET /api/v1/market/commodities endpoint logic."""
    data = list_commodities()
    assert data["count"] >= 10
    names = [c["canonical_name"] for c in data["commodities"]]
    assert "Wheat" in names
    assert "Tomato" in names
    assert "Paddy (Dhan)" in names


def test_api_list_mandis():
    """Verify GET /api/v1/market/mandis endpoint logic."""
    data = list_mandis(state="Uttar Pradesh")
    assert data["count"] >= 3
    for m in data["mandis"]:
        assert m["state"] == "Uttar Pradesh"
        assert -90 <= m["latitude"] <= 90
        assert -180 <= m["longitude"] <= 180


def test_api_get_latest_price():
    """Verify GET /api/v1/market/prices/latest endpoint logic."""
    data = get_latest_market_price(commodity="Wheat", market_id="MKT-UP-MEERUT")
    assert data["commodity_id"] == "CMD-WHEAT"
    assert data["modal_price"] > 0
    assert data["min_price"] <= data["modal_price"] <= data["max_price"]


def test_api_get_nearby_mandis():
    """Verify GET /api/v1/market/nearby endpoint logic."""
    data = discover_nearby_mandis(commodity="Wheat", location="Meerut", radius_km=150.0)
    assert data["count"] >= 1
    assert data["nearby_mandis"][0]["distance_km"] <= 150.0


def test_api_get_market_history():
    """Verify GET /api/v1/market/history endpoint logic."""
    data = get_market_history_and_trend(commodity="Wheat", market_id="MKT-UP-MEERUT", days=30)
    assert data["observations_count"] > 10
    assert "trend_analysis" in data
    assert "period_average" in data["trend_analysis"]


def test_api_get_msp_benchmark():
    """Verify GET /api/v1/market/msp endpoint logic."""
    data = get_msp_benchmark(commodity="Wheat", current_modal_price=2350.0)
    assert data["has_msp"] is True
    assert data["msp_price"] == 2275.0
    assert data["status"] == "ABOVE_MSP"


def test_api_get_price_forecast():
    """Verify GET /api/v1/market/forecast endpoint logic."""
    data = get_price_forecast(commodity="Wheat", market_id="MKT-UP-MEERUT", horizon_days=7)
    assert len(data["forecast_points"]) == 7
    assert data["backtest_mae"] >= 0


def test_api_calculate_net_realization():
    """Verify POST /api/v1/market/calculate-net-realization endpoint logic."""
    req = NetRealizationRequest(
        market_id="MKT-UP-MEERUT",
        market_name="Meerut APMC",
        commodity_name="Wheat",
        distance_km=50.0,
        quantity_quintals=40.0,
        reference_modal_price=2350.0,
    )
    data = calculate_net_realization(req)
    assert data["gross_value_inr"] == 94000.0
    assert data["estimated_net_realization_inr"] < 94000.0
    assert data["net_realization_per_quintal"] > 0
