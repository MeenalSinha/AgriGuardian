"""Tests for location-aware nearby APMC mandi discovery and straight-line distances."""

import pytest
from backend.app.integrations.market.agmarknet_provider import haversine_distance_km
from backend.app.services.market.nearby_mandi_service import NearbyMandiService


@pytest.fixture
def nearby_service():
    return NearbyMandiService()


def test_haversine_distance_calculation():
    """Verify haversine straight-line distance accuracy between known coordinates."""
    # Meerut (28.9845, 77.7064) to Agra (27.1767, 78.0081) ~ 203 km
    dist = haversine_distance_km(28.9845, 77.7064, 27.1767, 78.0081)
    assert 195.0 <= dist <= 215.0


def test_discover_nearby_mandis_for_wheat(nearby_service):
    """Verify discovering mandis trading wheat from Meerut coordinates within 150 km."""
    # Meerut coordinates
    lat = 28.9845
    lon = 77.7064
    results = nearby_service.discover_nearby_prices(
        commodity_id="CMD-WHEAT",
        latitude=lat,
        longitude=lon,
        radius_km=150.0,
    )

    assert len(results) >= 1
    # First result should be Meerut itself with distance ~ 0 km
    assert results[0].market_id == "MKT-UP-MEERUT"
    assert results[0].distance_km < 1.0
    assert results[0].modal_price > 0
    assert results[0].min_price <= results[0].modal_price <= results[0].max_price


def test_discover_nearby_mandis_radius_filtering(nearby_service):
    """Verify small radius excludes distant mandis."""
    # Tight radius 5 km around Meerut
    results = nearby_service.discover_nearby_prices(
        commodity_id="CMD-WHEAT",
        latitude=28.9845,
        longitude=77.7064,
        radius_km=5.0,
    )
    for m in results:
        assert m.distance_km <= 5.0
