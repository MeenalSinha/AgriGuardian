"""Tests for net farmgate realization calculator."""

import pytest
from backend.app.schemas.market import NetRealizationResult
from backend.app.services.market.net_realization_calculator import NetRealizationCalculator


def test_net_realization_standard_calculation():
    """Verify net realization formula factoring distance, transport, and hamali charges."""
    # 50 quintals of wheat at ₹2,400/qtl, distance 40 km
    res = NetRealizationCalculator.calculate(
        market_id="MKT-UP-MEERUT",
        market_name="Meerut APMC",
        commodity_name="Wheat",
        distance_km=40.0,
        quantity_quintals=50.0,
        reference_modal_price=2400.0,
    )

    assert isinstance(res, NetRealizationResult)
    # Gross value: 50 * 2400 = 120,000
    assert res.gross_value_inr == 120000.0
    # Transport: 40 km * 2.50 * 50 qtl = 5,000
    assert res.transport_cost_inr == 5000.0
    # Handling: (50 * 25.0) + (120,000 * 0.01) = 1,250 + 1,200 = 2,450
    assert res.handling_fees_inr == 2450.0
    # Total costs: 5,000 + 2,450 = 7,450
    assert res.total_estimated_costs_inr == 7450.0
    # Net: 120,000 - 7,450 = 112,550
    assert res.estimated_net_realization_inr == 112550.0
    assert res.net_realization_per_quintal == 2251.0
    assert len(res.assumptions) >= 2


def test_net_realization_user_overrides():
    """Verify user can provide exact negotiated truck hiring and labor costs."""
    res = NetRealizationCalculator.calculate(
        market_id="MKT-UP-MEERUT",
        market_name="Meerut APMC",
        commodity_name="Wheat",
        distance_km=40.0,
        quantity_quintals=50.0,
        reference_modal_price=2400.0,
        transport_cost_override=3500.0,
        handling_fee_override=1000.0,
    )

    assert res.transport_cost_inr == 3500.0
    assert res.handling_fees_inr == 1000.0
    assert res.total_estimated_costs_inr == 4500.0
    assert res.estimated_net_realization_inr == 115500.0
    assert res.net_realization_per_quintal == 2310.0


def test_net_realization_invalid_inputs():
    """Verify validation on non-positive quantities or prices."""
    with pytest.raises(ValueError):
        NetRealizationCalculator.calculate(
            market_id="MKT-1",
            market_name="Test",
            commodity_name="Wheat",
            distance_km=10.0,
            quantity_quintals=0.0,
            reference_modal_price=2000.0,
        )
