"""Tests for market data validator and unit normalizer."""

import pytest
from datetime import date, timedelta
from backend.app.integrations.market.market_data_validator import MarketDataValidator


def test_unit_normalization_kg_to_quintal():
    """Verify conversion of per-kg retail rates to standard ₹/quintal."""
    price_qtl, unit = MarketDataValidator.normalize_price_unit(raw_price=25.0, raw_unit="₹/kg")
    assert price_qtl == 2500.0
    assert unit == "₹/quintal"


def test_unit_normalization_tonne_to_quintal():
    """Verify conversion of metric tonne bulk rates to standard ₹/quintal."""
    price_qtl, unit = MarketDataValidator.normalize_price_unit(raw_price=34000.0, raw_unit="Rs/Tonne")
    assert price_qtl == 3400.0
    assert unit == "₹/quintal"


def test_unit_normalization_quintal_passthrough():
    """Verify quintal rates are preserved without alteration."""
    price_qtl, unit = MarketDataValidator.normalize_price_unit(raw_price=2275.0, raw_unit="Rs/Quintal")
    assert price_qtl == 2275.0
    assert unit == "₹/quintal"


def test_observation_bounds_valid():
    """Verify valid ordering min <= modal <= max passes."""
    is_valid, err = MarketDataValidator.validate_observation_bounds(
        min_p=2100.0,
        modal_p=2250.0,
        max_p=2400.0,
    )
    assert is_valid is True
    assert err is None


def test_observation_bounds_modal_below_min():
    """Verify error when modal price is strictly below minimum price."""
    is_valid, err = MarketDataValidator.validate_observation_bounds(
        min_p=2300.0,
        modal_p=2150.0,
        max_p=2400.0,
    )
    assert is_valid is False
    assert "exceeds modal price" in err


def test_observation_bounds_modal_above_max():
    """Verify error when modal price is strictly above maximum price."""
    is_valid, err = MarketDataValidator.validate_observation_bounds(
        min_p=2100.0,
        modal_p=2450.0,
        max_p=2400.0,
    )
    assert is_valid is False
    assert "exceeds maximum price" in err


def test_observation_bounds_biological_extremes():
    """Verify extreme unrealistic price bounds fail validation."""
    is_valid, err = MarketDataValidator.validate_observation_bounds(
        min_p=10.0,
        modal_p=25.0,
        max_p=30.0,
    )
    assert is_valid is False
    assert "violate biological" in err


def test_reporting_date_validation():
    """Verify current and historical dates pass, but distant future dates fail."""
    today_str = date.today().isoformat()
    is_valid, err = MarketDataValidator.validate_reporting_date(today_str)
    assert is_valid is True

    future_str = (date.today() + timedelta(days=5)).isoformat()
    is_valid, err = MarketDataValidator.validate_reporting_date(future_str)
    assert is_valid is False
    assert "future" in err.lower()
