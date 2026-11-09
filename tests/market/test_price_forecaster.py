"""Tests for time-series price forecasting engine, backtesting, and uncertainty intervals."""

import pytest
from datetime import date, timedelta
from backend.app.integrations.market.exceptions import InsufficientDataError
from backend.app.schemas.market import (
    MarketDataSource,
    MarketPriceObservation,
    PriceForecastResult,
)
from backend.app.services.market.price_forecaster import PriceForecaster


def generate_series(n_days=40, base=2300.0, trend=2.0):
    """Generate sequential synthetic observations for testing forecaster."""
    start = date(2025, 1, 1)
    obs = []
    p = base
    for i in range(n_days):
        d_str = (start + timedelta(days=i)).isoformat()
        p += trend
        obs.append(
            MarketPriceObservation(
                observation_id=f"OBS-FC-{i}",
                commodity_id="CMD-WHEAT",
                commodity_name="Wheat",
                market_id="MKT-UP-MEERUT",
                market_name="Meerut APMC",
                state="Uttar Pradesh",
                district="Meerut",
                reporting_date=d_str,
                min_price=round(p - 40.0, 2),
                modal_price=round(p, 2),
                max_price=round(p + 40.0, 2),
                unit="₹/quintal",
                arrivals_tonnes=150.0,
                source=MarketDataSource.AGMARKNET.value,
            )
        )
    return obs


def test_forecaster_insufficient_data_rejection():
    """Verify forecaster strictly refuses to make speculative predictions when observations < 14."""
    forecaster = PriceForecaster()
    short_series = generate_series(n_days=10)

    with pytest.raises(InsufficientDataError) as exc_info:
        forecaster.forecast_prices(short_series, horizon_days=7)
    
    assert "at least 14 sequential" in str(exc_info.value)


def test_forecaster_generates_valid_forecast():
    """Verify forecast result with 14 points, intervals, and evaluation metrics."""
    forecaster = PriceForecaster()
    series = generate_series(n_days=45, base=2250.0, trend=1.5)

    res = forecaster.forecast_prices(series, horizon_days=14)
    assert isinstance(res, PriceForecastResult)
    assert res.forecast_horizon_days == 14
    assert len(res.forecast_points) == 14
    assert res.training_observations_count == 45
    assert res.backtest_mae >= 0.0
    assert res.backtest_rmse >= 0.0
    assert 0.0 <= res.backtest_directional_accuracy_pct <= 100.0
    assert res.confidence_tier in ("High", "Medium", "Low")
    assert "never guaranteed" in res.statutory_disclaimer.lower()


def test_prediction_intervals_ordering():
    """Verify prediction intervals expand logically: lower_95 <= lower_80 <= pred <= upper_80 <= upper_95."""
    forecaster = PriceForecaster()
    series = generate_series(n_days=30, base=2300.0, trend=1.0)

    res = forecaster.forecast_prices(series, horizon_days=7)
    for pt in res.forecast_points:
        assert pt.lower_interval_95 <= pt.lower_interval_80
        assert pt.lower_interval_80 <= pt.predicted_modal_price
        assert pt.predicted_modal_price <= pt.upper_interval_80
        assert pt.upper_interval_80 <= pt.upper_interval_95
        assert pt.lower_interval_95 > 0
