"""Tests for time-series market price analytics and volatility classification."""

import pytest
from datetime import date, timedelta
from backend.app.schemas.market import (
    MarketDataSource,
    MarketPriceObservation,
    PriceTrendDirection,
)
from backend.app.services.market.market_analytics import MarketAnalyticsService


def make_dummy_observations(prices, start_date="2025-01-01"):
    """Helper to generate sequential dummy observations."""
    base_date = date.fromisoformat(start_date)
    obs_list = []
    for i, p in enumerate(prices):
        d_str = (base_date + timedelta(days=i)).isoformat()
        obs = MarketPriceObservation(
            observation_id=f"OBS-{i}",
            commodity_id="CMD-WHEAT",
            commodity_name="Wheat",
            market_id="MKT-UP-MEERUT",
            market_name="Meerut APMC",
            state="Uttar Pradesh",
            district="Meerut",
            reporting_date=d_str,
            min_price=p - 50.0,
            modal_price=p,
            max_price=p + 50.0,
            unit="₹/quintal",
            arrivals_tonnes=100.0,
            source=MarketDataSource.AGMARKNET.value,
        )
        obs_list.append(obs)
    return obs_list


def test_rising_trend_analysis():
    """Verify classification of rising price trajectory (> +3%)."""
    # 2000 to 2200 = +10%
    prices = [2000.0, 2020.0, 2050.0, 2100.0, 2150.0, 2180.0, 2200.0]
    obs_list = make_dummy_observations(prices)

    trend = MarketAnalyticsService.analyze_trend(obs_list, window_days=7)
    assert trend.trend_direction == PriceTrendDirection.RISING
    assert trend.percentage_change == 10.0
    assert trend.absolute_change == 200.0
    assert trend.period_min == 2000.0
    assert trend.period_max == 2200.0
    assert trend.volatility_std > 0


def test_falling_trend_analysis():
    """Verify classification of falling price trajectory (< -3%)."""
    # 2500 down to 2300 = -8%
    prices = [2500.0, 2480.0, 2420.0, 2380.0, 2340.0, 2300.0]
    obs_list = make_dummy_observations(prices)

    trend = MarketAnalyticsService.analyze_trend(obs_list, window_days=6)
    assert trend.trend_direction == PriceTrendDirection.FALLING
    assert trend.percentage_change == -8.0
    assert trend.absolute_change == -200.0


def test_stable_trend_analysis():
    """Verify classification of stable price trajectory (within +/- 3%)."""
    prices = [2200.0, 2205.0, 2195.0, 2210.0, 2200.0, 2202.0]
    obs_list = make_dummy_observations(prices)

    trend = MarketAnalyticsService.analyze_trend(obs_list, window_days=6)
    assert trend.trend_direction == PriceTrendDirection.STABLE
    assert abs(trend.percentage_change) < 3.0


def test_insufficient_data_trend():
    """Verify fewer than 3 observations yields INSUFFICIENT_DATA."""
    prices = [2200.0, 2250.0]
    obs_list = make_dummy_observations(prices)

    trend = MarketAnalyticsService.analyze_trend(obs_list, window_days=2)
    assert trend.trend_direction == PriceTrendDirection.INSUFFICIENT_DATA
