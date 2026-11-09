"""Time-series market price analytics, volatility calculations, and trend classification."""

import math
import statistics
from typing import List, Optional

from backend.app.core.logging import get_logger
from backend.app.schemas.market import (
    MarketPriceObservation,
    MarketTrendAnalysis,
    PriceTrendDirection,
)

logger = get_logger("MarketAnalytics")


class MarketAnalyticsService:
    """Computes time-series statistics, volatility, and trend trajectories for mandi commodities."""

    @classmethod
    def analyze_trend(
        cls,
        observations: List[MarketPriceObservation],
        window_days: int = 30,
    ) -> MarketTrendAnalysis:
        """Perform statistical trend analysis over historical observations.
        
        Args:
            observations: Chronologically sorted list of MarketPriceObservation.
            window_days: Analytical window length in calendar days.
            
        Returns:
            MarketTrendAnalysis with statistical metrics and trend classification.
        """
        if not observations:
            raise ValueError("Cannot perform trend analysis on an empty observation list.")

        # Sort chronologically by reporting date
        sorted_obs = sorted(observations, key=lambda x: x.reporting_date)
        first_obs = sorted_obs[0]
        latest_obs = sorted_obs[-1]

        modal_prices = [obs.modal_price for obs in sorted_obs]
        n = len(modal_prices)

        # Baseline metrics
        latest_price = latest_obs.modal_price
        previous_price = sorted_obs[0].modal_price if n > 1 else latest_price
        abs_change = round(latest_price - previous_price, 2)
        pct_change = round((abs_change / previous_price) * 100.0, 2) if previous_price > 0 else 0.0

        period_avg = round(statistics.mean(modal_prices), 2)
        period_med = round(statistics.median(modal_prices), 2)
        period_min = round(min(modal_prices), 2)
        period_max = round(max(modal_prices), 2)
        price_range = round(period_max - period_min, 2)

        # Volatility: Sample standard deviation
        volatility_std = round(statistics.stdev(modal_prices), 2) if n > 1 else 0.0
        cv = (volatility_std / period_avg) if period_avg > 0 else 0.0

        # Classification
        if n < 3:
            trend_dir = PriceTrendDirection.INSUFFICIENT_DATA
            explanation = "Insufficient sequential market trading days for confident trend identification."
        elif cv > 0.15 and abs(pct_change) < 5.0:
            trend_dir = PriceTrendDirection.VOLATILE
            explanation = (
                f"High price fluctuation observed (Standard Deviation: ₹{volatility_std:,.2f}/qtl, "
                f"CV: {cv:.1%}) with no single sustained directional momentum."
            )
        elif pct_change > 3.0:
            trend_dir = PriceTrendDirection.RISING
            explanation = (
                f"Upward price trajectory observed, advancing +{pct_change}% (+₹{abs_change:,.2f}/qtl) "
                f"over the {n}-observation period."
            )
        elif pct_change < -3.0:
            trend_dir = PriceTrendDirection.FALLING
            explanation = (
                f"Downward price trajectory observed, declining {pct_change}% (-₹{abs(abs_change):,.2f}/qtl) "
                f"over the {n}-observation period."
            )
        else:
            trend_dir = PriceTrendDirection.STABLE
            explanation = (
                f"Relatively stable price band between ₹{period_min:,.2f} and ₹{period_max:,.2f}/qtl "
                f"(net change: {pct_change}%)."
            )

        # Completeness (relative to 6 trading days/week over window)
        expected_trading_days = max(1, int(window_days * (6.0 / 7.0)))
        completeness_pct = min(100.0, round((n / expected_trading_days) * 100.0, 1))

        # Charting serialization
        history_points = [
            {
                "reporting_date": obs.reporting_date,
                "modal_price": obs.modal_price,
                "min_price": obs.min_price,
                "max_price": obs.max_price,
                "arrivals_tonnes": obs.arrivals_tonnes,
            }
            for obs in sorted_obs
        ]

        return MarketTrendAnalysis(
            commodity_id=latest_obs.commodity_id,
            commodity_name=latest_obs.commodity_name,
            market_id=latest_obs.market_id,
            market_name=latest_obs.market_name,
            analysis_window_days=window_days,
            observations_count=n,
            latest_price=latest_price,
            previous_price=previous_price if n > 1 else None,
            absolute_change=abs_change if n > 1 else None,
            percentage_change=pct_change if n > 1 else None,
            period_average=period_avg,
            period_median=period_med,
            period_min=period_min,
            period_max=period_max,
            price_range=price_range,
            volatility_std=volatility_std,
            trend_direction=trend_dir,
            trend_explanation=explanation,
            data_completeness_pct=completeness_pct,
            history_points=history_points,
        )
