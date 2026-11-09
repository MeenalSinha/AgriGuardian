"""Time-series market price forecasting engine with walk-forward backtesting and uncertainty intervals."""

import math
import statistics
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.integrations.market.exceptions import InsufficientDataError
from backend.app.schemas.market import (
    MarketPriceObservation,
    PriceForecastPoint,
    PriceForecastResult,
)

logger = get_logger("PriceForecaster")


class PriceForecaster:
    """Statistical time-series forecasting engine with Holt linear trend smoothing and walk-forward validation."""

    MIN_OBSERVATIONS_REQUIRED = 14

    def __init__(self, alpha: float = 0.35, beta: float = 0.15):
        """Initialize Holt exponential smoothing parameters.
        
        Args:
            alpha: Level smoothing factor (0 < alpha < 1).
            beta: Trend smoothing factor (0 < beta < 1).
        """
        self.alpha = alpha
        self.beta = beta

    def forecast_prices(
        self,
        observations: List[MarketPriceObservation],
        horizon_days: int = 14,
    ) -> PriceForecastResult:
        """Generate point forecasts and prediction intervals for future days using sequential history.
        
        Args:
            observations: Chronologically ordered daily observations.
            horizon_days: Forecast horizon (typically 7 to 14 days).
            
        Returns:
            PriceForecastResult containing forecast points, backtest metrics, and statutory disclaimers.
            
        Raises:
            InsufficientDataError: If valid sequential observations < MIN_OBSERVATIONS_REQUIRED.
        """
        sorted_obs = sorted(observations, key=lambda x: x.reporting_date)
        n = len(sorted_obs)

        if n < self.MIN_OBSERVATIONS_REQUIRED:
            raise InsufficientDataError(
                f"Price forecasting requires at least {self.MIN_OBSERVATIONS_REQUIRED} sequential daily observations. "
                f"Provided: {n} observations."
            )

        prices = [obs.modal_price for obs in sorted_obs]
        last_obs = sorted_obs[-1]
        last_date = datetime.strptime(last_obs.reporting_date, "%Y-%m-%d").date()

        # 1. Walk-Forward Rolling-Origin Backtesting (Strictly No Random Split)
        # Test on the last 14 historical observations (or last 20% of data)
        test_window = min(14, max(5, int(n * 0.2)))
        train_len = n - test_window
        
        backtest_mae, backtest_rmse, dir_acc, naive_mae, residual_std = self._walk_forward_evaluation(
            prices=prices,
            train_len=train_len,
            test_window=test_window,
        )

        # 2. Fit Holt Model on Full Historical Data
        level, trend = self._fit_holt(prices)

        # 3. Generate Forward Horizon Points with 80% and 95% Prediction Intervals
        forecast_points: List[PriceForecastPoint] = []
        
        # Horizon step loop
        for h in range(1, horizon_days + 1):
            future_date = last_date + timedelta(days=h)
            
            # Point forecast: level + h * trend (with slight dampening factor for longer horizons)
            dampening = 0.95 ** (h - 1)
            pred_price = round(max(10.0, level + (h * trend * dampening)), 2)
            
            # Prediction intervals expand with horizon: standard error ~ residual_std * sqrt(h)
            step_se = residual_std * math.sqrt(h)
            
            # 80% CI: z = 1.282
            lower_80 = round(max(5.0, pred_price - (1.282 * step_se)), 2)
            upper_80 = round(pred_price + (1.282 * step_se), 2)
            
            # 95% CI: z = 1.960
            lower_95 = round(max(5.0, pred_price - (1.960 * step_se)), 2)
            upper_95 = round(pred_price + (1.960 * step_se), 2)
            
            # Enforce logical boundary: lower <= pred <= upper
            lower_80 = min(lower_80, pred_price)
            upper_80 = max(upper_80, pred_price)
            lower_95 = min(lower_95, lower_80)
            upper_95 = max(upper_95, upper_80)

            forecast_points.append(
                PriceForecastPoint(
                    forecast_date=future_date.isoformat(),
                    predicted_modal_price=pred_price,
                    lower_interval_80=lower_80,
                    upper_interval_80=upper_80,
                    lower_interval_95=lower_95,
                    upper_interval_95=upper_95,
                )
            )

        # Confidence Tier Determination
        model_outperformed = backtest_mae < naive_mae
        pct_mae = (backtest_mae / last_obs.modal_price) if last_obs.modal_price > 0 else 1.0
        
        if n >= 60 and pct_mae < 0.05 and model_outperformed:
            confidence_tier = "High"
        elif n >= 25 and pct_mae < 0.10:
            confidence_tier = "Medium"
        else:
            confidence_tier = "Low"

        return PriceForecastResult(
            commodity_id=last_obs.commodity_id,
            commodity_name=last_obs.commodity_name,
            market_id=last_obs.market_id,
            market_name=last_obs.market_name,
            variety_name=last_obs.variety_name,
            forecast_horizon_days=horizon_days,
            model_name="Holt Linear Exponential Smoothing with Walk-Forward Backtesting",
            model_version="1.0.0",
            training_observations_count=n,
            last_historical_date=last_obs.reporting_date,
            last_observed_modal_price=last_obs.modal_price,
            forecast_points=forecast_points,
            backtest_mae=round(backtest_mae, 2),
            backtest_rmse=round(backtest_rmse, 2),
            backtest_directional_accuracy_pct=round(dir_acc, 1),
            baseline_naive_mae=round(naive_mae, 2),
            model_outperformed_baseline=model_outperformed,
            confidence_tier=confidence_tier,
        )

    def _fit_holt(self, series: List[float]) -> Tuple[float, float]:
        """Fit Holt linear exponential smoothing to series."""
        n = len(series)
        if n < 2:
            return series[0], 0.0

        # Initialization
        level = series[0]
        trend = series[1] - series[0]

        for t in range(1, n):
            prev_level = level
            prev_trend = trend
            val = series[t]
            level = self.alpha * val + (1.0 - self.alpha) * (prev_level + prev_trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * prev_trend

        return level, trend

    def _walk_forward_evaluation(
        self,
        prices: List[float],
        train_len: int,
        test_window: int,
    ) -> Tuple[float, float, float, float, float]:
        """Perform rolling-origin 1-step ahead backtesting across the test window."""
        errors: List[float] = []
        naive_errors: List[float] = []
        directional_correct = 0

        # Rolling loop
        for t in range(train_len, len(prices)):
            history_train = prices[:t]
            actual = prices[t]
            prev_actual = prices[t - 1]

            # Fit model on training slice
            lvl, trd = self._fit_holt(history_train)
            pred = lvl + trd

            # Model error
            err = actual - pred
            errors.append(err)

            # Persistence Naive baseline (y_hat = y_{t-1})
            naive_err = actual - prev_actual
            naive_errors.append(naive_err)

            # Directional accuracy
            actual_move = actual - prev_actual
            pred_move = pred - prev_actual
            if (actual_move >= 0 and pred_move >= 0) or (actual_move < 0 and pred_move < 0):
                directional_correct += 1

        mae = statistics.mean([abs(e) for e in errors]) if errors else 0.0
        mse = statistics.mean([e ** 2 for e in errors]) if errors else 0.0
        rmse = math.sqrt(mse) if mse > 0 else 0.0
        naive_mae = statistics.mean([abs(e) for e in naive_errors]) if naive_errors else 0.0
        residual_std = statistics.stdev(errors) if len(errors) > 1 else max(15.0, mae)
        dir_acc = (directional_correct / len(errors) * 100.0) if errors else 50.0

        return mae, rmse, dir_acc, naive_mae, residual_std
