"""Historical climate reanalysis client and agricultural baseline anomaly analyzer."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import requests
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.integrations.weather.weather_cache import weather_cache
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import HistoricalWeatherComparison

logger = get_logger("HistoricalWeatherIntegration")


class HistoricalWeatherService:
    """Computes climate anomaly departures against historical reanalysis normals (ERA5 / ERA5-Land)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.archive_base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.timeout = self.settings.weather_timeout_seconds

    def get_historical_anomaly(
        self,
        location: LocationRecord,
        current_mean_temp: float,
        current_cumulative_rain_mm: float,
        consecutive_dry_days: int = 0,
        reference_years_count: int = 5,
    ) -> HistoricalWeatherComparison:
        """Calculate temperature anomaly and rainfall departure % for current period vs 5-year normal.
        
        Args:
            location: Agricultural target location.
            current_mean_temp: Observed or forecast mean temperature for the current window (°C).
            current_cumulative_rain_mm: Observed or forecast cumulative precipitation (mm).
            consecutive_dry_days: Number of consecutive days with precipitation < 1.0 mm.
            reference_years_count: Number of baseline years to compare (default: 5 years).
            
        Returns:
            HistoricalWeatherComparison containing departures, dry days, and explicit reanalysis provenance.
        """
        cache_key = f"historical_{reference_years_count}"
        cached = weather_cache.get(location.latitude, location.longitude, cache_type=cache_key)
        if cached:
            logger.info(f"Using cached historical reanalysis baseline for ({location.latitude:.2f}, {location.longitude:.2f})")
            return self._calculate_anomaly(
                baseline_data=cached,
                current_mean_temp=current_mean_temp,
                current_cumulative_rain_mm=current_cumulative_rain_mm,
                consecutive_dry_days=consecutive_dry_days,
                ref_years_count=reference_years_count,
            )

        now = datetime.now(timezone.utc)
        current_month = now.month
        current_day = now.day

        # Define 14-day calendar window for preceding years (e.g. 2021-2025)
        baseline_temps: List[float] = []
        baseline_rains: List[float] = []

        try:
            # Query archive API for baseline window
            end_year = now.year - 1
            start_year = end_year - reference_years_count + 1

            start_date_str = f"{start_year}-{current_month:02d}-{max(1, current_day - 7):02d}"
            end_date_str = f"{end_year}-{current_month:02d}-{min(28, current_day + 7):02d}"

            params = {
                "latitude": round(location.latitude, 4),
                "longitude": round(location.longitude, 4),
                "start_date": start_date_str,
                "end_date": end_date_str,
                "daily": "temperature_2m_mean,precipitation_sum",
                "timezone": "Asia/Kolkata",
            }

            logger.info(f"Querying Open-Meteo ERA5 archive: {start_date_str} to {end_date_str}")
            resp = requests.get(self.archive_base_url, params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                d_temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
                d_rains = [r for r in daily.get("precipitation_sum", []) if r is not None]

                if d_temps and d_rains:
                    mean_temp = sum(d_temps) / len(d_temps)
                    # 14-day equivalent cumulative rainfall
                    mean_14d_rain = (sum(d_rains) / len(d_rains)) * 14.0
                    baseline_data = {
                        "mean_temp": mean_temp,
                        "mean_rain": mean_14d_rain,
                        "source": "Open-Meteo ERA5 / ERA5-Land Reanalysis (Modeled)",
                    }
                    weather_cache.set(
                        location.latitude,
                        location.longitude,
                        baseline_data,
                        cache_type=cache_key,
                        ttl_seconds=86400,  # Cache 24 hours
                    )
                    return self._calculate_anomaly(
                        baseline_data=baseline_data,
                        current_mean_temp=current_mean_temp,
                        current_cumulative_rain_mm=current_cumulative_rain_mm,
                        consecutive_dry_days=consecutive_dry_days,
                        ref_years_count=reference_years_count,
                    )
        except Exception as e:
            logger.warning(f"ERA5 reanalysis archive query failed ({e}). Using agro-climatic zone normals.")

        # Robust agricultural climatological fallback baseline (modeled normal for Indian coordinates)
        # North/Central/South India seasonal baseline temperature ~ 22-30°C depending on latitude and month
        approx_base_temp = 24.0 if current_month in [11, 12, 1, 2] else (32.0 if current_month in [3, 4, 5, 6] else 28.0)
        approx_base_rain = 5.0 if current_month in [11, 12, 1, 2, 3, 4] else 85.0

        fallback_baseline = {
            "mean_temp": approx_base_temp,
            "mean_rain": approx_base_rain,
            "source": "Agro-Climatic Normal Model (Regional Reference Baseline)",
        }
        return self._calculate_anomaly(
            baseline_data=fallback_baseline,
            current_mean_temp=current_mean_temp,
            current_cumulative_rain_mm=current_cumulative_rain_mm,
            consecutive_dry_days=consecutive_dry_days,
            ref_years_count=reference_years_count,
        )

    def _calculate_anomaly(
        self,
        baseline_data: Dict[str, Any],
        current_mean_temp: float,
        current_cumulative_rain_mm: float,
        consecutive_dry_days: int,
        ref_years_count: int,
    ) -> HistoricalWeatherComparison:
        """Derive quantitative departures."""
        base_temp = baseline_data["mean_temp"]
        base_rain = baseline_data["mean_rain"]

        temp_anomaly = round(current_mean_temp - base_temp, 2)
        if base_rain > 0.1:
            rain_departure_pct = round(((current_cumulative_rain_mm - base_rain) / base_rain) * 100.0, 1)
        else:
            rain_departure_pct = 0.0 if current_cumulative_rain_mm < 1.0 else 100.0

        notes = []
        if temp_anomaly > 2.5:
            notes.append(f"Temperature is significantly warmer than normal (+{temp_anomaly}°C). Elevated transpiration expected.")
        elif temp_anomaly < -2.5:
            notes.append(f"Temperature is significantly cooler than normal ({temp_anomaly}°C). Potential chill stress.")
        else:
            notes.append("Temperature is within normal seasonal variability.")

        if rain_departure_pct < -50.0:
            notes.append(f"Rainfall deficit of {abs(rain_departure_pct):.0f}% compared to 5-year climatological normal.")
        elif rain_departure_pct > 50.0:
            notes.append(f"Excess rainfall of +{rain_departure_pct:.0f}% compared to 5-year climatological normal.")

        if consecutive_dry_days >= 7:
            notes.append(f"Dry-spell in progress ({consecutive_dry_days} consecutive rainless days).")

        return HistoricalWeatherComparison(
            reference_period_years=ref_years_count,
            baseline_mean_temperature_c=round(base_temp, 2),
            current_period_mean_temperature_c=round(current_mean_temp, 2),
            temperature_anomaly_c=temp_anomaly,
            baseline_total_rainfall_mm=round(base_rain, 1),
            current_period_rainfall_mm=round(current_cumulative_rain_mm, 1),
            rainfall_anomaly_pct=rain_departure_pct,
            consecutive_dry_days=consecutive_dry_days,
            data_source=baseline_data["source"],
            is_modeled_reanalysis=True,
            notes=" ".join(notes),
        )

