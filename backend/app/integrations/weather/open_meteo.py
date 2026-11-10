"""Open-Meteo Weather API Integration with caching, provenance, and backward compatibility."""

import time
from typing import Any, Dict, Optional
import requests
from backend.app.core.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenException
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import WeatherAPIError
from backend.app.core.logging import get_logger
from backend.app.core.metrics import get_metrics_collector
from backend.app.integrations.weather.historical_weather import HistoricalWeatherService
from backend.app.integrations.weather.weather_cache import weather_cache
from backend.app.integrations.weather.weather_normalizer import normalize_open_meteo_response
from backend.app.schemas.advisory import WeatherContext
from backend.app.schemas.location import LocationPrecision, LocationRecord, PRESET_AGRICULTURAL_DISTRICTS
from backend.app.schemas.weather import WeatherProvenance, WeatherReport

logger = get_logger("WeatherIntegration")


class WeatherService:
    """Service for retrieving agricultural weather conditions, forecasts, and historical comparisons."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.open_meteo_base_url
        self.timeout = self.settings.weather_timeout_seconds
        self.historical_service = HistoricalWeatherService(settings=self.settings)

    def get_weather(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        location_name: Optional[str] = None,
    ) -> WeatherContext:
        """Fetch current weather and 3-day forecast from Open-Meteo (Phase 0-3 legacy compatibility).

        Args:
            lat: Latitude (defaults to configured default, e.g. Delhi).
            lon: Longitude (defaults to configured default).
            location_name: Optional human-readable location label.

        Returns:
            WeatherContext with temperatures, humidity, rainfall, and summary.
        """
        latitude = lat if lat is not None else self.settings.default_latitude
        longitude = lon if lon is not None else self.settings.default_longitude

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Kolkata",
            "forecast_days": 3,
        }

        cb = get_circuit_breaker("open_meteo")
        metrics = get_metrics_collector()
        try:
            logger.info(f"Fetching legacy weather for coordinates ({latitude:.4f}, {longitude:.4f})")
            cb.before_call()
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            cb.record_success()
            metrics.record_provider_call("open_meteo", success=True)
            data = response.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            temp = float(current.get("temperature_2m", 25.0))
            humidity = float(current.get("relative_humidity_2m", 50.0))
            precipitation = float(current.get("precipitation", 0.0))

            max_temps = daily.get("temperature_2m_max", [30.0, 31.0, 32.0])[:3]
            min_temps = daily.get("temperature_2m_min", [20.0, 21.0, 22.0])[:3]
            precip_sums = daily.get("precipitation_sum", [0.0, 0.0, 0.0])[:3]

            logger.info(f"Weather fetched: {temp}°C, {humidity}% humidity, {precipitation}mm rain")

            return WeatherContext(
                temperature=temp,
                humidity=humidity,
                precipitation=precipitation,
                rain_recent=precipitation > 1.0,
                temp_high=temp > 35.0,
                temp_very_high=temp > 40.0,
                humidity_high=humidity > 80.0,
                forecast={
                    "max_temp": max_temps,
                    "min_temp": min_temps,
                    "precipitation": precip_sums,
                },
                summary=f"{temp:.1f}°C, {humidity:.0f}% humidity, {precipitation:.1f}mm rain",
                is_live=True,
            )

        except Exception as e:
            if not isinstance(e, CircuitBreakerOpenException):
                cb.record_failure(e)
            metrics.record_provider_call("open_meteo", success=False)
            logger.warning(f"Live weather fetch failed ({e}). Returning explicit unavailable context.")
            return WeatherContext(
                temperature=None,
                humidity=None,
                precipitation=None,
                rain_recent=False,
                temp_high=False,
                temp_very_high=False,
                humidity_high=False,
                forecast={},
                summary="Weather data temporarily unavailable (offline/disconnected)",
                is_live=False,
                is_available=False,
            )

    def get_weather_report(
        self,
        location: Optional[LocationRecord] = None,
        forecast_days: int = 7,
    ) -> WeatherReport:
        """Fetch comprehensive agricultural weather intelligence report with hourly, daily, and reanalysis.
        
        Args:
            location: Target farm or district location.
            forecast_days: Days of forecast to request (1 to 16, default 7).
            
        Returns:
            Normalized, typed WeatherReport with provenance metadata.
        """
        loc = location or LocationRecord(
            location_id="LOC-DEFAULT-VARANASI",
            display_name="Varanasi, Uttar Pradesh",
            district="Varanasi",
            state="Uttar Pradesh",
            latitude=self.settings.default_latitude,
            longitude=self.settings.default_longitude,
            elevation_m=81.0,
            precision=LocationPrecision.DISTRICT_CENTROID.value,
        )

        lat = loc.latitude
        lon = loc.longitude
        cache_key = f"forecast_d{forecast_days}"

        # 1. Check in-memory spatial cache
        cached_entry = weather_cache.get(lat, lon, cache_type=cache_key)
        if cached_entry:
            raw_data, cached_at = cached_entry
            age_s = time.time() - cached_at
            logger.info(f"Serving weather report from cache for ({lat:.2f}, {lon:.2f}), age={age_s:.0f}s")
            report = normalize_open_meteo_response(raw_data, loc, is_cached=True, data_age_seconds=age_s)
            self._attach_historical_analysis(report)
            return report

        # 2. Build Open-Meteo API query
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "current": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover",
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,precipitation_probability,weather_code,wind_speed_10m,wind_gusts_10m,shortwave_radiation,et0_fao_evapotranspiration",
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,apparent_temperature_min,precipitation_sum,rain_sum,precipitation_probability_max,precipitation_hours,weather_code,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,et0_fao_evapotranspiration,sunshine_duration",
            "timezone": "Asia/Kolkata",
            "forecast_days": min(max(forecast_days, 1), 16),
        }

        cb = get_circuit_breaker("open_meteo")
        metrics = get_metrics_collector()
        try:
            logger.info(f"Querying Open-Meteo API for ({lat:.4f}, {lon:.4f})")
            cb.before_call()
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            cb.record_success()
            metrics.record_provider_call("open_meteo", success=True)
            data = resp.json()

            # Cache the raw response
            weather_cache.set(lat, lon, (data, time.time()), cache_type=cache_key, ttl_seconds=1800)

            report = normalize_open_meteo_response(data, loc, is_cached=False, data_age_seconds=0.0)
            self._attach_historical_analysis(report)
            return report

        except Exception as e:
            if not isinstance(e, CircuitBreakerOpenException):
                cb.record_failure(e)
            metrics.record_provider_call("open_meteo", success=False)
            logger.warning(f"Live weather API request failed ({e}). Returning transparent unavailable report.")
            return WeatherReport(
                location=loc,
                current=None,
                hourly=[],
                daily=[],
                historical=None,
                provenance=WeatherProvenance(
                    provider="Open-Meteo",
                    is_live=False,
                    is_unavailable=True,
                    source_type="unavailable",
                    missing_variables=["temperature_2m", "relative_humidity_2m", "precipitation"],
                ),
            )

    def _attach_historical_analysis(self, report: WeatherReport) -> None:
        """Compute and attach historical climate comparison to report."""
        if not report.daily:
            return

        # Calculate current window mean temp and cumulative rain
        mean_temps = [d.mean_temperature_c for d in report.daily if d.mean_temperature_c is not None]
        if not mean_temps:
            mean_temps = [(d.max_temperature_c + d.min_temperature_c) / 2.0 for d in report.daily]
        
        current_mean_t = sum(mean_temps) / len(mean_temps) if mean_temps else 25.0
        current_rain_sum = sum(d.precipitation_sum_mm for d in report.daily)

        # Count consecutive dry days (< 1.0 mm)
        dry_days = 0
        for d in report.daily:
            if d.precipitation_sum_mm < 1.0:
                dry_days += 1
            else:
                break

        comparison = self.historical_service.get_historical_anomaly(
            location=report.location,
            current_mean_temp=current_mean_t,
            current_cumulative_rain_mm=current_rain_sum,
            consecutive_dry_days=dry_days,
        )
        report.historical = comparison
