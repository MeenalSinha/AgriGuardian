"""Normalizer for converting raw Open-Meteo responses to typed WeatherReport models."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import (
    WMO_WEATHER_CODES,
    CurrentWeather,
    DailyForecastPoint,
    HourlyWeatherPoint,
    WeatherProvenance,
    WeatherReport,
)


def _get_wmo_text(code: int, lang: str = "en") -> str:
    """Return text description for WMO weather code."""
    meta = WMO_WEATHER_CODES.get(code, {"en": "Unknown", "hi": "अज्ञात", "icon": "❓"})
    return meta.get(lang, "Unknown")


def normalize_open_meteo_response(
    raw_json: Dict[str, Any],
    location: LocationRecord,
    is_cached: bool = False,
    data_age_seconds: float = 0.0,
) -> WeatherReport:
    """Transform Open-Meteo API response into structured WeatherReport.
    
    Args:
        raw_json: Raw JSON parsed response from Open-Meteo forecast endpoint.
        location: Target agricultural location metadata.
        is_cached: Flag indicating whether response was served from cache.
        data_age_seconds: Cache age in seconds.
        
    Returns:
        Structured WeatherReport object.
    """
    current_data = raw_json.get("current", {})
    hourly_data = raw_json.get("hourly", {})
    daily_data = raw_json.get("daily", {})

    # 1. Parse Current Weather
    current = None
    if current_data:
        code = int(current_data.get("weather_code", 0))
        current = CurrentWeather(
            temperature_c=float(current_data.get("temperature_2m", 25.0)),
            relative_humidity_pct=float(current_data.get("relative_humidity_2m", 50.0)),
            dew_point_c=float(current_data.get("dew_point_2m")) if "dew_point_2m" in current_data and current_data["dew_point_2m"] is not None else None,
            apparent_temperature_c=float(current_data.get("apparent_temperature")) if "apparent_temperature" in current_data and current_data["apparent_temperature"] is not None else None,
            precipitation_mm=max(0.0, float(current_data.get("precipitation", 0.0))),
            rain_mm=max(0.0, float(current_data.get("rain", 0.0))) if "rain" in current_data and current_data["rain"] is not None else None,
            weather_code=code,
            condition_text=_get_wmo_text(code),
            wind_speed_kmh=max(0.0, float(current_data.get("wind_speed_10m", 0.0))),
            wind_direction_deg=float(current_data.get("wind_direction_10m")) if "wind_direction_10m" in current_data and current_data["wind_direction_10m"] is not None else None,
            wind_gusts_kmh=float(current_data.get("wind_gusts_10m")) if "wind_gusts_10m" in current_data and current_data["wind_gusts_10m"] is not None else None,
            cloud_cover_pct=float(current_data.get("cloud_cover")) if "cloud_cover" in current_data and current_data["cloud_cover"] is not None else None,
            observation_time=current_data.get("time", datetime.now(timezone.utc).isoformat()),
        )

    # 2. Parse Hourly Forecasts
    hourly_list: List[HourlyWeatherPoint] = []
    times = hourly_data.get("time", [])
    temps = hourly_data.get("temperature_2m", [])
    humidities = hourly_data.get("relative_humidity_2m", [])
    dew_points = hourly_data.get("dew_point_2m", [])
    apparent_temps = hourly_data.get("apparent_temperature", [])
    precips = hourly_data.get("precipitation", [])
    precip_probs = hourly_data.get("precipitation_probability", [])
    winds = hourly_data.get("wind_speed_10m", [])
    gusts = hourly_data.get("wind_gusts_10m", [])
    radiations = hourly_data.get("shortwave_radiation", [])
    et0s = hourly_data.get("et0_fao_evapotranspiration", [])
    codes = hourly_data.get("weather_code", [])

    for i in range(len(times)):
        code_val = int(codes[i]) if i < len(codes) and codes[i] is not None else 0
        hourly_list.append(
            HourlyWeatherPoint(
                time_iso=str(times[i]),
                temperature_c=float(temps[i]) if i < len(temps) and temps[i] is not None else 25.0,
                relative_humidity_pct=float(humidities[i]) if i < len(humidities) and humidities[i] is not None else 50.0,
                dew_point_c=float(dew_points[i]) if i < len(dew_points) and dew_points[i] is not None else None,
                apparent_temperature_c=float(apparent_temps[i]) if i < len(apparent_temps) and apparent_temps[i] is not None else None,
                precipitation_mm=max(0.0, float(precips[i])) if i < len(precips) and precips[i] is not None else 0.0,
                precipitation_probability_pct=float(precip_probs[i]) if i < len(precip_probs) and precip_probs[i] is not None else None,
                wind_speed_kmh=max(0.0, float(winds[i])) if i < len(winds) and winds[i] is not None else 0.0,
                wind_gusts_kmh=float(gusts[i]) if i < len(gusts) and gusts[i] is not None else None,
                shortwave_radiation_wm2=float(radiations[i]) if i < len(radiations) and radiations[i] is not None else None,
                et0_mm=float(et0s[i]) if i < len(et0s) and et0s[i] is not None else None,
                weather_code=code_val,
                condition_text=_get_wmo_text(code_val),
            )
        )

    # 3. Parse Daily Forecasts
    daily_list: List[DailyForecastPoint] = []
    d_times = daily_data.get("time", [])
    max_temps = daily_data.get("temperature_2m_max", [])
    min_temps = daily_data.get("temperature_2m_min", [])
    mean_temps = daily_data.get("temperature_2m_mean", [])
    d_precip_sums = daily_data.get("precipitation_sum", [])
    rain_sums = daily_data.get("rain_sum", [])
    d_precip_probs = daily_data.get("precipitation_probability_max", [])
    precip_hrs = daily_data.get("precipitation_hours", [])
    d_wind_max = daily_data.get("wind_speed_10m_max", [])
    d_gust_max = daily_data.get("wind_gusts_10m_max", [])
    wind_dirs = daily_data.get("wind_direction_10m_dominant", [])
    d_et0s = daily_data.get("et0_fao_evapotranspiration", [])
    sunshines = daily_data.get("sunshine_duration", [])
    d_codes = daily_data.get("weather_code", [])

    for i in range(len(d_times)):
        code_val = int(d_codes[i]) if i < len(d_codes) and d_codes[i] is not None else 0
        daily_list.append(
            DailyForecastPoint(
                date_str=str(d_times[i]),
                min_temperature_c=float(min_temps[i]) if i < len(min_temps) and min_temps[i] is not None else 20.0,
                max_temperature_c=float(max_temps[i]) if i < len(max_temps) and max_temps[i] is not None else 30.0,
                mean_temperature_c=float(mean_temps[i]) if i < len(mean_temps) and mean_temps[i] is not None else None,
                precipitation_sum_mm=max(0.0, float(d_precip_sums[i])) if i < len(d_precip_sums) and d_precip_sums[i] is not None else 0.0,
                rain_sum_mm=float(rain_sums[i]) if i < len(rain_sums) and rain_sums[i] is not None else None,
                precipitation_probability_max_pct=float(d_precip_probs[i]) if i < len(d_precip_probs) and d_precip_probs[i] is not None else None,
                precipitation_hours=float(precip_hrs[i]) if i < len(precip_hrs) and precip_hrs[i] is not None else None,
                max_wind_speed_kmh=max(0.0, float(d_wind_max[i])) if i < len(d_wind_max) and d_wind_max[i] is not None else 0.0,
                max_wind_gusts_kmh=float(d_gust_max[i]) if i < len(d_gust_max) and d_gust_max[i] is not None else None,
                dominant_wind_direction_deg=float(wind_dirs[i]) if i < len(wind_dirs) and wind_dirs[i] is not None else None,
                et0_evapotranspiration_mm=float(d_et0s[i]) if i < len(d_et0s) and d_et0s[i] is not None else None,
                sunshine_duration_s=float(sunshines[i]) if i < len(sunshines) and sunshines[i] is not None else None,
                weather_code=code_val,
                condition_text=_get_wmo_text(code_val),
            )
        )

    # 4. Provenance
    provenance = WeatherProvenance(
        provider="Open-Meteo",
        model_name="ECMWF_IFS / GFS (best_match)",
        retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
        data_freshness_seconds=data_age_seconds,
        is_live=not is_cached,
        is_stale=data_age_seconds > 3600,
        is_unavailable=False,
        source_type="cached_forecast" if is_cached else "live_forecast",
    )

    return WeatherReport(
        location=location,
        current=current,
        hourly=hourly_list,
        daily=daily_list,
        provenance=provenance,
    )
