"""Standardized agricultural weather data models with provenance and unit normalization."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.advisory import WeatherContext
from backend.app.schemas.location import LocationRecord

# WMO Weather Interpretation Codes (WW) mapping to English and Hindi descriptions
WMO_WEATHER_CODES: Dict[int, Dict[str, str]] = {
    0: {"en": "Clear sky", "hi": "साफ आसमान", "icon": "☀️"},
    1: {"en": "Mainly clear", "hi": "मुख्य रूप से साफ", "icon": "🌤️"},
    2: {"en": "Partly cloudy", "hi": "आंशिक रूप से बादल", "icon": "⛅"},
    3: {"en": "Overcast", "hi": "घने बादल", "icon": "☁️"},
    45: {"en": "Fog", "hi": "कोहरा", "icon": "🌫️"},
    48: {"en": "Depositing rime fog", "hi": "घना बर्फीला कोहरा", "icon": "🌫️"},
    51: {"en": "Light drizzle", "hi": "हल्की बूंदाबांदी", "icon": "🌦️"},
    53: {"en": "Moderate drizzle", "hi": "मध्यम बूंदाबांदी", "icon": "🌦️"},
    55: {"en": "Dense drizzle", "hi": "घनी बूंदाबांदी", "icon": "🌧️"},
    61: {"en": "Slight rain", "hi": "हल्की बारिश", "icon": "🌧️"},
    63: {"en": "Moderate rain", "hi": "मध्यम बारिश", "icon": "🌧️"},
    65: {"en": "Heavy rain", "hi": "भारी बारिश", "icon": "⛈️"},
    71: {"en": "Slight snow fall", "hi": "हल्की बर्फबारी", "icon": "🌨️"},
    73: {"en": "Moderate snow fall", "hi": "मध्यम बर्फबारी", "icon": "🌨️"},
    75: {"en": "Heavy snow fall", "hi": "भारी बर्फबारी", "icon": "❄️"},
    80: {"en": "Slight rain showers", "hi": "हल्की फुहारें", "icon": "🌦️"},
    81: {"en": "Moderate rain showers", "hi": "मध्यम फुहारें", "icon": "🌧️"},
    82: {"en": "Violent rain showers", "hi": "अति मूसलाधार बारिश", "icon": "⛈️"},
    95: {"en": "Thunderstorm", "hi": "गरज-चमक के साथ आंधी", "icon": "🌩️"},
    96: {"en": "Thunderstorm with slight hail", "hi": "ओलावृष्टि के साथ तूफान", "icon": "⛈️"},
    99: {"en": "Thunderstorm with heavy hail", "hi": "भारी ओलावृष्टि के साथ तूफान", "icon": "⛈️"},
}


class CurrentWeather(BaseModel):
    """Real-time ambient weather observation or current model step."""
    temperature_c: float = Field(..., description="Air temperature at 2m in Celsius")
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity in percentage")
    dew_point_c: Optional[float] = Field(None, description="Dew point temperature at 2m in Celsius")
    apparent_temperature_c: Optional[float] = Field(None, description="Feels-like temperature accounting for wind and humidity")
    precipitation_mm: float = Field(0.0, ge=0.0, description="Current precipitation in mm")
    rain_mm: Optional[float] = Field(None, ge=0.0, description="Current liquid rain in mm")
    weather_code: int = Field(0, description="WMO standard weather condition code")
    condition_text: str = Field("Clear sky", description="Human-readable weather description")
    wind_speed_kmh: float = Field(0.0, ge=0.0, description="Wind speed at 10m in km/h")
    wind_direction_deg: Optional[float] = Field(None, ge=0.0, le=360.0, description="Dominant wind direction in degrees")
    wind_gusts_kmh: Optional[float] = Field(None, ge=0.0, description="Maximum wind gusts in km/h")
    cloud_cover_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Cloud cover percentage")
    observation_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HourlyWeatherPoint(BaseModel):
    """Hourly forecasted meteorological data point."""
    time_iso: str = Field(..., description="Timestamp of the forecast step (ISO 8601)")
    temperature_c: float = Field(..., description="Temperature at 2m in Celsius")
    relative_humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage")
    dew_point_c: Optional[float] = Field(None, description="Dew point in Celsius")
    apparent_temperature_c: Optional[float] = Field(None, description="Apparent temperature in Celsius")
    precipitation_mm: float = Field(0.0, ge=0.0, description="Expected precipitation in mm")
    precipitation_probability_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Rainfall probability %")
    wind_speed_kmh: float = Field(0.0, ge=0.0, description="Wind speed at 10m in km/h")
    wind_gusts_kmh: Optional[float] = Field(None, ge=0.0, description="Wind gusts in km/h")
    shortwave_radiation_wm2: Optional[float] = Field(None, description="Solar irradiance in W/m²")
    et0_mm: Optional[float] = Field(None, description="Hourly FAO reference evapotranspiration in mm")
    weather_code: int = Field(0, description="WMO weather code")
    condition_text: str = Field("Clear", description="Weather condition description")


class DailyForecastPoint(BaseModel):
    """Daily aggregated meteorological forecast."""
    date_str: str = Field(..., description="Date (YYYY-MM-DD)")
    min_temperature_c: float = Field(..., description="Minimum daily temperature in Celsius")
    max_temperature_c: float = Field(..., description="Maximum daily temperature in Celsius")
    mean_temperature_c: Optional[float] = Field(None, description="Mean temperature in Celsius")
    precipitation_sum_mm: float = Field(0.0, ge=0.0, description="Accumulated precipitation in mm")
    rain_sum_mm: Optional[float] = Field(None, ge=0.0, description="Accumulated liquid rain in mm")
    precipitation_probability_max_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Peak daily rain probability %")
    precipitation_hours: Optional[float] = Field(None, ge=0.0, le=24.0, description="Hours with rainfall")
    max_wind_speed_kmh: float = Field(0.0, ge=0.0, description="Maximum sustained wind speed in km/h")
    max_wind_gusts_kmh: Optional[float] = Field(None, ge=0.0, description="Maximum wind gust in km/h")
    dominant_wind_direction_deg: Optional[float] = Field(None, description="Dominant wind direction degrees")
    et0_evapotranspiration_mm: Optional[float] = Field(None, description="Daily FAO-56 reference evapotranspiration in mm")
    sunshine_duration_s: Optional[float] = Field(None, description="Sunshine duration in seconds")
    weather_code: int = Field(0, description="Representative daily WMO weather code")
    condition_text: str = Field("Fair", description="Representative condition")


class HistoricalWeatherComparison(BaseModel):
    """Historical baseline comparison for agricultural climate anomaly detection."""
    reference_period_years: int = Field(5, description="Number of baseline years analyzed (e.g. 5-year climatological normal)")
    baseline_mean_temperature_c: float = Field(..., description="Historical average mean temperature for this calendar window")
    current_period_mean_temperature_c: float = Field(..., description="Observed / forecast mean temperature for current period")
    temperature_anomaly_c: float = Field(..., description="Temperature departure from historical baseline in °C (+ above normal, - below)")
    baseline_total_rainfall_mm: float = Field(..., description="Historical average cumulative precipitation for this period")
    current_period_rainfall_mm: float = Field(..., description="Current observed/forecast cumulative precipitation")
    rainfall_anomaly_pct: float = Field(..., description="Percentage departure from historical rainfall baseline (-100% to +inf)")
    consecutive_dry_days: int = Field(0, ge=0, description="Count of consecutive rainless days (< 1mm)")
    data_source: str = Field(
        "Open-Meteo ERA5 / ERA5-Land Reanalysis (Modeled)",
        description="Dataset origin. Note: Reanalysis data is modeled and gridded, not physical on-farm station observation."
    )
    is_modeled_reanalysis: bool = Field(True, description="Explicit flag clarifying this is reanalysis rather than direct farm sensor")
    notes: str = Field("", description="Agronomic contextual commentary on observed anomaly")


class WeatherProvenance(BaseModel):
    """Metadata tracking provider identity, timestamps, and data freshness."""
    provider: str = Field("Open-Meteo", description="Meteorological API or archival provider")
    model_name: Optional[str] = Field("best_match", description="Numerical Weather Prediction model (e.g. GFS, ECMWF, ICON)")
    retrieval_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    forecast_issue_time: Optional[str] = Field(None, description="Model execution initialization timestamp")
    data_freshness_seconds: float = Field(0.0, description="Age of data in seconds since retrieval")
    is_live: bool = Field(True, description="True if retrieved live from weather provider; False if unavailable or degraded")
    is_stale: bool = Field(False, description="True if cached data exceeds recommended TTL window")
    is_unavailable: bool = Field(False, description="True if weather services could not be reached and no cache exists")
    source_type: str = Field(
        "live_forecast",
        description="Data type: 'live_forecast', 'cached_forecast', 'reanalysis_archive', 'official_bulletin', 'unavailable'"
    )
    missing_variables: List[str] = Field(default_factory=list, description="Variables not available from provider")


class WeatherReport(BaseModel):
    """Comprehensive agricultural weather intelligence payload."""
    location: LocationRecord = Field(..., description="Target farm location and spatial precision metadata")
    current: Optional[CurrentWeather] = Field(None, description="Current ambient conditions")
    hourly: List[HourlyWeatherPoint] = Field(default_factory=list, description="Hourly forecast steps")
    daily: List[DailyForecastPoint] = Field(default_factory=list, description="Multi-day daily forecast steps")
    historical: Optional[HistoricalWeatherComparison] = Field(None, description="Historical climate baseline & anomaly metrics")
    provenance: WeatherProvenance = Field(default_factory=WeatherProvenance, description="Data provenance and reliability metadata")

    def to_legacy_context(self) -> WeatherContext:
        """Convert to backward-compatible WeatherContext schema for Phase 0-3 components."""
        if self.current and not self.provenance.is_unavailable:
            temp = self.current.temperature_c
            humidity = self.current.relative_humidity_pct
            precip = self.current.precipitation_mm
            rain_recent = precip > 1.0 or any(d.precipitation_sum_mm > 1.0 for d in self.daily[:1])
            is_live = self.provenance.is_live
            summary = f"{temp:.1f}°C, {humidity:.0f}% humidity, {precip:.1f}mm rain"
            max_temps = [d.max_temperature_c for d in self.daily[:3]] if self.daily else []
            min_temps = [d.min_temperature_c for d in self.daily[:3]] if self.daily else []
            precips = [d.precipitation_sum_mm for d in self.daily[:3]] if self.daily else []

            return WeatherContext(
                temperature=temp,
                humidity=humidity,
                precipitation=precip,
                rain_recent=rain_recent,
                temp_high=temp > 35.0,
                temp_very_high=temp > 40.0,
                humidity_high=humidity > 80.0,
                forecast={
                    "max_temp": max_temps,
                    "min_temp": min_temps,
                    "precipitation": precips,
                },
                summary=summary,
                is_live=is_live,
                is_available=True,
            )
        else:
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

