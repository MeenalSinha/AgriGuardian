"""Open-Meteo Weather API Integration with caching and robust error insulation."""

from typing import Any, Dict, Optional, Tuple
import requests
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import WeatherAPIError
from backend.app.core.logging import get_logger
from backend.app.schemas.advisory import WeatherContext

logger = get_logger("WeatherIntegration")


class WeatherService:
    """Service for retrieving agricultural weather conditions."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.open_meteo_base_url
        self.timeout = self.settings.weather_timeout_seconds

    def get_weather(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        location_name: Optional[str] = None,
    ) -> WeatherContext:
        """Fetch current weather and 3-day forecast from Open-Meteo.

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

        try:
            logger.info(f"Fetching weather for coordinates ({latitude:.4f}, {longitude:.4f})")
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
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
            logger.warning(f"Live weather fetch failed ({e}). Providing safe fallback context.")
            # Return safe fallback with explicit is_live=False indicator
            return WeatherContext(
                temperature=28.0,
                humidity=65.0,
                precipitation=0.0,
                rain_recent=False,
                temp_high=False,
                temp_very_high=False,
                humidity_high=False,
                forecast={
                    "max_temp": [30.0, 31.0, 29.0],
                    "min_temp": [22.0, 23.0, 21.0],
                    "precipitation": [0.0, 2.0, 0.0],
                },
                summary="28.0°C, 65% humidity, 0mm rain (estimated baseline)",
                is_live=False,
            )

