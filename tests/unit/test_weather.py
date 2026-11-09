"""Unit tests for Open-Meteo weather service and error fallbacks."""

import pytest
from unittest.mock import MagicMock, patch
from backend.app.integrations.weather.open_meteo import WeatherService
from backend.app.schemas.advisory import WeatherContext


def test_weather_success_parsing(test_settings):
    """Verify parsing of valid Open-Meteo response."""
    service = WeatherService(settings=test_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 32.5,
            "relative_humidity_2m": 72.0,
            "precipitation": 4.5,
        },
        "daily": {
            "temperature_2m_max": [35.0, 36.0, 34.0],
            "temperature_2m_min": [24.0, 25.0, 23.0],
            "precipitation_sum": [5.0, 0.0, 2.0],
        },
    }

    with patch("requests.get", return_value=mock_resp):
        w = service.get_weather(lat=28.61, lon=77.20)
        assert isinstance(w, WeatherContext)
        assert w.temperature == 32.5
        assert w.humidity == 72.0
        assert w.precipitation == 4.5
        assert w.rain_recent is True
        assert w.is_live is True


def test_weather_fallback_on_network_error(test_settings):
    """Verify graceful degraded fallback when network call fails."""
    service = WeatherService(settings=test_settings)

    with patch("requests.get", side_effect=Exception("Network disconnected")):
        w = service.get_weather()
        assert isinstance(w, WeatherContext)
        assert w.temperature == 28.0
        assert w.is_live is False
        assert "estimated baseline" in w.summary

