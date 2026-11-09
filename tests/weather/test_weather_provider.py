"""Unit tests for weather data normalization, caching, and error resilience."""

import time
from unittest.mock import MagicMock, patch
import pytest
from backend.app.integrations.weather.open_meteo import WeatherService
from backend.app.integrations.weather.weather_cache import WeatherCache
from backend.app.integrations.weather.weather_normalizer import normalize_open_meteo_response
from backend.app.schemas.advisory import WeatherContext
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import WeatherReport


@pytest.fixture
def sample_location():
    return LocationRecord(
        location_id="LOC-TEST-01",
        display_name="Meerut, Uttar Pradesh",
        state="Uttar Pradesh",
        district="Meerut",
        latitude=28.9845,
        longitude=77.7064,
        elevation_m=219.0,
    )


@pytest.fixture
def mock_open_meteo_json():
    return {
        "current": {
            "time": "2026-11-09T12:00:00Z",
            "temperature_2m": 31.5,
            "relative_humidity_2m": 68.0,
            "dew_point_2m": 24.5,
            "apparent_temperature": 34.0,
            "precipitation": 0.0,
            "weather_code": 1,
            "wind_speed_10m": 12.5,
            "wind_direction_10m": 180.0,
            "wind_gusts_10m": 18.0,
            "cloud_cover": 25.0,
        },
        "hourly": {
            "time": ["2026-11-09T12:00", "2026-11-09T13:00"],
            "temperature_2m": [31.5, 32.0],
            "relative_humidity_2m": [68.0, 65.0],
            "precipitation": [0.0, 0.0],
            "weather_code": [1, 1],
            "wind_speed_10m": [12.5, 14.0],
        },
        "daily": {
            "time": ["2026-11-09", "2026-11-10", "2026-11-11"],
            "temperature_2m_max": [34.0, 35.5, 36.0],
            "temperature_2m_min": [21.0, 22.0, 23.0],
            "temperature_2m_mean": [27.5, 28.5, 29.5],
            "precipitation_sum": [0.0, 0.0, 2.5],
            "weather_code": [1, 0, 61],
            "wind_speed_10m_max": [15.0, 18.0, 22.0],
            "et0_fao_evapotranspiration": [4.5, 4.8, 5.1],
        },
    }


def test_weather_normalization(mock_open_meteo_json, sample_location):
    """Verify raw Open-Meteo payload normalization into typed WeatherReport."""
    report = normalize_open_meteo_response(mock_open_meteo_json, sample_location)
    assert isinstance(report, WeatherReport)
    assert report.current is not None
    assert report.current.temperature_c == 31.5
    assert report.current.relative_humidity_pct == 68.0
    assert report.current.condition_text == "Mainly clear"
    assert len(report.daily) == 3
    assert report.daily[0].max_temperature_c == 34.0
    assert report.daily[0].et0_evapotranspiration_mm == 4.5
    assert report.provenance.provider == "Open-Meteo"
    assert report.provenance.is_live is True


def test_weather_cache_spatial_snapping():
    """Verify cache snaps close coordinates to spatial grid and avoids duplicate calls."""
    cache = WeatherCache(default_ttl_seconds=10, spatial_grid_deg=0.05)

    data = {"forecast": "sunny"}
    # Store at (28.981, 77.701)
    cache.set(28.981, 77.701, data)

    # Lookup at very close coordinate (28.983, 77.703) -> snaps to same 0.05 grid cell
    hit = cache.get(28.983, 77.703)
    assert hit == data
    assert cache.stats()["hits"] == 1

    # Lookup at distant coordinate (29.50, 78.50) -> miss
    miss = cache.get(29.50, 78.50)
    assert miss is None
    assert cache.stats()["misses"] == 1


def test_weather_cache_ttl_expiration():
    """Verify cache entries expire after TTL window."""
    cache = WeatherCache(default_ttl_seconds=1)
    cache.set(28.0, 77.0, "temp_data", ttl_seconds=1)

    assert cache.get(28.0, 77.0) == "temp_data"
    time.sleep(1.1)
    assert cache.get(28.0, 77.0) is None


def test_weather_service_offline_resilience(sample_location, test_settings):
    """Verify network disconnect produces transparent unavailable report without fake values."""
    from backend.app.integrations.weather.weather_cache import weather_cache
    weather_cache.invalidate()

    service = WeatherService(settings=test_settings)

    with patch("requests.get", side_effect=Exception("Connection timed out")):
        report = service.get_weather_report(location=sample_location)
        assert isinstance(report, WeatherReport)
        assert report.provenance.is_live is False
        assert report.provenance.is_unavailable is True
        assert report.provenance.source_type == "unavailable"
        assert report.current is None


def test_weather_report_to_legacy_context(mock_open_meteo_json, sample_location):
    """Verify 100% backward compatibility of WeatherReport.to_legacy_context()."""
    report = normalize_open_meteo_response(mock_open_meteo_json, sample_location)
    legacy = report.to_legacy_context()

    assert isinstance(legacy, WeatherContext)
    assert legacy.temperature == 31.5
    assert legacy.humidity == 68.0
    assert legacy.precipitation == 0.0
    assert legacy.is_live is True
    assert "max_temp" in legacy.forecast
    assert len(legacy.forecast["max_temp"]) == 3
