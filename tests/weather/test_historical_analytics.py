"""Unit tests for historical climate reanalysis baseline and anomaly calculation."""

import pytest
from backend.app.integrations.weather.historical_weather import HistoricalWeatherService
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import HistoricalWeatherComparison


@pytest.fixture
def sample_location():
    return LocationRecord(
        location_id="LOC-HIS-01",
        display_name="Karnal, Haryana",
        state="Haryana",
        district="Karnal",
        latitude=29.6857,
        longitude=76.9905,
    )


def test_calculate_positive_temperature_anomaly(sample_location, test_settings):
    """Verify temperature anomaly calculation when current period is warmer than baseline."""
    service = HistoricalWeatherService(settings=test_settings)

    baseline = {
        "mean_temp": 24.0,
        "mean_rain": 10.0,
        "source": "Open-Meteo ERA5 / ERA5-Land Reanalysis (Modeled)",
    }

    comp = service._calculate_anomaly(
        baseline_data=baseline,
        current_mean_temp=27.5,
        current_cumulative_rain_mm=2.0,
        consecutive_dry_days=9,
        ref_years_count=5,
    )

    assert isinstance(comp, HistoricalWeatherComparison)
    assert comp.temperature_anomaly_c == 3.5  # 27.5 - 24.0
    assert comp.rainfall_anomaly_pct == -80.0  # (2.0 - 10.0)/10.0 * 100
    assert comp.consecutive_dry_days == 9
    assert comp.is_modeled_reanalysis is True
    assert "Open-Meteo ERA5" in comp.data_source
    assert "warmer than normal" in comp.notes.lower()
    assert "dry-spell" in comp.notes.lower()


def test_calculate_rainfall_excess_anomaly(sample_location, test_settings):
    """Verify excess rainfall percentage calculation."""
    service = HistoricalWeatherService(settings=test_settings)

    baseline = {
        "mean_temp": 28.0,
        "mean_rain": 20.0,
        "source": "Open-Meteo ERA5 / ERA5-Land Reanalysis (Modeled)",
    }

    comp = service._calculate_anomaly(
        baseline_data=baseline,
        current_mean_temp=28.2,
        current_cumulative_rain_mm=50.0,
        consecutive_dry_days=0,
        ref_years_count=5,
    )

    assert comp.rainfall_anomaly_pct == 150.0  # +150% excess
    assert "excess rainfall" in comp.notes.lower()

