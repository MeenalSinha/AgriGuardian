"""Unit tests for farm location management and spatial precision."""

import pytest
from backend.app.schemas.location import LocationPrecision, LocationRecord
from backend.app.services.weather.location_manager import LocationManager


@pytest.fixture
def loc_manager():
    return LocationManager()


def test_preset_districts_listing(loc_manager):
    """Verify supported states and district preset queries."""
    states = loc_manager.get_supported_states()
    assert "Uttar Pradesh" in states
    assert "Punjab" in states
    assert "Karnataka" in states
    assert len(states) >= 10

    up_districts = loc_manager.get_districts_for_state("Uttar Pradesh")
    assert "Meerut" in up_districts
    assert "Varanasi" in up_districts


def test_resolve_preset_district(loc_manager):
    """Verify resolving preset district into LocationRecord."""
    rec = loc_manager.resolve_preset_district("Uttar Pradesh", "Meerut")
    assert isinstance(rec, LocationRecord)
    assert rec.district == "Meerut"
    assert rec.state == "Uttar Pradesh"
    assert rec.precision == LocationPrecision.DISTRICT_CENTROID.value
    assert rec.is_approximate is True
    assert 28.0 <= rec.latitude <= 30.0
    assert 76.0 <= rec.longitude <= 79.0


def test_resolve_preset_invalid_district(loc_manager):
    """Verify exception raised for invalid state/district combination."""
    with pytest.raises(ValueError):
        loc_manager.resolve_preset_district("NonExistentState", "Atlantis")


def test_resolve_coordinates_map_pin(loc_manager):
    """Verify manual map pin coordinate resolution and nearest district tagging."""
    # Near Meerut (28.98, 77.70)
    rec = loc_manager.resolve_coordinates(28.9900, 77.7100, source="map_click")
    assert rec.precision == LocationPrecision.MANUAL_PIN.value
    assert rec.source == "map_click"
    assert "Meerut" in rec.display_name


def test_gps_privacy_consent(loc_manager):
    """Verify GPS coordinates require explicit consent, falling back to default if absent."""
    # Without consent: must reject GPS and return safe default
    fallback_rec = loc_manager.resolve_coordinates(28.5, 77.2, source="gps", consent_granted=False)
    assert fallback_rec.precision == LocationPrecision.DISTRICT_CENTROID.value
    assert fallback_rec.is_approximate is True

    # With consent: returns exact GPS precision
    gps_rec = loc_manager.resolve_coordinates(28.5, 77.2, source="gps", consent_granted=True)
    assert gps_rec.precision == LocationPrecision.EXACT_GPS.value
    assert gps_rec.source == "gps"
    assert gps_rec.is_approximate is False


def test_coordinate_validation():
    """Verify coordinate boundary validation."""
    with pytest.raises(ValueError):
        LocationRecord(
            location_id="LOC-INVALID",
            display_name="Invalid Lat",
            state="UP",
            district="Meerut",
            latitude=95.0,  # Invalid: > 90
            longitude=77.0,
        )

    with pytest.raises(ValueError):
        LocationRecord(
            location_id="LOC-INVALID",
            display_name="Invalid Lon",
            state="UP",
            district="Meerut",
            latitude=28.0,
            longitude=195.0,  # Invalid: > 180
        )
