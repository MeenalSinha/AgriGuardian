"""Unit tests for weather alert lifecycle, deduplication, and acknowledgment management."""

import pytest
from backend.app.schemas.weather_alert import (
    AlertStatus,
    HazardType,
    RiskLevel,
    WeatherAlert,
)
from backend.app.services.weather.alert_manager import AlertManager


@pytest.fixture
def manager():
    m = AlertManager()
    m.clear()
    return m


def test_alert_registration_and_activation(manager):
    """Verify new alerts are registered with ACTIVE status."""
    alert = WeatherAlert(
        alert_id="ALT-TEST-01",
        hazard_type=HazardType.HEAT_STRESS,
        risk_level=RiskLevel.HIGH,
        title="Heat Stress Alert",
        summary="High temperatures expected.",
        crop="tomato",
        explanation="Testing rationale.",
        location_district="Meerut",
    )

    reg = manager.register_alerts([alert])
    assert len(reg) == 1
    assert reg[0].status == AlertStatus.ACTIVE

    active = manager.get_active_alerts(district="Meerut")
    assert len(active) == 1
    assert active[0].alert_id == "ALT-TEST-01"


def test_alert_deduplication_and_update(manager):
    """Verify duplicate alert updates existing alert instead of creating a second entry."""
    alert1 = WeatherAlert(
        alert_id="ALT-HEAT-1",
        hazard_type=HazardType.HEAT_STRESS,
        risk_level=RiskLevel.MODERATE,
        title="Heat Stress",
        summary="Moderate heat",
        crop="tomato",
        explanation="Rationale",
        location_district="Meerut",
    )
    manager.register_alerts([alert1])

    # Trigger same alert with escalated risk
    alert2 = WeatherAlert(
        alert_id="ALT-HEAT-2",
        hazard_type=HazardType.HEAT_STRESS,
        risk_level=RiskLevel.HIGH,
        title="Heat Stress Escalated",
        summary="Severe heatwave",
        crop="tomato",
        explanation="Rationale",
        location_district="Meerut",
    )
    reg = manager.register_alerts([alert2])
    assert len(reg) == 1
    assert reg[0].alert_id == "ALT-HEAT-1"  # Preserved original ID
    assert reg[0].risk_level == RiskLevel.HIGH  # Escalated level
    assert reg[0].status == AlertStatus.UPDATED

    # Still only 1 total alert in memory
    assert len(manager.get_all_alerts()) == 1


def test_alert_acknowledgment_and_dismissal(manager):
    """Verify acknowledgment and dismissal state transitions."""
    alert = WeatherAlert(
        alert_id="ALT-TEST-ACK",
        hazard_type=HazardType.HEAVY_RAINFALL,
        risk_level=RiskLevel.HIGH,
        title="Rain Warning",
        summary="Heavy rain forecast.",
        explanation="Rationale",
    )
    manager.register_alerts([alert])

    # Acknowledge
    assert manager.acknowledge_alert("ALT-TEST-ACK") is True
    assert manager.get_active_alerts()[0].acknowledged is True

    # Dismiss
    assert manager.dismiss_alert("ALT-TEST-ACK") is True
    assert len(manager.get_active_alerts()) == 0  # Dismissed alerts no longer appear in active list
