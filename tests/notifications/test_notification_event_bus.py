"""Unit and integration tests for NotificationEventBus and TemplateService."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord, UserRole
from backend.app.schemas.notification import (
    NotificationCategory,
    NotificationChannel,
    NotificationEvent,
    NotificationPriority,
)
from backend.app.services.notifications.notification_event_bus import NotificationEventBus
from backend.app.services.notifications.notification_policy_engine import NotificationPolicyEngine
from backend.app.services.notifications.notification_template_service import NotificationTemplateService


@pytest.fixture
def event_bus_fixture():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    # Seed user and farmer
    user = UserRecord(
        user_id="usr_test_01",
        username="farmer_test_01",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-09T00:00:00",
    )
    repo.create_user(user)

    farmer = FarmerProfile(
        farmer_id="fmr_test_01",
        user_id="usr_test_01",
        full_name="Harpreet Singh",
        phone_number="9876543210",
        state="Punjab",
        district="Ludhiana",
        village="Gill",
        preferred_language="hi-IN",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farmer_profile(farmer)

    policy_engine = NotificationPolicyEngine(repo)
    template_service = NotificationTemplateService(repo)
    bus = NotificationEventBus(
        repo=repo,
        policy_engine=policy_engine,
        template_service=template_service,
    )

    yield {
        "repo": repo,
        "bus": bus,
        "farmer": farmer,
        "template_service": template_service,
    }

    try:
        os.remove(path)
    except Exception:
        pass


def test_weather_alert_event_dispatched_and_rendered(event_bus_fixture):
    """Verify extreme weather event generates bilingual notification and in-app entry."""
    bus = event_bus_fixture["bus"]
    repo = event_bus_fixture["repo"]

    res = bus.emit_weather_alert_event(
        farmer_id="fmr_test_01",
        hazard_type="Hailstorm & Gusty Winds",
        severity="Critical",
        summary="Severe hailstorm predicted in next 6 hours.",
        village_or_district="Gill, Ludhiana",
    )

    assert res["status"] == "dispatched"
    notif_id = res["notification_id"]
    notif = repo.get_notification(notif_id)

    assert notif is not None
    assert notif.priority == NotificationPriority.EMERGENCY
    assert notif.category == NotificationCategory.WEATHER_ALERT
    assert "Hailstorm" in notif.title_en
    assert "गंभीर मौसम चेतावनी" in notif.title_hi
    assert "Gill, Ludhiana" in notif.body_en
    assert "Gill, Ludhiana" in notif.body_hi


def test_market_price_surge_event_generation(event_bus_fixture):
    """Verify market surge event produces correct bilingual copy and action link."""
    bus = event_bus_fixture["bus"]
    repo = event_bus_fixture["repo"]

    res = bus.emit_market_price_event(
        farmer_id="fmr_test_01",
        crop_name="Wheat",
        mandi_name="Khanna APMC",
        current_price=2450.0,
        previous_price=2200.0,
        price_change_pct=11.36,
        is_surge=True,
    )

    assert res["status"] == "dispatched"
    notif = repo.get_notification(res["notification_id"])
    assert notif is not None
    assert "Wheat up 11.4%" in notif.title_en
    assert "गेहूं" in notif.title_hi or "Wheat" in notif.title_hi
    assert "₹2450.0/qtl" in notif.body_en
    assert notif.action_url == "/market"


def test_duplicate_events_suppressed_by_dedup_key(event_bus_fixture):
    """Verify duplicate non-emergency events with identical dedup_key are suppressed."""
    bus = event_bus_fixture["bus"]

    # First event succeeds
    res1 = bus.emit_market_price_event(
        farmer_id="fmr_test_01",
        crop_name="Tomato",
        mandi_name="Karnal APMC",
        current_price=1800.0,
        previous_price=1700.0,
        price_change_pct=5.8,
        is_surge=True,
    )
    assert res1["status"] == "dispatched"

    # Second identical event is suppressed
    res2 = bus.emit_market_price_event(
        farmer_id="fmr_test_01",
        crop_name="Tomato",
        mandi_name="Karnal APMC",
        current_price=1800.0,
        previous_price=1700.0,
        price_change_pct=5.8,
        is_surge=True,
    )
    assert res2["status"] == "suppressed"
    assert "Duplicate notification" in res2["reason"]


def test_expert_review_event_renders_verified_provenance(event_bus_fixture):
    """Verify expert review completion informs farmer with agronomist credentials."""
    bus = event_bus_fixture["bus"]
    repo = event_bus_fixture["repo"]

    res = bus.emit_expert_review_event(
        farmer_id="fmr_test_01",
        case_id="cas_12345",
        crop_name="Mustard",
        expert_name="Dr. Rajinder Gupta",
        institution="ICAR-IARI",
        is_secondary_approved=False,
    )

    assert res["status"] == "dispatched"
    notif = repo.get_notification(res["notification_id"])
    assert notif is not None
    assert notif.category == NotificationCategory.EXPERT_REVIEW
    assert "Dr. Rajinder Gupta" in notif.body_en
    assert "ICAR-IARI" in notif.body_en
