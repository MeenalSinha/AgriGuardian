"""Unit tests for NotificationPolicyEngine, quiet hours, and channel selection."""

import os
import tempfile
from datetime import datetime
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord
from backend.app.schemas.notification import (
    NotificationCategory,
    NotificationChannel,
    NotificationChannelConsent,
    NotificationEvent,
    NotificationPreference,
    NotificationPriority,
)
from backend.app.services.notifications.notification_policy_engine import NotificationPolicyEngine


@pytest.fixture
def policy_fixture():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    now_str = datetime.now().isoformat()
    user = UserRecord(
        user_id="usr_policy_01",
        username="surjit_singh",
        password_hash="fakehash",
        salt="fakesalt",
        role="farmer",
        created_at=now_str,
    )
    repo.create_user(user)

    farmer = FarmerProfile(
        farmer_id="fmr_policy_01",
        user_id="usr_policy_01",
        full_name="Surjit Singh",
        phone_number="9876543210",
        preferred_language="hi-IN",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at=now_str,
        updated_at=now_str,
    )
    repo.create_farmer_profile(farmer)
    engine = NotificationPolicyEngine(repo)

    yield {
        "repo": repo,
        "engine": engine,
        "farmer": farmer,
    }

    try:
        os.remove(path)
    except Exception:
        pass


def test_quiet_hours_blocks_non_emergency_external_channels(policy_fixture):
    """Verify quiet hours (e.g. 23:30) restricts dispatch to in-app or defers noisy channels."""
    engine = policy_fixture["engine"]
    night_time = datetime(2026, 11, 9, 23, 30)  # 11:30 PM

    event = NotificationEvent(
        event_id="evt_night_01",
        event_type="market_price_surge",
        category=NotificationCategory.MARKET_INTELLIGENCE,
        priority=NotificationPriority.MEDIUM,
        farmer_id="fmr_policy_01",
        payload={"crop_name": "Wheat"},
    )

    eval_result = engine.evaluate(event, now=night_time)
    assert eval_result.allowed is True
    assert eval_result.is_quiet_hours is True
    assert eval_result.bypass_quiet_hours is False
    # Noisy channels like SMS/WhatsApp omitted during quiet hours
    assert eval_result.target_channels == [NotificationChannel.IN_APP]
    assert eval_result.defer_until is not None


def test_emergency_weather_overrides_quiet_hours(policy_fixture):
    """Verify emergency priority alert bypasses quiet hours and activates SMS."""
    engine = policy_fixture["engine"]
    night_time = datetime(2026, 11, 9, 2, 15)  # 2:15 AM

    event = NotificationEvent(
        event_id="evt_emergency_night",
        event_type="weather_alert_extreme",
        category=NotificationCategory.WEATHER_ALERT,
        priority=NotificationPriority.EMERGENCY,
        farmer_id="fmr_policy_01",
        payload={"hazard_type": "Flash Flood"},
    )

    eval_result = engine.evaluate(event, now=night_time)
    assert eval_result.allowed is True
    assert eval_result.is_quiet_hours is True
    assert eval_result.bypass_quiet_hours is True
    # Emergency forces external reachability
    assert NotificationChannel.SMS in eval_result.target_channels


def test_farmer_category_opt_out_suppresses_event(policy_fixture):
    """Verify when farmer disables market alerts in preferences, market events are suppressed."""
    engine = policy_fixture["engine"]
    repo = policy_fixture["repo"]

    # Configure preferences to exclude market_intelligence
    pref = NotificationPreference(
        farmer_id="fmr_policy_01",
        channels_enabled=[NotificationChannel.IN_APP, NotificationChannel.SMS],
        categories_enabled=[NotificationCategory.WEATHER_ALERT, NotificationCategory.CROP_REMINDER],
    )
    repo.save_notification_preference(pref)

    event = NotificationEvent(
        event_id="evt_market_optout",
        event_type="market_price_surge",
        category=NotificationCategory.MARKET_INTELLIGENCE,
        priority=NotificationPriority.MEDIUM,
        farmer_id="fmr_policy_01",
        payload={"crop_name": "Paddy"},
    )

    day_time = datetime(2026, 11, 9, 14, 0)
    eval_result = engine.evaluate(event, now=day_time)
    assert eval_result.allowed is False
    assert "opted out by farmer preference" in eval_result.suppression_reason
