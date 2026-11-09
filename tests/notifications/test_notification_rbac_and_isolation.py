"""Security, Object-Level Authorization, and Multi-Tenant Isolation tests for Phase 9 APIs."""

import os
import tempfile
import uuid
import pytest
from fastapi import HTTPException

from backend.app.main import (
    complete_farm_reminder,
    dismiss_notification,
    get_admin_outbox_queue,
    get_notification_channels_telemetry,
    get_notification_preferences,
    get_unread_notification_count,
    list_farm_reminders,
    list_farmer_notifications,
    mark_all_notifications_read,
    mark_notification_read,
    process_outbox_batch,
    require_roles,
    retry_failed_outbox_item,
    snooze_farm_reminder,
    system,
    update_channel_consent,
    update_notification_preferences,
)
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord, UserRole
from backend.app.schemas.notification import (
    ConsentUpdateRequest,
    FarmReminderCreate,
    FarmReminderRecord,
    FarmReminderSnooze,
    NotificationCategory,
    NotificationChannel,
    NotificationPreference,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    ReminderActivityType,
)


@pytest.fixture
def rbac_isolation_fixture():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    old_repo = system.farm_context.repo
    system.farm_context.repo = repo
    system.notification_service.repo = repo
    system.notification_service.outbox_service.repo = repo
    system.notification_service.event_bus.repo = repo
    system.notification_service.reminder_service.repo = repo

    rand_sfx = uuid.uuid4().hex[:6]

    # 1. Admin
    u_admin = UserRecord(
        user_id=f"usr_admin_{rand_sfx}",
        username=f"admin_{rand_sfx}",
        password_hash="hash",
        salt="salt",
        role=UserRole.ADMINISTRATOR,
        created_at="2026-11-09T00:00:00",
    )
    repo.create_user(u_admin)

    # 2. Farmer A
    u_farmer_a = UserRecord(
        user_id=f"usr_fa_{rand_sfx}",
        username=f"farmer_a_{rand_sfx}",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-09T00:00:00",
    )
    repo.create_user(u_farmer_a)
    prof_a = FarmerProfile(
        farmer_id=f"fmr_a_{rand_sfx}",
        user_id=u_farmer_a.user_id,
        full_name="Farmer Alpha",
        phone_number="9876543210",
        preferred_language="hi-IN",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farmer_profile(prof_a)

    # 3. Farmer B
    u_farmer_b = UserRecord(
        user_id=f"usr_fb_{rand_sfx}",
        username=f"farmer_b_{rand_sfx}",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-09T00:00:00",
    )
    repo.create_user(u_farmer_b)
    prof_b = FarmerProfile(
        farmer_id=f"fmr_b_{rand_sfx}",
        user_id=u_farmer_b.user_id,
        full_name="Farmer Beta",
        phone_number="9876543211",
        preferred_language="en-IN",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farmer_profile(prof_b)

    # Seed notification and reminder for Farmer B
    notif_b = NotificationRecord(
        notification_id=f"notif_b_{rand_sfx}",
        farmer_id=prof_b.farmer_id,
        category=NotificationCategory.MARKET_INTELLIGENCE,
        priority=NotificationPriority.MEDIUM,
        title_en="Price Alert for Farmer B",
        title_hi="किसान बी के लिए भाव सूचना",
        body_en="Confidential price insight.",
        body_hi="गोपनीय भाव जानकारी।",
        created_at="2026-11-09T10:00:00",
    )
    repo.create_notification(notif_b)

    rem_b = FarmReminderRecord(
        reminder_id=f"rem_b_{rand_sfx}",
        farmer_id=prof_b.farmer_id,
        activity_type=ReminderActivityType.IRRIGATION,
        title_en="Farmer B Irrigation",
        title_hi="सिंचाई",
        scheduled_date="2026-11-10",
        created_at="2026-11-09T10:00:00",
        updated_at="2026-11-09T10:00:00",
    )
    repo.create_farm_reminder(rem_b)

    yield {
        "repo": repo,
        "admin": u_admin,
        "farmer_a": u_farmer_a,
        "prof_a": prof_a,
        "farmer_b": u_farmer_b,
        "prof_b": prof_b,
        "notif_b": notif_b,
        "rem_b": rem_b,
    }

    system.farm_context.repo = old_repo
    try:
        os.remove(path)
    except Exception:
        pass


def test_farmer_a_cannot_view_farmer_b_notifications(rbac_isolation_fixture):
    """Verify listing notifications strictly scopes to authenticated farmer."""
    farmer_a = rbac_isolation_fixture["farmer_a"]
    notifs_a = list_farmer_notifications(user=farmer_a)
    assert len(notifs_a) == 0  # Farmer A has no notifications yet


def test_farmer_a_cannot_mark_or_dismiss_farmer_b_notification(rbac_isolation_fixture):
    """Verify Farmer A cannot alter Farmer B's notification read/dismiss status (403 Forbidden)."""
    farmer_a = rbac_isolation_fixture["farmer_a"]
    notif_b = rbac_isolation_fixture["notif_b"]

    # Attempt to mark read
    with pytest.raises(HTTPException) as exc1:
        mark_notification_read(notification_id=notif_b.notification_id, user=farmer_a)
    assert exc1.value.status_code == 403
    assert "Cannot access another farmer's notification" in exc1.value.detail

    # Attempt to dismiss
    with pytest.raises(HTTPException) as exc2:
        dismiss_notification(notification_id=notif_b.notification_id, user=farmer_a)
    assert exc2.value.status_code == 403
    assert "Cannot access another farmer's notification" in exc2.value.detail


def test_farmer_a_cannot_snooze_or_complete_farmer_b_reminder(rbac_isolation_fixture):
    """Verify Farmer A cannot modify Farmer B's farm reminders (403 Forbidden)."""
    farmer_a = rbac_isolation_fixture["farmer_a"]
    rem_b = rbac_isolation_fixture["rem_b"]

    # Attempt complete
    with pytest.raises(HTTPException) as exc1:
        complete_farm_reminder(reminder_id=rem_b.reminder_id, user=farmer_a)
    assert exc1.value.status_code == 403
    assert "Cannot modify another farmer's reminder" in exc1.value.detail

    # Attempt snooze
    with pytest.raises(HTTPException) as exc2:
        snooze_farm_reminder(reminder_id=rem_b.reminder_id, req=FarmReminderSnooze(snooze_days=2), user=farmer_a)
    assert exc2.value.status_code == 403
    assert "Cannot modify another farmer's reminder" in exc2.value.detail


def test_farmer_cannot_access_admin_outbox_queue(rbac_isolation_fixture):
    """Verify non-admin roles cannot view outbox or trigger delivery cycles (403 Forbidden)."""
    farmer_a = rbac_isolation_fixture["farmer_a"]

    check_admin = require_roles([UserRole.ADMINISTRATOR])
    with pytest.raises(HTTPException) as exc:
        check_admin(user=farmer_a)
    assert exc.value.status_code == 403


def test_admin_outbox_process_and_telemetry(rbac_isolation_fixture):
    """Verify administrator successfully accesses outbox queue, processes batch, and views telemetry."""
    admin = rbac_isolation_fixture["admin"]

    # 1. Admin views queue
    queue_data = get_admin_outbox_queue(admin=admin)
    assert "metrics" in queue_data
    assert "items" in queue_data

    # 2. Admin processes batch
    batch_res = process_outbox_batch(batch_size=10, admin=admin)
    assert batch_res.processed_count == 0

    # 3. Admin views genuine channel telemetry
    telemetry = get_notification_channels_telemetry(admin=admin)
    assert len(telemetry) == 5  # IN_APP, SMS, WHATSAPP, EMAIL, VOICE_CALL
    in_app_status = next(t for t in telemetry if t.channel == NotificationChannel.IN_APP)
    assert in_app_status.configured is True
