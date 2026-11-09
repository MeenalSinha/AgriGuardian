"""Tests for Phase 10 Offline Bootstrap and Incremental Pull Change Feeds."""

import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycle,
    CropCycleStatus,
    CropGrowthStage,
    CropSeason,
    Farm,
    FarmActivity,
    FarmerProfile,
    LiteracyPreference,
    OwnershipType,
    Plot,
    UserRecord,
    UserRole,
)
from backend.app.schemas.notification import (
    FarmReminderRecord,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.schemas.sync import DeviceRegisterRequest, DeviceType, SyncPullRequest
from backend.app.services.sync.sync_device_service import SyncDeviceService
from backend.app.services.sync.sync_pull_service import SyncPullService


@pytest.fixture
def bootstrap_env():
    """Create test repo and pull service with seeded farm records."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_ramesh",
        username="ramesh",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_ramesh",
        user_id="usr_ramesh",
        full_name="Rameshwar Patel",
        phone_number="+919876543210",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_sehore_01",
        farmer_id="fmr_ramesh",
        farm_name="Narmada Valley Farm",
        state="Madhya Pradesh",
        district="Sehore",
        total_area=5.0,
        area_unit=AreaUnit.ACRE,
        total_area_acres=5.0,
        ownership_type=OwnershipType.OWNED,
        is_active_farm=True,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_01",
        farm_id="farm_sehore_01",
        farmer_id="fmr_ramesh",
        plot_name="North Field",
        area=2.5,
        area_unit=AreaUnit.ACRE,
        area_acres=2.5,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    dev_svc = SyncDeviceService(repo)
    dev_svc.register_device(
        user_id="usr_ramesh",
        farmer_id="fmr_ramesh",
        req=DeviceRegisterRequest(
            device_id="dev_pwa_01",
            device_name="Ramesh Chrome PWA",
            device_type=DeviceType.PWA_MOBILE,
            app_version="1.10.0",
        ),
    )

    pull_svc = SyncPullService(repo=repo, device_service=dev_svc)
    yield repo, pull_svc
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass


def test_offline_bootstrap_bundle(bootstrap_env):
    repo, pull_svc = bootstrap_env

    # Seed an activity and a reminder
    act = FarmActivity(
        activity_id="act_boot_01",
        farm_id="farm_sehore_01",
        farmer_id="fmr_ramesh",
        activity_type=ActivityType.FERTILIZER_APPLICATION,
        activity_title="Urea Application",
        scheduled_date="2026-11-12",
        status=ActivityStatus.PLANNED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm_activity(act)

    rem = FarmReminderRecord(
        reminder_id="rem_boot_01",
        farmer_id="fmr_ramesh",
        farm_id="farm_sehore_01",
        activity_type=ReminderActivityType.FERTILIZATION,
        title_en="Urea Top Dressing Due",
        title_hi="यूरिया का छिड़काव",
        scheduled_date="2026-11-12",
        due_time="08:00",
        status=ReminderStatus.SCHEDULED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm_reminder(rem)

    bundle = pull_svc.bootstrap_farmer_data(
        user_id="usr_ramesh",
        farmer_id="fmr_ramesh",
        device_id="dev_pwa_01",
    )

    assert bundle.farmer_profile["full_name"] == "Rameshwar Patel"
    assert len(bundle.farms) == 1
    assert bundle.farms[0]["farm_name"] == "Narmada Valley Farm"
    assert len(bundle.plots) == 1
    assert len(bundle.activities) >= 1
    assert len(bundle.reminders) >= 1
    assert bundle.sequence_cursor >= 0
    assert bundle.server_timestamp is not None


def test_incremental_change_feed_pull(bootstrap_env):
    repo, pull_svc = bootstrap_env

    # Get initial cursor
    cursor_before = repo.get_sync_cursor(device_id="dev_pwa_01")

    # Log a new activity on server
    act = FarmActivity(
        activity_id="act_pull_02",
        farm_id="farm_sehore_01",
        farmer_id="fmr_ramesh",
        activity_type=ActivityType.IRRIGATION,
        activity_title="Evening Tube Well Irrigation",
        scheduled_date="2026-11-11",
        status=ActivityStatus.PLANNED,
        created_at="2026-11-10T01:00:00Z",
        updated_at="2026-11-10T01:00:00Z",
    )
    repo.create_farm_activity(act)

    # Pull changes since initial cursor
    pull_req = SyncPullRequest(
        device_id="dev_pwa_01",
        since_sequence_id=cursor_before,
        limit=50,
    )
    resp = pull_svc.pull_incremental_changes(
        user_id="usr_ramesh",
        farmer_id="fmr_ramesh",
        req=pull_req,
    )

    assert len(resp.changes) >= 1
    matched = [c for c in resp.changes if c.resource_id == "act_pull_02"]
    assert len(matched) == 1
    assert matched[0].operation == "INSERT"
    assert matched[0].resource_type == "farm_activity"
    assert resp.new_sequence_id > cursor_before

    # Subsequent pull with updated cursor yields no new changes
    pull_req2 = SyncPullRequest(
        device_id="dev_pwa_01",
        since_sequence_id=resp.new_sequence_id,
        limit=50,
    )
    resp2 = pull_svc.pull_incremental_changes(
        user_id="usr_ramesh",
        farmer_id="fmr_ramesh",
        req=pull_req2,
    )
    assert len(resp2.changes) == 0
    assert resp2.has_more is False


def test_deletion_tombstone_in_pull(bootstrap_env):
    repo, pull_svc = bootstrap_env

    rem = FarmReminderRecord(
        reminder_id="rem_to_delete",
        farmer_id="fmr_ramesh",
        farm_id="farm_sehore_01",
        activity_type=ReminderActivityType.CUSTOM,
        title_en="Temporary Reminder",
        title_hi="अस्थायी सूचना",
        scheduled_date="2026-11-14",
        status=ReminderStatus.SCHEDULED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm_reminder(rem)

    cursor_before = repo.get_sync_cursor(device_id="dev_pwa_01")

    # Delete reminder
    repo.delete_farm_reminder("rem_to_delete", farmer_id="fmr_ramesh")

    # Pull changes
    resp = pull_svc.pull_incremental_changes(
        user_id="usr_ramesh",
        farmer_id="fmr_ramesh",
        req=SyncPullRequest(device_id="dev_pwa_01", since_sequence_id=cursor_before),
    )

    tombstones = [c for c in resp.changes if c.resource_id == "rem_to_delete" and c.operation == "DELETE"]
    assert len(tombstones) == 1
    assert tombstones[0].resource_type == "farm_reminder"
