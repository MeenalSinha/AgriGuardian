"""Tests for Phase 10 Low-Bandwidth Optimizations, Pagination, and Telemetry."""

import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    Farm,
    FarmActivity,
    FarmerProfile,
    LiteracyPreference,
    OwnershipType,
    UserRecord,
    UserRole,
)
from backend.app.schemas.sync import DeviceRegisterRequest, DeviceType, SyncPullRequest
from backend.app.services.sync.sync_service import SyncService


@pytest.fixture
def telemetry_env():
    """Create test repo and sync service."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_tel_01",
        username="tel_farmer",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_tel_01",
        user_id="usr_tel_01",
        full_name="Telemetry Farmer",
        phone_number="+919988776655",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_tel_01",
        farmer_id="fmr_tel_01",
        farm_name="Tel Farm",
        state="Madhya Pradesh",
        district="Sehore",
        total_area=2.0,
        area_unit=AreaUnit.ACRE,
        total_area_acres=2.0,
        ownership_type=OwnershipType.OWNED,
        is_active_farm=True,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    sync_service = SyncService(repo=repo)
    sync_service.register_device(
        user_id="usr_tel_01",
        farmer_id="fmr_tel_01",
        req=DeviceRegisterRequest(
            device_id="dev_tel_01",
            device_name="Tel Phone",
            device_type=DeviceType.PWA_MOBILE,
            app_version="1.10.0",
        ),
    )

    yield repo, sync_service
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass


def test_sync_telemetry_recording(telemetry_env):
    repo, sync_svc = telemetry_env

    # Execute bootstrap
    boot = sync_svc.get_bootstrap(
        user_id="usr_tel_01",
        farmer_id="fmr_tel_01",
        device_id="dev_tel_01",
    )
    assert boot.farmer_profile is not None

    # Verify telemetry row created in sync_telemetry
    with repo.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM sync_telemetry WHERE device_id = 'dev_tel_01'")
        rows = cursor.fetchall()
        assert len(rows) >= 1
        assert rows[0]["sync_type"] == "bootstrap"
        assert rows[0]["status"] == "success"


def test_pull_pagination_limits(telemetry_env):
    repo, sync_svc = telemetry_env

    # Seed 15 activities in DB, each producing a sync_change_log record
    for i in range(15):
        act = FarmActivity(
            activity_id=f"act_page_{i:02d}",
            farm_id="farm_tel_01",
            farmer_id="fmr_tel_01",
            activity_type=ActivityType.WEEDING,
            activity_title=f"Page Activity {i}",
            scheduled_date="2026-11-10",
            status=ActivityStatus.PLANNED,
            created_at="2026-11-10T00:00:00Z",
            updated_at="2026-11-10T00:00:00Z",
        )
        repo.create_farm_activity(act)

    # Request with page limit of 5
    req1 = SyncPullRequest(
        device_id="dev_tel_01",
        since_sequence_id=0,
        limit=5,
    )
    res1 = sync_svc.pull_changes(
        user_id="usr_tel_01",
        farmer_id="fmr_tel_01",
        req=req1,
    )
    assert len(res1.changes) == 5
    assert res1.has_more is True

    # Request next page
    req2 = SyncPullRequest(
        device_id="dev_tel_01",
        since_sequence_id=res1.new_sequence_id,
        limit=5,
    )
    res2 = sync_svc.pull_changes(
        user_id="usr_tel_01",
        farmer_id="fmr_tel_01",
        req=req2,
    )
    assert len(res2.changes) == 5
    assert res2.new_sequence_id > res1.new_sequence_id


def test_sync_status_reachability(telemetry_env):
    repo, sync_svc = telemetry_env

    status = sync_svc.get_sync_status()
    assert status["status"] == "healthy"
    assert status["service"] == "agri_guardian_sync"
    assert status["protocol_version"] == "1.10.0"
    assert "server_timestamp" in status
