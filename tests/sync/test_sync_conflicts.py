"""Tests for Phase 10 Intelligent Conflict Detection and Resolution."""

import tempfile
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord, UserRole
from backend.app.schemas.notification import (
    FarmReminderRecord,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.schemas.sync import (
    ConflictResolutionStrategy,
    ConflictResolveRequest,
    DeviceRegisterRequest,
    DeviceType,
    SyncMutationItem,
    SyncPushRequest,
)
from backend.app.services.sync.sync_conflict_service import SyncConflictService
from backend.app.services.sync.sync_device_service import SyncDeviceService
from backend.app.services.sync.sync_push_service import SyncPushService


@pytest.fixture
def conflict_env():
    """Create test repo, push service, and conflict service."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user1 = UserRecord(
        user_id="usr_conf_01",
        username="farmer_one",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    user2 = UserRecord(
        user_id="usr_conf_02",
        username="farmer_two",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user1)
    repo.create_user(user2)

    profile1 = FarmerProfile(
        farmer_id="fmr_conf_01",
        user_id="usr_conf_01",
        full_name="Farmer One",
        phone_number="+919999900001",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile1)

    profile2 = FarmerProfile(
        farmer_id="fmr_conf_02",
        user_id="usr_conf_02",
        full_name="Farmer Two",
        phone_number="+919999900002",
        state="Madhya Pradesh",
        district="Bhopal",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile2)

    dev_svc = SyncDeviceService(repo)
    dev_svc.register_device(
        user_id="usr_conf_01",
        farmer_id="fmr_conf_01",
        req=DeviceRegisterRequest(
            device_id="dev_conf_01",
            device_name="Phone 1",
            device_type=DeviceType.PWA_MOBILE,
            app_version="1.10.0",
        ),
    )

    conflict_svc = SyncConflictService(repo)
    push_svc = SyncPushService(repo=repo, device_service=dev_svc, conflict_service=conflict_svc)
    yield repo, push_svc, conflict_svc
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass


def test_version_conflict_detection_on_reminder(conflict_env):
    repo, push_svc, conflict_svc = conflict_env

    # 1. Create reminder on server (initial version = 1)
    rem = FarmReminderRecord(
        reminder_id="rem_conflict_test",
        farmer_id="fmr_conf_01",
        activity_type=ReminderActivityType.PEST_SPRAY,
        title_en="Spray Chlorpyrifos",
        title_hi="कीटनाशक छिड़काव",
        scheduled_date="2026-11-12",
        status=ReminderStatus.SCHEDULED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm_reminder(rem)

    # 2. Server updates reminder online (bumps version_number to 2)
    repo.update_farm_reminder(
        reminder_id="rem_conflict_test",
        farmer_id="fmr_conf_01",
        update_data={"scheduled_date": "2026-11-14", "title_en": "Spray Chlorpyrifos (Delayed by Agronomist)"},
    )

    # 3. Client device pushes an offline edit based on base_version = 1 (stale edit)
    stale_mut = SyncMutationItem(
        mutation_id="mut_stale_reminder_edit",
        resource_type="farm_reminder",
        operation="UPDATE",
        client_record_id="rem_conflict_test",
        payload={
            "reminder_id": "rem_conflict_test",
            "scheduled_date": "2026-11-11",
            "title_en": "Spray Chlorpyrifos (Early Client Edit)",
        },
        base_version=1,
    )

    req = SyncPushRequest(device_id="dev_conf_01", mutations=[stale_mut])
    resp = push_svc.process_push_mutations(
        user_id="usr_conf_01",
        farmer_id="fmr_conf_01",
        req=req,
    )

    assert resp.conflict_count == 1
    assert resp.results[0].status == "conflict"
    assert resp.results[0].conflict_details is not None

    # Verify conflict persisted in sync_conflicts
    conflicts = conflict_svc.list_conflicts("fmr_conf_01")
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.resource_type == "farm_reminder"
    assert c.status == "pending"


def test_conflict_resolution_strategies(conflict_env):
    repo, push_svc, conflict_svc = conflict_env

    # Seed conflict
    conflict_rec = conflict_svc.record_conflict(
        mutation_id="mut_test_res",
        device_id="dev_conf_01",
        farmer_id="fmr_conf_01",
        resource_type="farm_reminder",
        client_record_id="rem_res_01",
        server_record_id="rem_res_01",
        client_payload={"title_en": "Client Preferred Title", "scheduled_date": "2026-11-15"},
        server_payload={"title_en": "Server Original Title", "scheduled_date": "2026-11-18"},
    )
    conflict_id = conflict_rec["conflict_id"]

    # Also seed underlying reminder
    rem = FarmReminderRecord(
        reminder_id="rem_res_01",
        farmer_id="fmr_conf_01",
        activity_type=ReminderActivityType.CUSTOM,
        title_en="Server Original Title",
        title_hi="मूल शीर्षक",
        scheduled_date="2026-11-18",
        status=ReminderStatus.SCHEDULED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm_reminder(rem)

    # Test CLIENT_WINS
    res = conflict_svc.resolve_conflict(
        conflict_id=conflict_id,
        farmer_id="fmr_conf_01",
        req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.CLIENT_WINS),
    )
    assert res.status == "resolved"
    assert res.resolution_strategy == "client_wins"

    # Verify underlying reminder updated to client payload
    updated_rem = repo.get_farm_reminder("rem_res_01")
    assert updated_rem.title_en == "Client Preferred Title"
    assert updated_rem.scheduled_date == "2026-11-15"


def test_cross_tenant_conflict_isolation(conflict_env):
    repo, push_svc, conflict_svc = conflict_env

    conflict_rec = conflict_svc.record_conflict(
        mutation_id="mut_iso_01",
        device_id="dev_conf_01",
        farmer_id="fmr_conf_01",
        resource_type="farm_reminder",
        client_record_id="rem_iso",
        server_record_id="rem_iso",
        client_payload={"foo": "bar"},
        server_payload={"foo": "baz"},
    )
    conflict_id = conflict_rec["conflict_id"]

    # Farmer 2 cannot view or resolve Farmer 1's conflict
    with pytest.raises(HTTPException) as exc:
        conflict_svc.get_conflict(conflict_id=conflict_id, farmer_id="fmr_conf_02")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException):
        conflict_svc.resolve_conflict(
            conflict_id=conflict_id,
            farmer_id="fmr_conf_02",
            req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.CLIENT_WINS),
        )
