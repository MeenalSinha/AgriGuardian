"""Tests for Phase 10 Incremental Push Mutations, Idempotency, and Concurrency."""

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
from backend.app.schemas.sync import (
    DeviceRegisterRequest,
    DeviceType,
    SyncMutationItem,
    SyncPushRequest,
)
from backend.app.services.sync.sync_conflict_service import SyncConflictService
from backend.app.services.sync.sync_device_service import SyncDeviceService
from backend.app.services.sync.sync_push_service import SyncPushService


@pytest.fixture
def push_env():
    """Create test repo and push service with registered farmer & device."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_farmer_kailash",
        username="kailash",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_kailash",
        user_id="usr_farmer_kailash",
        full_name="Kailash Chandra",
        phone_number="+919812345678",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_kailash_01",
        farmer_id="fmr_kailash",
        farm_name="Kailash Krishi Farm",
        state="Madhya Pradesh",
        district="Sehore",
        total_area=4.0,
        area_unit=AreaUnit.ACRE,
        total_area_acres=4.0,
        ownership_type=OwnershipType.OWNED,
        is_active_farm=True,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    dev_svc = SyncDeviceService(repo)
    dev_svc.register_device(
        user_id="usr_farmer_kailash",
        farmer_id="fmr_kailash",
        req=DeviceRegisterRequest(
            device_id="dev_kailash_pwa",
            device_name="Kailash Phone",
            device_type=DeviceType.PWA_MOBILE,
            app_version="1.10.0",
        ),
    )

    conflict_svc = SyncConflictService(repo)
    push_svc = SyncPushService(repo=repo, device_service=dev_svc, conflict_service=conflict_svc)
    yield repo, push_svc
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass


def test_batch_push_mutations(push_env):
    repo, push_svc = push_env

    # 1. Prepare batch: CREATE activity and CREATE reminder
    mut_act = SyncMutationItem(
        mutation_id="mut_create_act_001",
        resource_type="farm_activity",
        operation="CREATE",
        client_record_id="client_act_99",
        payload={
            "farm_id": "farm_kailash_01",
            "activity_type": "irrigation",
            "activity_title": "Drip Irrigation Cycle",
            "scheduled_date": "2026-11-10",
            "status": "completed",
            "completed_date": "2026-11-10",
            "cost_incurred": 150.0,
            "notes": "Optimal moisture maintained",
        },
    )

    mut_rem = SyncMutationItem(
        mutation_id="mut_create_rem_002",
        resource_type="farm_reminder",
        operation="CREATE",
        client_record_id="client_rem_88",
        payload={
            "farm_id": "farm_kailash_01",
            "activity_type": "pest_spray",
            "title_en": "Spray Neem Oil",
            "title_hi": "नीम के तेल का छिड़काव",
            "scheduled_date": "2026-11-12",
            "due_time": "09:00",
        },
    )

    req = SyncPushRequest(
        device_id="dev_kailash_pwa",
        mutations=[mut_act, mut_rem],
    )

    resp = push_svc.process_push_mutations(
        user_id="usr_farmer_kailash",
        farmer_id="fmr_kailash",
        req=req,
    )

    assert resp.processed_count == 2
    assert resp.applied_count == 2
    assert resp.conflict_count == 0
    assert len(resp.results) == 2

    # Verify records created in DB
    activities = repo.get_activities_by_farmer("fmr_kailash")
    assert any(a.activity_title == "Drip Irrigation Cycle" for a in activities)

    reminders = repo.list_farm_reminders("fmr_kailash")
    assert any("Spray Neem Oil" in r.title_en for r in reminders)


def test_mutation_idempotency_replay(push_env):
    repo, push_svc = push_env

    mut = SyncMutationItem(
        mutation_id="mut_idempotent_test_101",
        resource_type="farm_activity",
        operation="CREATE",
        client_record_id="client_act_idem",
        payload={
            "farm_id": "farm_kailash_01",
            "activity_type": "weeding",
            "activity_title": "Manual Inter-row Weeding",
            "scheduled_date": "2026-11-15",
            "status": "planned",
        },
    )

    req = SyncPushRequest(device_id="dev_kailash_pwa", mutations=[mut])

    # First push
    resp1 = push_svc.process_push_mutations(
        user_id="usr_farmer_kailash",
        farmer_id="fmr_kailash",
        req=req,
    )
    assert resp1.applied_count == 1
    server_id_1 = resp1.results[0].server_record_id

    # Replay identical push (simulating network disconnection before client received response)
    resp2 = push_svc.process_push_mutations(
        user_id="usr_farmer_kailash",
        farmer_id="fmr_kailash",
        req=req,
    )
    assert resp2.processed_count == 1
    assert resp2.applied_count == 1
    assert resp2.results[0].server_record_id == server_id_1
    assert resp2.results[0].status == "applied"

    # Verify no duplicate activity created
    activities = repo.get_activities_by_farmer("fmr_kailash")
    matching = [a for a in activities if a.activity_title == "Manual Inter-row Weeding"]
    assert len(matching) == 1


def test_partial_batch_fault_tolerance(push_env):
    repo, push_svc = push_env

    # Valid mutation
    valid_mut = SyncMutationItem(
        mutation_id="mut_valid_item_01",
        resource_type="farm_activity",
        operation="CREATE",
        client_record_id="c_valid",
        payload={
            "farm_id": "farm_kailash_01",
            "activity_type": "scouting",
            "activity_title": "Field Inspection",
            "scheduled_date": "2026-11-10",
        },
    )

    # Invalid mutation (unsupported resource type)
    invalid_mut = SyncMutationItem(
        mutation_id="mut_invalid_item_02",
        resource_type="unsupported_entity_type",
        operation="CREATE",
        client_record_id="c_invalid",
        payload={"foo": "bar"},
    )

    req = SyncPushRequest(device_id="dev_kailash_pwa", mutations=[valid_mut, invalid_mut])

    resp = push_svc.process_push_mutations(
        user_id="usr_farmer_kailash",
        farmer_id="fmr_kailash",
        req=req,
    )

    assert resp.processed_count == 2
    assert resp.applied_count == 1
    assert resp.results[0].status == "applied"
    assert resp.results[1].status == "rejected"
    assert "Unsupported resource type" in resp.results[1].error_message
