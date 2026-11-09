"""Tests for Phase 11 Offline Economics Sync, Idempotency, and Optimistic Locking."""

import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    CropCycle,
    CropCycleStatus,
    CropGrowthStage,
    CropSeason,
    Farm,
    FarmerProfile,
    LiteracyPreference,
    OwnershipType,
    Plot,
    UserRecord,
    UserRole,
)
from backend.app.schemas.sync import (
    DeviceRegisterRequest,
    DeviceType,
    SyncMutationItem,
    SyncPushRequest,
)
from backend.app.services.sync.sync_device_service import SyncDeviceService
from backend.app.services.sync.sync_push_service import SyncPushService
from backend.app.services.sync.sync_service import SyncService


@pytest.fixture
def sync_env():
    """Create test repo and services with registered device."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_jagdish",
        username="jagdish",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_jagdish",
        user_id="usr_jagdish",
        full_name="Jagdish Yadav",
        phone_number="9876543260",
        state="Bihar",
        district="Samastipur",
        sub_district="Samastipur",
        village="Pusa",
        primary_language="hi",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_jagdish_1",
        farmer_id="fmr_jagdish",
        farm_name="Pusa Krishi Farm",
        total_area=3.0,
        total_area_acres=3.0,
        ownership_type=OwnershipType.OWNED,
        state="Bihar",
        district="Samastipur",
        village="Pusa",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_jagdish_1",
        farm_id="farm_jagdish_1",
        farmer_id="fmr_jagdish",
        plot_name="Maize Field",
        area=3.0,
        area_acres=3.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    cycle = CropCycle(
        cycle_id="cycle_maize_2026",
        farmer_id="fmr_jagdish",
        farm_id="farm_jagdish_1",
        plot_id="plot_jagdish_1",
        crop_name="Maize",
        variety_name="Shaktiman-1",
        season=CropSeason.KHARIF,
        sowing_date="2026-06-10",
        current_stage=CropGrowthStage.HARVEST_READY,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle)

    dev_srv = SyncDeviceService(repo)
    device = dev_srv.register_device(
        user_id="usr_jagdish",
        farmer_id="fmr_jagdish",
        req=DeviceRegisterRequest(
            device_id="dev_redmi_offline_1",
            device_name="Redmi 9A Android",
            device_type=DeviceType.PWA_MOBILE,
            app_version="11.0.0",
        ),
    )

    push_srv = SyncPushService(repo, device_service=dev_srv)
    full_sync = SyncService(repo)

    return {
        "repo": repo,
        "push_service": push_srv,
        "sync_service": full_sync,
        "farmer_id": "fmr_jagdish",
        "farm_id": "farm_jagdish_1",
        "cycle_id": "cycle_maize_2026",
        "device_id": device.device_id,
        "user_id": "usr_jagdish",
    }


def test_push_expense_mutation_and_idempotency(sync_env):
    """Test offline creation of an expense mutation and verify replay idempotency."""
    srv = sync_env["push_service"]
    farmer_id = sync_env["farmer_id"]
    user_id = sync_env["user_id"]
    dev_id = sync_env["device_id"]
    farm_id = sync_env["farm_id"]
    cycle_id = sync_env["cycle_id"]

    mutation = SyncMutationItem(
        mutation_id="mut_exp_001",
        client_timestamp="2026-11-10T08:00:00Z",
        operation="CREATE",
        resource_type="farm_expense",
        client_record_id="client_exp_local_1",
        base_version=0,
        payload={
            "farm_id": farm_id,
            "crop_cycle_id": cycle_id,
            "expense_category": "seeds_planting",
            "expense_description": "Maize Hybrid Seed",
            "quantity": 10.0,
            "quantity_unit": "kg",
            "unit_price": 250.0,
            "total_amount": 2500.0,
            "purchase_date": "2026-06-08",
        },
    )

    req = SyncPushRequest(device_id=dev_id, mutations=[mutation])
    resp = srv.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=req)

    assert resp.applied_count == 1
    assert resp.conflict_count == 0
    item = resp.results[0]
    assert item.status == "applied"
    assert item.server_record_id.startswith("exp_")

    # Idempotent replay of same mutation
    resp_replay = srv.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=req)
    assert resp_replay.applied_count == 1
    assert resp_replay.results[0].status == "applied"
    assert resp_replay.results[0].server_record_id == item.server_record_id


def test_push_harvest_and_sale_mutations(sync_env):
    """Test pushing offline harvest and sales mutations in sequence."""
    srv = sync_env["push_service"]
    farmer_id = sync_env["farmer_id"]
    user_id = sync_env["user_id"]
    dev_id = sync_env["device_id"]
    farm_id = sync_env["farm_id"]
    cycle_id = sync_env["cycle_id"]

    mut_harvest = SyncMutationItem(
        mutation_id="mut_hrv_001",
        client_timestamp="2026-11-10T09:00:00Z",
        operation="CREATE",
        resource_type="farm_harvest",
        client_record_id="client_hrv_local_1",
        base_version=0,
        payload={
            "farm_id": farm_id,
            "crop_cycle_id": cycle_id,
            "harvest_date": "2026-10-15",
            "crop_name": "Maize",
            "harvested_quantity": 45.0,
            "quantity_unit": "quintal",
            "stored_quantity": 45.0,
        },
    )

    mut_sale = SyncMutationItem(
        mutation_id="mut_sale_001",
        client_timestamp="2026-11-10T10:00:00Z",
        operation="CREATE",
        resource_type="farm_sale",
        client_record_id="client_sale_local_1",
        base_version=0,
        payload={
            "farm_id": farm_id,
            "crop_cycle_id": cycle_id,
            "crop_name": "Maize",
            "quantity_sold": 30.0,
            "realized_price_per_unit": 2050.0,
            "gross_amount": 61500.0,
            "transport_charges": 600.0,
            "net_amount_received": 60900.0,
            "sale_date": "2026-10-18",
            "buyer_name": "Samastipur APMC Trader",
        },
    )

    req = SyncPushRequest(device_id=dev_id, mutations=[mut_harvest, mut_sale])
    resp = srv.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=req)

    assert resp.applied_count == 2
    assert resp.results[0].server_record_id.startswith("hrv_")
    assert resp.results[1].server_record_id.startswith("sale_")


def test_optimistic_concurrency_conflict_on_expense(sync_env):
    """Test that editing an expense with stale base_version triggers conflict."""
    srv = sync_env["push_service"]
    repo = sync_env["repo"]
    farmer_id = sync_env["farmer_id"]
    user_id = sync_env["user_id"]
    dev_id = sync_env["device_id"]
    farm_id = sync_env["farm_id"]

    # 1. Create initial expense (version = 1)
    mut_create = SyncMutationItem(
        mutation_id="mut_exp_conflict_init",
        client_timestamp="2026-11-10T08:00:00Z",
        operation="CREATE",
        resource_type="farm_expense",
        client_record_id="c_exp_conf_1",
        base_version=0,
        payload={
            "farm_id": farm_id,
            "expense_category": "fertilizer_soil",
            "expense_description": "Urea",
            "total_amount": 1000.0,
            "purchase_date": "2026-07-01",
        },
    )
    resp = srv.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=SyncPushRequest(device_id=dev_id, mutations=[mut_create]))
    server_id = resp.results[0].server_record_id

    # 2. Server advances version to 2 (e.g. from web UI)
    repo.update_farm_expense(server_id, farmer_id, {"expense_description": "Urea (Updated from Web)"})

    # 3. Offline client attempts UPDATE using stale base_version = 1
    mut_update = SyncMutationItem(
        mutation_id="mut_exp_conflict_attempt",
        client_timestamp="2026-11-10T09:00:00Z",
        operation="UPDATE",
        resource_type="farm_expense",
        client_record_id="c_exp_conf_1",
        base_version=1,  # Stale! Server is now at 2
        payload={
            "expense_id": server_id,
            "expense_description": "Urea (Client Offline Edit)",
            "total_amount": 1050.0,
        },
    )
    resp_conf = srv.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=SyncPushRequest(device_id=dev_id, mutations=[mut_update]))
    assert resp_conf.conflict_count == 1
    assert resp_conf.results[0].status == "conflict"
    assert resp_conf.results[0].conflict_details is not None


def test_sync_bootstrap_includes_economics(sync_env):
    """Test that SyncService.get_bootstrap packages expenses, harvests, sales, and budgets."""
    full_sync = sync_env["sync_service"]
    farmer_id = sync_env["farmer_id"]
    user_id = sync_env["user_id"]
    dev_id = sync_env["device_id"]

    bootstrap = full_sync.get_bootstrap(user_id=user_id, farmer_id=farmer_id, device_id=dev_id)
    assert hasattr(bootstrap, "expenses")
    assert hasattr(bootstrap, "harvests")
    assert hasattr(bootstrap, "sales")
    assert hasattr(bootstrap, "budgets")
