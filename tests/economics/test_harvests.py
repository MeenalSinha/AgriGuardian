"""Unit and integration tests for Phase 11 Crop Harvest & Production Service."""

import tempfile
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import FarmHarvestCreate, FarmHarvestUpdate
from backend.app.schemas.farm import (
    CropCycle,
    CropCycleCreate,
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
from backend.app.services.economics.harvest_service import HarvestService


@pytest.fixture
def harvest_env():
    """Create test repository with farmer, farm, plot, and crop cycle."""
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
        full_name="Ramesh Kumar",
        phone_number="9876543220",
        state="Haryana",
        district="Karnal",
        sub_district="Karnal",
        village="Taraori",
        primary_language="hi",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_ramesh_1",
        farmer_id="fmr_ramesh",
        farm_name="Taraori Basmati Farm",
        total_area=4.0,
        total_area_acres=4.0,
        ownership_type=OwnershipType.OWNED,
        state="Haryana",
        district="Karnal",
        village="Taraori",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_ramesh_1",
        farm_id="farm_ramesh_1",
        farmer_id="fmr_ramesh",
        plot_name="North Field",
        area=4.0,
        area_acres=4.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    cycle = CropCycle(
        cycle_id="cycle_paddy_2026",
        farmer_id="fmr_ramesh",
        farm_id="farm_ramesh_1",
        plot_id="plot_ramesh_1",
        crop_name="Paddy",
        variety_name="Pusa Basmati 1121",
        season=CropSeason.KHARIF,
        sowing_date="2026-06-15",
        current_stage=CropGrowthStage.HARVEST_READY,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle)

    service = HarvestService(repo)
    return {"repo": repo, "service": service, "farmer": profile, "farm": farm, "cycle": cycle}


def test_record_harvest_with_reconciliation_success(harvest_env):
    """Test harvest event with reconciled storage, sales, and damaged quantities."""
    service = harvest_env["service"]
    farmer_id = harvest_env["farmer"].farmer_id
    farm_id = harvest_env["farm"].farm_id
    cycle_id = harvest_env["cycle"].cycle_id

    # Harvested 40 quintals: 25 sold directly, 12 stored in warehouse, 2 retained for seed, 1 damaged = 40
    payload = FarmHarvestCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        harvest_date="2026-10-25",
        crop_name="Paddy",
        harvested_quantity=40.0,
        quantity_unit="quintal",
        quality_grade="Grade A",
        sold_quantity=25.0,
        stored_quantity=12.0,
        retained_quantity=2.0,
        damaged_quantity=1.0,
        rejected_quantity=0.0,
        moisture_percentage=14.5,
        notes="Excellent grain quality, optimal moisture level.",
    )

    h_rec = service.record_harvest(farmer_id=farmer_id, payload=payload, user_id="usr_ramesh")
    assert h_rec.harvest_id.startswith("hrv_")
    assert h_rec.harvested_quantity == 40.0
    assert h_rec.stored_quantity == 12.0
    assert h_rec.sold_quantity == 25.0
    assert h_rec.version_number == 1


def test_record_harvest_over_allocation_fails(harvest_env):
    """Test that allocating more than harvested quantity raises 400 Bad Request."""
    service = harvest_env["service"]
    farmer_id = harvest_env["farmer"].farmer_id
    farm_id = harvest_env["farm"].farm_id
    cycle_id = harvest_env["cycle"].cycle_id

    # Harvested 20 quintals, but claiming 25 sold + 10 stored = 35 quintals!
    payload = FarmHarvestCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        harvest_date="2026-10-25",
        crop_name="Paddy",
        harvested_quantity=20.0,
        sold_quantity=25.0,
        stored_quantity=10.0,
    )

    with pytest.raises(HTTPException) as exc_info:
        service.record_harvest(farmer_id=farmer_id, payload=payload, user_id="usr_ramesh")
    assert exc_info.value.status_code == 400
    assert "exceeds total harvested quantity" in exc_info.value.detail


def test_multi_harvest_events_for_cycle(harvest_env):
    """Test recording multiple harvest pickings for the same crop cycle."""
    service = harvest_env["service"]
    farmer_id = harvest_env["farmer"].farmer_id
    farm_id = harvest_env["farm"].farm_id
    cycle_id = harvest_env["cycle"].cycle_id

    # Picking 1: 18 quintals
    p1 = service.record_harvest(
        farmer_id=farmer_id,
        payload=FarmHarvestCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            harvest_date="2026-10-20",
            crop_name="Paddy",
            harvested_quantity=18.0,
            stored_quantity=18.0,
        ),
        user_id="usr_ramesh",
    )

    # Picking 2: 22 quintals
    p2 = service.record_harvest(
        farmer_id=farmer_id,
        payload=FarmHarvestCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            harvest_date="2026-10-28",
            crop_name="Paddy",
            harvested_quantity=22.0,
            stored_quantity=22.0,
        ),
        user_id="usr_ramesh",
    )

    records = service.list_harvests(farmer_id=farmer_id, crop_cycle_id=cycle_id)
    assert len(records) == 2
    total_harvested = sum(r.harvested_quantity for r in records)
    assert total_harvested == 40.0


def test_update_harvest_record(harvest_env):
    """Test updating quality grade and stored allocation."""
    service = harvest_env["service"]
    farmer_id = harvest_env["farmer"].farmer_id
    farm_id = harvest_env["farm"].farm_id
    cycle_id = harvest_env["cycle"].cycle_id

    rec = service.record_harvest(
        farmer_id=farmer_id,
        payload=FarmHarvestCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            harvest_date="2026-10-22",
            crop_name="Paddy",
            harvested_quantity=30.0,
            stored_quantity=30.0,
        ),
        user_id="usr_ramesh",
    )

    updated = service.update_harvest(
        harvest_id=rec.harvest_id,
        farmer_id=farmer_id,
        payload=FarmHarvestUpdate(quality_grade="Grade A+", notes="Re-tested at warehouse"),
        user_id="usr_ramesh",
    )
    assert updated.quality_grade == "Grade A+"
    assert updated.notes == "Re-tested at warehouse"
    assert updated.version_number == 2


def test_harvest_tenant_isolation(harvest_env):
    """Test that another farmer cannot view or modify harvest records."""
    service = harvest_env["service"]
    farmer_id = harvest_env["farmer"].farmer_id
    farm_id = harvest_env["farm"].farm_id
    cycle_id = harvest_env["cycle"].cycle_id

    rec = service.record_harvest(
        farmer_id=farmer_id,
        payload=FarmHarvestCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            harvest_date="2026-10-22",
            crop_name="Paddy",
            harvested_quantity=25.0,
            stored_quantity=25.0,
        ),
        user_id="usr_ramesh",
    )

    with pytest.raises(HTTPException) as exc:
        service.get_harvest(rec.harvest_id, farmer_id="fmr_other_farmer")
    assert exc.value.status_code == 404
