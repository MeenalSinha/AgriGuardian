"""Unit and integration tests for Phase 11 Produce Sales & Realized Revenue Service."""

import tempfile
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import FarmHarvestCreate, FarmSaleCreate, FarmSaleUpdate, PaymentStatus
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
from backend.app.services.economics.harvest_service import HarvestService
from backend.app.services.economics.sales_service import SalesService


@pytest.fixture
def sales_env():
    """Create test repository with farmer, farm, plot, crop cycle, and harvest."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_baldev",
        username="baldev",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_baldev",
        user_id="usr_baldev",
        full_name="Baldev Singh",
        phone_number="9876543230",
        state="Punjab",
        district="Ludhiana",
        sub_district="Ludhiana West",
        village="Gill",
        primary_language="pa",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_baldev_1",
        farmer_id="fmr_baldev",
        farm_name="Gill Village Fields",
        total_area=6.0,
        total_area_acres=6.0,
        ownership_type=OwnershipType.OWNED,
        state="Punjab",
        district="Ludhiana",
        village="Gill",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_baldev_1",
        farm_id="farm_baldev_1",
        farmer_id="fmr_baldev",
        plot_name="Main Field",
        area=6.0,
        area_acres=6.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    cycle = CropCycle(
        cycle_id="cycle_wheat_2026",
        farmer_id="fmr_baldev",
        farm_id="farm_baldev_1",
        plot_id="plot_baldev_1",
        crop_name="Wheat",
        variety_name="HD 3086",
        season=CropSeason.RABI,
        sowing_date="2025-11-10",
        current_stage=CropGrowthStage.HARVESTED,
        status=CropCycleStatus.COMPLETED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle)

    harvest_srv = HarvestService(repo)
    h_rec = harvest_srv.record_harvest(
        farmer_id="fmr_baldev",
        payload=FarmHarvestCreate(
            farm_id="farm_baldev_1",
            crop_cycle_id="cycle_wheat_2026",
            harvest_date="2026-04-15",
            crop_name="Wheat",
            harvested_quantity=120.0,
            stored_quantity=120.0,
        ),
        user_id="usr_baldev",
    )

    sales_srv = SalesService(repo)
    return {
        "repo": repo,
        "sales_service": sales_srv,
        "harvest_service": harvest_srv,
        "farmer": profile,
        "farm": farm,
        "cycle": cycle,
        "harvest": h_rec,
    }


def test_record_produce_sale_strict_arithmetic(sales_env):
    """Test produce sale recording with accurate gross and net revenue arithmetic."""
    srv = sales_env["sales_service"]
    farmer_id = sales_env["farmer"].farmer_id
    farm_id = sales_env["farm"].farm_id
    cycle_id = sales_env["cycle"].cycle_id
    harvest_id = sales_env["harvest"].harvest_id

    # Sell 50 quintals @ 2275 = Gross 1,13,750
    # Commission: 1500, Transport: 1200, Loading: 300 -> Total Deductions: 3000
    # Net: 1,13,750 - 3000 = 1,10,750
    payload = FarmSaleCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        harvest_id=harvest_id,
        crop_name="Wheat",
        quantity_sold=50.0,
        quantity_unit="quintal",
        realized_price_per_unit=2275.0,
        gross_amount=113750.0,
        commission_charges=1500.0,
        transport_charges=1200.0,
        deductions_charges=300.0,
        net_amount_received=110750.0,
        sale_date="2026-04-20",
        buyer_name="Khanna Grain Market Commission Agent #42",
        payment_status=PaymentStatus.PAID,
    )

    sale_rec = srv.record_sale(farmer_id=farmer_id, payload=payload, user_id="usr_baldev")
    assert sale_rec.sale_id.startswith("sal_")
    assert sale_rec.gross_amount == 113750.0
    assert sale_rec.net_amount_received == 110750.0
    assert sale_rec.version_number == 1
    assert sale_rec.payment_status == PaymentStatus.PAID


def test_record_sale_arithmetic_mismatch_raises_400(sales_env):
    """Test that mismatched gross revenue raises 400 Bad Request."""
    srv = sales_env["sales_service"]
    farmer_id = sales_env["farmer"].farmer_id
    farm_id = sales_env["farm"].farm_id
    cycle_id = sales_env["cycle"].cycle_id

    # 10 qtl @ 2000 = 20000, but supplied 25000
    payload = FarmSaleCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        crop_name="Wheat",
        quantity_sold=10.0,
        quantity_unit="quintal",
        realized_price_per_unit=2000.0,
        gross_amount=25000.0,  # mismatch!
        sale_date="2026-04-21",
    )

    with pytest.raises(HTTPException) as exc_info:
        srv.record_sale(farmer_id=farmer_id, payload=payload, user_id="usr_baldev")
    assert exc_info.value.status_code == 400
    assert "Gross amount mismatch" in exc_info.value.detail


def test_update_produce_sale_recalculates_net(sales_env):
    """Test that updating sale price or deductions correctly recalculates gross and net revenue."""
    srv = sales_env["sales_service"]
    farmer_id = sales_env["farmer"].farmer_id
    farm_id = sales_env["farm"].farm_id
    cycle_id = sales_env["cycle"].cycle_id

    sale = srv.record_sale(
        farmer_id=farmer_id,
        payload=FarmSaleCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            crop_name="Wheat",
            quantity_sold=20.0,
            realized_price_per_unit=2100.0,
            sale_date="2026-04-22",
        ),
        user_id="usr_baldev",
    )
    assert sale.gross_amount == 42000.0
    assert sale.net_amount_received == 42000.0

    # Update realized price to 2200 and add 1000 transport deduction
    # New Gross: 20 * 2200 = 44000; New Net: 44000 - 1000 = 43000
    updated = srv.update_sale(
        sale_id=sale.sale_id,
        farmer_id=farmer_id,
        payload=FarmSaleUpdate(realized_price_per_unit=2200.0, transport_charges=1000.0),
        user_id="usr_baldev",
    )
    assert updated.realized_price_per_unit == 2200.0
    assert updated.gross_amount == 44000.0
    assert updated.net_amount_received == 43000.0
    assert updated.version_number == 2


def test_sale_tenant_isolation(sales_env):
    """Test that unauthorized farmers cannot access sale records."""
    srv = sales_env["sales_service"]
    farmer_id = sales_env["farmer"].farmer_id
    farm_id = sales_env["farm"].farm_id
    cycle_id = sales_env["cycle"].cycle_id

    sale = srv.record_sale(
        farmer_id=farmer_id,
        payload=FarmSaleCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            crop_name="Wheat",
            quantity_sold=15.0,
            realized_price_per_unit=2200.0,
            sale_date="2026-04-23",
        ),
        user_id="usr_baldev",
    )

    with pytest.raises(HTTPException) as exc:
        srv.get_sale(sale.sale_id, farmer_id="fmr_intruder")
    assert exc.value.status_code == 404
