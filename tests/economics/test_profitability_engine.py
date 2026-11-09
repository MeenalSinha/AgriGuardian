"""Unit and integration tests for Phase 11 Crop Cycle Profitability & Break-Even Intelligence."""

import tempfile
from decimal import Decimal
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import (
    ExpenseCategory,
    FarmExpenseCreate,
    FarmHarvestCreate,
    FarmSaleCreate,
    PaymentStatus,
)
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
from backend.app.services.economics.expense_service import ExpenseService
from backend.app.services.economics.harvest_service import HarvestService
from backend.app.services.economics.profitability_service import ProfitabilityService
from backend.app.services.economics.sales_service import SalesService


@pytest.fixture
def prof_env():
    """Create test repository and complete financial lifecycle for a crop cycle."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_mohan",
        username="mohan",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_mohan",
        user_id="usr_mohan",
        full_name="Mohan Lal",
        phone_number="9876543240",
        state="Madhya Pradesh",
        district="Ujjain",
        sub_district="Ghatiya",
        village="Panbihar",
        primary_language="hi",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_mohan_1",
        farmer_id="fmr_mohan",
        farm_name="Malwa Black Soil Farms",
        total_area=5.0,
        total_area_acres=5.0,
        ownership_type=OwnershipType.OWNED,
        state="Madhya Pradesh",
        district="Ujjain",
        village="Panbihar",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_mohan_1",
        farm_id="farm_mohan_1",
        farmer_id="fmr_mohan",
        plot_name="Soybean Plot A",
        area=2.0,
        area_acres=2.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    # Completed Soybean Cycle (2 acres)
    cycle_comp = CropCycle(
        cycle_id="cycle_soybean_2026",
        farmer_id="fmr_mohan",
        farm_id="farm_mohan_1",
        plot_id="plot_mohan_1",
        crop_name="Soybean",
        variety_name="JS 20-34",
        season=CropSeason.KHARIF,
        sowing_date="2026-06-20",
        current_stage=CropGrowthStage.HARVESTED,
        status=CropCycleStatus.COMPLETED,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle_comp)

    # Ongoing Wheat Cycle (3 acres) - Provisional
    plot2 = Plot(
        plot_id="plot_mohan_2",
        farm_id="farm_mohan_1",
        farmer_id="fmr_mohan",
        plot_name="Wheat Plot B",
        area=3.0,
        area_acres=3.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot2)

    cycle_ongoing = CropCycle(
        cycle_id="cycle_wheat_ongoing",
        farmer_id="fmr_mohan",
        farm_id="farm_mohan_1",
        plot_id="plot_mohan_2",
        crop_name="Wheat",
        variety_name="Lok-1",
        season=CropSeason.RABI,
        sowing_date="2026-11-01",
        current_stage=CropGrowthStage.VEGETATIVE_TILLERING,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle_ongoing)

    exp_srv = ExpenseService(repo)
    hrv_srv = HarvestService(repo)
    sale_srv = SalesService(repo)
    prof_srv = ProfitabilityService(repo)

    # Incur expenses on Soybean Cycle (Total: 25,000)
    exp_srv.record_expense(
        farmer_id="fmr_mohan",
        payload=FarmExpenseCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            expense_category=ExpenseCategory.SEEDS_PLANTING,
            expense_description="Soybean certified seed",
            total_amount=6000.0,
            purchase_date="2026-06-18",
        ),
        user_id="usr_mohan",
    )
    exp_srv.record_expense(
        farmer_id="fmr_mohan",
        payload=FarmExpenseCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            expense_category=ExpenseCategory.FERTILIZER_SOIL,
            expense_description="SSP and Potash",
            total_amount=5000.0,
            purchase_date="2026-06-25",
        ),
        user_id="usr_mohan",
    )
    exp_srv.record_expense(
        farmer_id="fmr_mohan",
        payload=FarmExpenseCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            expense_category=ExpenseCategory.LABOR,
            expense_description="Weeding and spraying labor",
            total_amount=7000.0,
            purchase_date="2026-07-20",
        ),
        user_id="usr_mohan",
    )
    exp_srv.record_expense(
        farmer_id="fmr_mohan",
        payload=FarmExpenseCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            expense_category=ExpenseCategory.MACHINERY_FUEL,
            expense_description="Harvester rent",
            total_amount=7000.0,
            purchase_date="2026-10-05",
        ),
        user_id="usr_mohan",
    )

    # Harvest Soybean: 16 quintals from 2 acres = 8 qtl/acre
    hrv_srv.record_harvest(
        farmer_id="fmr_mohan",
        payload=FarmHarvestCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            harvest_date="2026-10-06",
            crop_name="Soybean",
            harvested_quantity=16.0,
            stored_quantity=16.0,
        ),
        user_id="usr_mohan",
    )

    # Sale Soybean: 16 quintals @ 4800 = Gross 76,800. Deductions: 1800 -> Net 75,000
    sale_srv.record_sale(
        farmer_id="fmr_mohan",
        payload=FarmSaleCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_1",
            crop_cycle_id="cycle_soybean_2026",
            crop_name="Soybean",
            quantity_sold=16.0,
            realized_price_per_unit=4800.0,
            gross_amount=76800.0,
            deductions_charges=600.0,
            commission_charges=800.0,
            transport_charges=400.0,
            net_amount_received=75000.0,
            sale_date="2026-10-12",
        ),
        user_id="usr_mohan",
    )

    # Incur preliminary expense on ongoing Wheat cycle (8,000)
    exp_srv.record_expense(
        farmer_id="fmr_mohan",
        payload=FarmExpenseCreate(
            farm_id="farm_mohan_1",
            plot_id="plot_mohan_2",
            crop_cycle_id="cycle_wheat_ongoing",
            expense_category=ExpenseCategory.SEEDS_PLANTING,
            expense_description="Wheat Seed Lok-1",
            total_amount=8000.0,
            purchase_date="2026-11-02",
        ),
        user_id="usr_mohan",
    )

    return {
        "repo": repo,
        "profitability_service": prof_srv,
        "farmer_id": "fmr_mohan",
        "cycle_completed_id": "cycle_soybean_2026",
        "cycle_ongoing_id": "cycle_wheat_ongoing",
    }


def test_completed_cycle_profitability(prof_env):
    """Test comprehensive financial metrics on completed crop cycle."""
    srv = prof_env["profitability_service"]
    farmer_id = prof_env["farmer_id"]
    cycle_id = prof_env["cycle_completed_id"]

    m = srv.calculate_crop_cycle_profitability(cycle_id, farmer_id)

    # Total cost: 6000 + 5000 + 7000 + 7000 = 25,000
    assert m.total_actual_cost == 25000.0
    # Net revenue: 75,000
    assert m.total_net_revenue == 75000.0
    assert m.total_gross_revenue == 76800.0
    assert m.total_deductions == 1800.0

    # Net Profit = 75,000 - 25,000 = 50,000
    assert m.net_profit_or_loss == 50000.0
    assert m.is_profitable is True
    assert m.is_provisional is False
    assert len(m.missing_data_fields) == 0

    # Area = 2.0 acres
    assert m.cultivated_area_acres == 2.0
    # Cost per acre = 25,000 / 2 = 12,500
    assert m.cost_per_acre == 12500.0
    # Cost per quintal = 25,000 / 16 = 1562.50
    assert m.cost_per_quintal == 1562.50
    # Cost per kg = 1562.50 / 100 = 15.63
    assert m.cost_per_kg == 15.63

    # Break-Even Price = 25,000 / 16 = 1562.50
    assert m.break_even_selling_price_per_quintal == 1562.50

    # ROI % = (50,000 / 25,000) * 100 = 200.0%
    assert m.return_on_cultivation_cost_pct == 200.0

    # Provenance
    assert "Net Outcome: ₹50,000.00 (PROFIT)" in m.explanation_en
    assert "शुद्ध परिणाम: ₹50,000.00 (लाभ)" in m.explanation_hi


def test_ongoing_cycle_is_flagged_provisional(prof_env):
    """Test that ongoing cycle with missing sales is explicitly marked provisional."""
    srv = prof_env["profitability_service"]
    farmer_id = prof_env["farmer_id"]
    cycle_id = prof_env["cycle_ongoing_id"]

    m = srv.calculate_crop_cycle_profitability(cycle_id, farmer_id)

    assert m.is_provisional is True
    assert "no_harvest_recorded" in m.missing_data_fields
    assert "no_sales_recorded" in m.missing_data_fields
    assert m.total_actual_cost == 8000.0
    assert m.total_net_revenue == 0.0
    assert "Provisional" in m.explanation_en
    assert "अस्थायी परिणाम" in m.explanation_hi


def test_farmer_financial_overview(prof_env):
    """Test aggregation of farmer cash flow and completed vs provisional cycles."""
    srv = prof_env["profitability_service"]
    farmer_id = prof_env["farmer_id"]

    ov = srv.get_farmer_financial_overview(farmer_id)

    # Total expenses incurred = 25,000 + 8,000 = 33,000
    assert ov.total_cultivation_expenses == 33000.0
    # Total net revenue = 75,000
    assert ov.total_realized_revenue == 75000.0
    # Net balance = 75,000 - 33,000 = 42,000
    assert ov.net_balance == 42000.0

    assert ov.completed_cycles_count == 1
    assert ov.provisional_cycles_count == 1
    assert ov.completed_crop_cycles_profit == 50000.0
    assert len(ov.recent_expenses) == 5
    assert len(ov.recent_sales) == 1
