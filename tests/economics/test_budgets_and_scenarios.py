"""Unit and integration tests for Phase 11 Budgeting, Variance Analysis, Scenarios, and Crop Comparison."""

import tempfile
from decimal import Decimal
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import (
    BudgetItemCreate,
    CropComparisonItem,
    CropComparisonRequest,
    CropCycleBudgetCreate,
    CropCycleBudgetUpdate,
    ExpenseCategory,
    FarmExpenseCreate,
    ScenarioSimulateRequest,
    ScenarioType,
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
from backend.app.services.economics.budget_service import BudgetService
from backend.app.services.economics.expense_service import ExpenseService
from backend.app.services.economics.financial_scenario_service import FinancialScenarioService


@pytest.fixture
def plan_env():
    """Create test repository with farmer, farm, plot, and crop cycle."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user = UserRecord(
        user_id="usr_sunil",
        username="sunil",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_sunil",
        user_id="usr_sunil",
        full_name="Sunil Sharma",
        phone_number="9876543250",
        state="Uttar Pradesh",
        district="Aligarh",
        sub_district="Koil",
        village="Sasni",
        primary_language="hi",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_sunil_1",
        farmer_id="fmr_sunil",
        farm_name="Braj Green Farms",
        total_area=4.0,
        total_area_acres=4.0,
        ownership_type=OwnershipType.OWNED,
        state="Uttar Pradesh",
        district="Aligarh",
        village="Sasni",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_sunil_1",
        farm_id="farm_sunil_1",
        farmer_id="fmr_sunil",
        plot_name="Mustard Field",
        area=4.0,
        area_acres=4.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    cycle = CropCycle(
        cycle_id="cycle_mustard_2026",
        farmer_id="fmr_sunil",
        farm_id="farm_sunil_1",
        plot_id="plot_sunil_1",
        crop_name="Mustard",
        variety_name="Pusa Bold",
        season=CropSeason.RABI,
        sowing_date="2026-10-15",
        current_stage=CropGrowthStage.VEGETATIVE_TILLERING,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle)

    budget_srv = BudgetService(repo)
    expense_srv = ExpenseService(repo)
    scenario_srv = FinancialScenarioService(repo)

    return {
        "repo": repo,
        "budget_service": budget_srv,
        "expense_service": expense_srv,
        "scenario_service": scenario_srv,
        "farmer_id": "fmr_sunil",
        "farm_id": "farm_sunil_1",
        "cycle_id": "cycle_mustard_2026",
    }


def test_create_and_fetch_crop_cycle_budget(plan_env):
    """Test creating a seasonal budget with category items."""
    srv = plan_env["budget_service"]
    farmer_id = plan_env["farmer_id"]
    farm_id = plan_env["farm_id"]
    cycle_id = plan_env["cycle_id"]

    items = [
        BudgetItemCreate(expense_category=ExpenseCategory.SEEDS_PLANTING, planned_amount=4000.0),
        BudgetItemCreate(expense_category=ExpenseCategory.FERTILIZER_SOIL, planned_amount=6000.0),
        BudgetItemCreate(expense_category=ExpenseCategory.CROP_PROTECTION, planned_amount=3000.0),
        BudgetItemCreate(expense_category=ExpenseCategory.LABOR, planned_amount=8000.0),
        BudgetItemCreate(expense_category=ExpenseCategory.MACHINERY_FUEL, planned_amount=5000.0),
    ]

    payload = CropCycleBudgetCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        budget_name="Rabi 2026 Mustard Budget",
        season="Rabi",
        total_budget_amount=26000.0,
        items=items,
    )

    budget = srv.create_budget(farmer_id=farmer_id, payload=payload, user_id="usr_sunil")
    assert budget.budget_id.startswith("bdg_")
    assert budget.total_budget_amount == 26000.0
    assert len(budget.items) == 5

    # Fetch budget by ID
    fetched = srv.get_budget(budget.budget_id, farmer_id=farmer_id)
    assert fetched.budget_name == "Rabi 2026 Mustard Budget"
    assert len(fetched.items) == 5


def test_budget_variance_and_over_budget_detection(plan_env):
    """Test budget variance analysis and threshold exceed detection."""
    b_srv = plan_env["budget_service"]
    e_srv = plan_env["expense_service"]
    farmer_id = plan_env["farmer_id"]
    farm_id = plan_env["farm_id"]
    cycle_id = plan_env["cycle_id"]

    # 1. Create a planned budget of 10,000 (Seeds: 3,000, Fertilizer: 7,000)
    b_srv.create_budget(
        farmer_id=farmer_id,
        payload=CropCycleBudgetCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            budget_name="Small Mustard Budget",
            total_budget_amount=10000.0,
            items=[
                BudgetItemCreate(expense_category=ExpenseCategory.SEEDS_PLANTING, planned_amount=3000.0),
                BudgetItemCreate(expense_category=ExpenseCategory.FERTILIZER_SOIL, planned_amount=7000.0),
            ],
        ),
        user_id="usr_sunil",
    )

    # 2. Incur expenses: Seeds = 2,500 (under budget), Fertilizer = 9,000 (over budget by 2,000)
    # Total actual = 11,500 > Planned 10,000 (Over budget by 1,500!)
    e_srv.record_expense(
        farmer_id=farmer_id,
        payload=FarmExpenseCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            expense_category=ExpenseCategory.SEEDS_PLANTING,
            expense_description="Mustard seed",
            total_amount=2500.0,
            purchase_date="2026-10-16",
        ),
        user_id="usr_sunil",
    )
    e_srv.record_expense(
        farmer_id=farmer_id,
        payload=FarmExpenseCreate(
            farm_id=farm_id,
            crop_cycle_id=cycle_id,
            expense_category=ExpenseCategory.FERTILIZER_SOIL,
            expense_description="Fertilizers",
            total_amount=9000.0,
            purchase_date="2026-10-20",
        ),
        user_id="usr_sunil",
    )

    # 3. Calculate Variance Report
    report = b_srv.calculate_budget_variance(crop_cycle_id=cycle_id, farmer_id=farmer_id)
    assert report.total_planned == 10000.0
    assert report.total_actual == 11500.0
    assert report.amount_over_budget == 1500.0
    assert report.is_exceeded is True
    assert report.total_variance_pct == 15.0  # 1,500 / 10,000 * 100 = 15%

    # Category checks
    seed_item = next(it for it in report.categories if it.expense_category == ExpenseCategory.SEEDS_PLANTING)
    assert seed_item.planned_amount == 3000.0
    assert seed_item.actual_amount == 2500.0
    assert seed_item.variance_amount == -500.0
    assert seed_item.is_over_budget is False

    fert_item = next(it for it in report.categories if it.expense_category == ExpenseCategory.FERTILIZER_SOIL)
    assert fert_item.planned_amount == 7000.0
    assert fert_item.actual_amount == 9000.0
    assert fert_item.variance_amount == 2000.0
    assert fert_item.is_over_budget is True


def test_scenario_simulator_deterministic(plan_env):
    """Test deterministic scenario projection and sensitivity analysis."""
    srv = plan_env["scenario_service"]
    farmer_id = plan_env["farmer_id"]

    # 2.5 acres Mustard: Cost 13,000/ac, Yield 8.0 qtl/ac, Price 5450/qtl, 5% loss, 35/qtl transport
    req = ScenarioSimulateRequest(
        crop_name="Mustard",
        scenario_name="Mustard Plan Base",
        scenario_type=ScenarioType.BASE,
        planned_area_acres=2.5,
        expected_cost_per_acre=13000.0,
        expected_yield_quintals_per_acre=8.0,
        expected_price_per_quintal=5450.0,
        expected_harvest_loss_pct=5.0,
        transport_cost_per_quintal=35.0,
    )

    res = srv.simulate_scenario(req)

    # Total Cost = 2.5 * 13,000 = 32,500
    assert res.total_expected_cost == 32500.0
    # Gross Prod = 2.5 * 8 = 20.0 qtl; Saleable Prod = 20.0 * 0.95 = 19.0 qtl
    assert res.gross_production_quintals == 20.0
    assert res.saleable_production_quintals == 19.0

    # Gross Rev = 19.0 * 5450 = 1,03,550
    assert res.expected_gross_revenue == 103550.0
    # Transport = 19.0 * 35 = 665; Net Rev = 1,03,550 - 665 = 1,02,885
    assert res.expected_net_revenue == 102885.0
    # Net Return = 1,02,885 - 32,500 = 70,385
    assert res.expected_net_return == 70385.0

    # Break-Even Price = 32,500 / 19.0 = 1710.53/qtl
    assert res.break_even_price_per_quintal == 1710.53

    # Sensitivity analysis keys
    assert "price_minus_10pct" in res.sensitivity_analysis
    assert "price_plus_10pct" in res.sensitivity_analysis
    assert "yield_minus_10pct" in res.sensitivity_analysis

    # Save scenario
    saved = srv.save_scenario(farmer_id, req, "usr_sunil")
    assert saved.scenario_id.startswith("scn_")

    fetched = srv.get_scenario(saved.scenario_id, farmer_id)
    assert fetched.crop_name == "Mustard"


def test_prospective_crop_comparison(plan_env):
    """Test comparative economic evaluation between multiple prospective crops on a plot."""
    srv = plan_env["scenario_service"]

    crops = [
        CropComparisonItem(
            crop_name="Wheat",
            estimated_cost_per_acre=18000.0,
            expected_yield_per_acre=19.0,
            expected_price_per_quintal=2275.0,
            gross_revenue_per_acre=19.0 * 2275.0,  # 43,225
            net_return_per_acre=(19.0 * 2275.0) - 18000.0,  # 25,225
            break_even_price=18000.0 / 19.0,
            roi_pct=((25225.0 / 18000.0) * 100.0),
            data_source="assumptions",
        ),
        CropComparisonItem(
            crop_name="Mustard",
            estimated_cost_per_acre=13000.0,
            expected_yield_per_acre=8.5,
            expected_price_per_quintal=5450.0,
            gross_revenue_per_acre=8.5 * 5450.0,  # 46,325
            net_return_per_acre=(8.5 * 5450.0) - 13000.0,  # 33,325
            break_even_price=13000.0 / 8.5,
            roi_pct=((33325.0 / 13000.0) * 100.0),
            data_source="assumptions",
        ),
    ]

    req = CropComparisonRequest(
        plot_area_acres=3.0,
        season="Rabi",
        state="Uttar Pradesh",
        district="Aligarh",
        crops=crops,
    )

    report = srv.compare_crops(req)
    assert report.plot_area_acres == 3.0
    assert report.best_net_return_crop == "Mustard"
    assert report.best_roi_crop == "Mustard"
    assert "Agricultural scenario comparisons are based strictly" in report.disclaimer
