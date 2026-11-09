"""FastAPI REST Endpoint and RBAC security tests for Phase 11 Farm Economics."""

import tempfile
import pytest
from fastapi import HTTPException

import backend.app.main as main_app
from backend.app.main import (
    create_crop_cycle_budget,
    create_farm_expense,
    create_farm_harvest,
    create_farm_sale,
    create_financial_benchmark,
    delete_farm_expense,
    delete_farm_harvest,
    delete_farm_sale,
    delete_financial_scenario,
    get_crop_cycle_budget,
    get_crop_cycle_budget_variance,
    get_crop_cycle_profitability,
    get_farm_expense,
    get_farm_harvest,
    get_farm_sale,
    get_farmer_financial_overview,
    get_financial_scenario,
    list_crop_cycle_budgets,
    list_farm_expenses,
    list_farm_harvests,
    list_farm_sales,
    list_financial_benchmarks,
    list_financial_scenarios,
    save_financial_scenario,
    simulate_financial_scenario,
    update_crop_cycle_budget,
    update_farm_expense,
    update_farm_harvest,
    update_farm_sale,
)
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import (
    BudgetItemCreate,
    CropCycleBudgetCreate,
    CropCycleBudgetUpdate,
    ExpenseCategory,
    ExpenseCorrectionCreate,
    FarmExpenseCreate,
    FarmExpenseUpdate,
    FarmHarvestCreate,
    FarmHarvestUpdate,
    FarmSaleCreate,
    FarmSaleUpdate,
    FinancialBenchmarkRecord,
    PaymentStatus,
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
from backend.app.services.economics import EconomicsService


@pytest.fixture
def api_env(monkeypatch):
    """Set up temporary test database and inject repository into main_app.economics_service."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    # Patch system repo and economics_service in main_app
    test_econ = EconomicsService(repo=repo)
    monkeypatch.setattr(main_app, "economics_service", test_econ)
    monkeypatch.setattr(main_app.system.farm_context, "repo", repo)

    # 1. Farmer User
    farmer_user = UserRecord(
        user_id="usr_api_farmer",
        username="apifarmer",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(farmer_user)
    farmer_prof = FarmerProfile(
        farmer_id="fmr_api_1",
        user_id="usr_api_farmer",
        full_name="API Farmer Dev",
        phone_number="9876543290",
        state="Maharashtra",
        district="Nashik",
        sub_district="Dindori",
        village="Dindori",
        primary_language="mr",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(farmer_prof)

    # 2. Admin User
    admin_user = UserRecord(
        user_id="usr_api_admin",
        username="apiadmin",
        password_hash="hash",
        salt="salt",
        role=UserRole.ADMINISTRATOR,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(admin_user)

    # 3. Farm, Plot, CropCycle
    farm = Farm(
        farm_id="farm_api_1",
        farmer_id="fmr_api_1",
        farm_name="Nashik Grape & Onion Farm",
        total_area=4.0,
        total_area_acres=4.0,
        ownership_type=OwnershipType.OWNED,
        state="Maharashtra",
        district="Nashik",
        village="Dindori",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plot_api_1",
        farm_id="farm_api_1",
        farmer_id="fmr_api_1",
        plot_name="Onion Field",
        area=4.0,
        area_acres=4.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_plot(plot)

    cycle = CropCycle(
        cycle_id="cycle_onion_2026",
        farmer_id="fmr_api_1",
        farm_id="farm_api_1",
        plot_id="plot_api_1",
        crop_name="Onion",
        variety_name="Bhima Super",
        season=CropSeason.KHARIF,
        sowing_date="2026-07-01",
        current_stage=CropGrowthStage.HARVEST_READY,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_crop_cycle(cycle)

    return {
        "repo": repo,
        "farmer_user": farmer_user,
        "admin_user": admin_user,
        "farmer_id": "fmr_api_1",
        "farm_id": "farm_api_1",
        "cycle_id": "cycle_onion_2026",
    }


def test_api_expense_crud_lifecycle(api_env):
    """Test FastAPI expense creation, listing, updating, and deletion."""
    user = api_env["farmer_user"]
    farm_id = api_env["farm_id"]
    cycle_id = api_env["cycle_id"]

    # 1. Create Expense
    payload = FarmExpenseCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        expense_category=ExpenseCategory.FERTILIZER_SOIL,
        expense_description="Complex NPK Fertilizer",
        total_amount=3200.0,
        purchase_date="2026-07-15",
    )
    exp = create_farm_expense(payload=payload, user=user)
    assert exp.expense_id.startswith("exp_")
    assert exp.total_amount == 3200.0

    # 2. List Expenses
    lst = list_farm_expenses(farm_id=farm_id, user=user)
    assert len(lst) == 1
    assert lst[0].expense_id == exp.expense_id

    # 3. Get Expense
    fetched = get_farm_expense(expense_id=exp.expense_id, user=user)
    assert fetched.expense_description == "Complex NPK Fertilizer"

    # 4. Update Expense
    up_payload = FarmExpenseUpdate(expense_description="Complex NPK 19-19-19 (Updated)")
    updated = update_farm_expense(expense_id=exp.expense_id, payload=up_payload, user=user)
    assert updated.expense_description == "Complex NPK 19-19-19 (Updated)"

    # 5. Delete Expense
    del_res = delete_farm_expense(expense_id=exp.expense_id, user=user)
    assert del_res["message"] == "Expense deleted successfully"

    # Verify deleted
    with pytest.raises(HTTPException) as exc:
        get_farm_expense(expense_id=exp.expense_id, user=user)
    assert exc.value.status_code == 404


def test_api_harvest_and_sales_flow(api_env):
    """Test harvest and produce sales API endpoints."""
    user = api_env["farmer_user"]
    farm_id = api_env["farm_id"]
    cycle_id = api_env["cycle_id"]

    # Record harvest
    h_payload = FarmHarvestCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        harvest_date="2026-10-10",
        crop_name="Onion",
        harvested_quantity=60.0,
        stored_quantity=60.0,
    )
    h_rec = create_farm_harvest(payload=h_payload, user=user)
    assert h_rec.harvest_id.startswith("hrv_")
    assert h_rec.harvested_quantity == 60.0

    # Record sale
    s_payload = FarmSaleCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        harvest_id=h_rec.harvest_id,
        crop_name="Onion",
        quantity_sold=40.0,
        realized_price_per_unit=1800.0,
        gross_amount=72000.0,
        transport_charges=800.0,
        commission_charges=1200.0,
        net_amount_received=70000.0,
        sale_date="2026-10-14",
    )
    s_rec = create_farm_sale(payload=s_payload, user=user)
    assert s_rec.sale_id.startswith("sal_")
    assert s_rec.net_amount_received == 70000.0

    # Profitability overview
    overview = get_farmer_financial_overview(user=user)
    assert overview.total_realized_revenue == 70000.0


def test_api_budget_and_variance(api_env):
    """Test crop cycle budget API endpoints and variance report."""
    user = api_env["farmer_user"]
    farm_id = api_env["farm_id"]
    cycle_id = api_env["cycle_id"]

    b_payload = CropCycleBudgetCreate(
        farm_id=farm_id,
        crop_cycle_id=cycle_id,
        budget_name="Kharif Onion Budget",
        total_budget_amount=20000.0,
        items=[BudgetItemCreate(expense_category=ExpenseCategory.SEEDS_PLANTING, planned_amount=5000.0)],
    )
    b_rec = create_crop_cycle_budget(payload=b_payload, user=user)
    assert b_rec.budget_id.startswith("bdg_")

    variance = get_crop_cycle_budget_variance(crop_cycle_id=cycle_id, user=user)
    assert variance.total_planned == 20000.0


def test_benchmark_creation_rbac(api_env):
    """Test RBAC on benchmark creation: Farmer is forbidden (403), Admin is authorized."""
    farmer_user = api_env["farmer_user"]
    admin_user = api_env["admin_user"]

    benchmark = FinancialBenchmarkRecord(
        benchmark_id="bm_onion_nashik_2026",
        crop_name="Onion",
        state="Maharashtra",
        district="Nashik",
        reporting_period="Kharif 2026",
        cost_per_acre=24000.0,
        expected_yield_per_acre=80.0,
        source_name="ICAR-DOGR (Directorate of Onion and Garlic Research)",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )

    # 1. Farmer attempting to create benchmark -> 403 Forbidden
    with pytest.raises(HTTPException) as exc:
        create_financial_benchmark(payload=benchmark, user=farmer_user)
    assert exc.value.status_code == 403
    assert "Only administrators and field officers" in exc.value.detail

    # 2. Administrator creating benchmark -> Success
    created = create_financial_benchmark(payload=benchmark, user=admin_user)
    assert created.benchmark_id == "bm_onion_nashik_2026"

    # 3. Query benchmarks via API -> Both can read
    benchmarks = list_financial_benchmarks(crop_name="Onion", user=farmer_user)
    assert len(benchmarks) == 1
    assert benchmarks[0].source_name.startswith("ICAR-DOGR")

