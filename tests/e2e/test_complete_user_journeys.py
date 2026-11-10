"""Realistic end-to-end user journeys for Farmer, Field Officer, Agronomist, and Administrator (Phase 12)."""

import os
import tempfile
import uuid
from decimal import Decimal
import pytest

from backend.app.main import (
    auth_service,
    create_farm_expense,
    create_farm_harvest,
    create_farm_sale,
    get_admin_dashboard_overview,
    get_crop_cycle_profitability,
    get_field_officer_dashboard_overview,
    get_prometheus_metrics,
    list_review_cases,
    system,
)
from backend.app.schemas.farm import (
    CropCycleCreate,
    CropSeason,
    FarmCreate,
    PlotCreate,
    UserRegistrationRequest,
    UserRole,
)
from backend.app.schemas.economics import (
    ExpenseCategory,
    FarmExpenseCreate,
    FarmHarvestCreate,
    FarmSaleCreate,
    PaymentStatus,
)


def test_complete_farmer_journey():
    """Complete Farmer Journey: Register -> Farm -> Plot -> Cycle -> Expense -> Harvest -> Sale -> Profitability."""
    rand = uuid.uuid4().hex[:6]

    # 1. Farmer Registration & Authentication
    reg_req = UserRegistrationRequest(
        username=f"kisan_{rand}",
        password="KisanPassword2026!",
        full_name="Ramesh Patel",
        phone_number=f"98765{rand[:5]}",
        preferred_language="hi-IN",
    )
    user, profile, token = auth_service.register_farmer(reg_req)
    assert user.role == UserRole.FARMER

    # 2. Create Farm
    farm = system.farm_context.farm_service.create_farm(
        profile.farmer_id,
        FarmCreate(
            farm_name="Patel Krishi Farm",
            state="Gujarat",
            district="Anand",
            village="Mogar",
            total_area=3.0,
            area_unit="acre",
            ownership_type="owned",
        ),
    )
    assert farm.farm_id is not None
    assert farm.farm_name == "Patel Krishi Farm"

    # 3. Add Plot
    plot = system.farm_context.plot_service.create_plot(
        farm.farm_id,
        profile.farmer_id,
        PlotCreate(
            plot_name="North Field",
            area=3.0,
            area_unit="acre",
        ),
    )
    assert plot.plot_id is not None

    # 4. Add Crop Cycle (Wheat, Rabi)
    cycle = system.farm_context.cycle_service.create_crop_cycle(
        plot.plot_id,
        profile.farmer_id,
        CropCycleCreate(
            crop_name="Wheat",
            season=CropSeason.RABI,
            sowing_date="2026-10-15",
            target_yield=22.0,
            target_yield_unit="quintal",
        ),
    )
    assert cycle.cycle_id is not None

    # 5. Record Cultivation Expense (Fertilizer DAP: ₹2,700)
    exp = create_farm_expense(
        FarmExpenseCreate(
            farm_id=farm.farm_id,
            plot_id=plot.plot_id,
            crop_cycle_id=cycle.cycle_id,
            expense_category=ExpenseCategory.FERTILIZER_SOIL,
            subcategory="DAP",
            expense_description="Fertilizer DAP 2 bags",
            quantity=2.0,
            quantity_unit="Bags",
            unit_price=1350.0,
            total_amount=2700.0,
            purchase_date="2026-10-20",
            payment_status=PaymentStatus.PAID,
        ),
        user=user,
    )
    assert exp.expense_id is not None

    # 6. Record Harvest Event (50 Quintals)
    harv = create_farm_harvest(
        FarmHarvestCreate(
            farm_id=farm.farm_id,
            plot_id=plot.plot_id,
            crop_cycle_id=cycle.cycle_id,
            crop_name="Wheat",
            harvest_date="2026-11-01",
            harvested_quantity=50.0,
            quantity_unit="quintal",
            sold_quantity=45.0,
            retained_quantity=5.0,
        ),
        user=user,
    )
    assert harv.harvest_id is not None

    # 7. Record Produce Mandi Sale (45 Quintals @ ₹2,200 = ₹99,000 - ₹1,000 deductions = ₹98,000)
    sale = create_farm_sale(
        FarmSaleCreate(
            farm_id=farm.farm_id,
            plot_id=plot.plot_id,
            crop_cycle_id=cycle.cycle_id,
            crop_name="Wheat",
            sale_date="2026-11-03",
            quantity_sold=45.0,
            quantity_unit="quintal",
            realized_price_per_unit=2200.0,
            gross_amount=99000.0,
            deductions_charges=1000.0,
            net_amount_received=98000.0,
            payment_status=PaymentStatus.PAID,
        ),
        user=user,
    )
    assert sale.sale_id is not None

    # 8. View Profitability Report
    metrics = get_crop_cycle_profitability(cycle.cycle_id, user=user)
    assert metrics.crop_name == "Wheat"
    assert metrics.total_actual_cost == 2700.0
    assert metrics.total_net_revenue == 98000.0
    assert metrics.net_profit_or_loss == 95300.0
    assert metrics.is_profitable is True
    assert metrics.is_provisional is True


def test_field_officer_journey():
    """Field Officer Journey: Authenticate -> View assigned territory -> Review officer overview."""
    rand = uuid.uuid4().hex[:6]
    officer_user, officer_token = auth_service.provision_privileged_user(
        username=f"fo_{rand}",
        password="OfficerPass2026!",
        role=UserRole.FIELD_OFFICER,
        full_name=f"Officer {rand}",
    )
    assert officer_user.role == UserRole.FIELD_OFFICER

    # Check Officer Dashboard Overview
    overview = get_field_officer_dashboard_overview(user=officer_user)
    assert overview.officer_name == f"fo_{rand}"
    assert isinstance(overview.metrics, dict)
    assert overview.data_freshness is not None


def test_agronomist_expert_journey():
    """Agronomist Journey: Authenticate with AGRONOMIST role -> Access expert queue."""
    rand = uuid.uuid4().hex[:6]
    agro_user, agro_token = auth_service.provision_privileged_user(
        username=f"dr_{rand}",
        password="DoctorPass2026!",
        role=UserRole.AGRONOMIST,
        full_name=f"Dr. {rand} (ICAR)",
    )
    assert agro_user.role == UserRole.AGRONOMIST

    # Access expert review cases queue
    cases = list_review_cases(user=agro_user)
    assert isinstance(cases, list)


def test_administrator_platform_journey():
    """Administrator Journey: Authenticate -> Platform telemetry -> Health inspection."""
    rand = uuid.uuid4().hex[:6]
    admin_user, admin_token = auth_service.provision_privileged_user(
        username=f"admin_{rand}",
        password="AdminPassword2026!",
        role=UserRole.ADMINISTRATOR,
        full_name=f"Admin {rand}",
    )
    assert admin_user.role == UserRole.ADMINISTRATOR

    # 1. Check Platform Admin Dashboard
    admin_dash = get_admin_dashboard_overview(user=admin_user)
    assert admin_dash.admin_name == f"admin_{rand}"
    assert admin_dash.data_freshness == "Live Operational Telemetry"
    assert isinstance(admin_dash.metrics, dict)

    # 2. Query Prometheus Metrics
    metrics_resp = get_prometheus_metrics(format="text")
    body_text = metrics_resp.body.decode("utf-8") if hasattr(metrics_resp, "body") else str(metrics_resp)
    assert "agriguardian_uptime_seconds" in body_text
