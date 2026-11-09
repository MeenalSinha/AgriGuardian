"""Unit and integration tests for Farmer Analytics Service."""

from datetime import date, timedelta
import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityType,
    AreaUnit,
    CropCycleCreate,
    CropGrowthStage,
    CropSeason,
    FarmActivityCreate,
    FarmCreate,
    OwnershipType,
    PlotCreate,
    UserRegistrationRequest,
    UserRole,
)
from backend.app.services.analytics.farmer_analytics import FarmerAnalyticsService
from backend.app.services.auth.auth_service import AuthService
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_context_service import FarmContextService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService


@pytest.fixture
def test_setup():
    """Create isolated SQLite database and services."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=db_path)
    auth_srv = AuthService(repo=repo)
    profile_srv = FarmerProfileService(repo=repo)
    farm_srv = FarmService(repo=repo)
    plot_srv = PlotService(repo=repo)
    cycle_srv = CropCycleService(repo=repo)
    calendar_srv = CropCalendarService(repo=repo)
    context_srv = FarmContextService(
        repo=repo,
        profile_service=profile_srv,
        farm_service=farm_srv,
        plot_service=plot_srv,
        cycle_service=cycle_srv,
        calendar_service=calendar_srv,
    )
    analytics_srv = FarmerAnalyticsService(
        farm_db=repo,
        farm_context_service=context_srv,
    )

    # Register Farmer A
    _, profile_a, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_a_dash",
        password="ValidPassword123!",
        full_name="Ramesh Kumar",
        phone_number="9876543210",
        preferred_language="hi",
        role=UserRole.FARMER,
    ))

    # Register Farmer B
    _, profile_b, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_b_dash",
        password="ValidPassword123!",
        full_name="Suresh Verma",
        phone_number="9876543211",
        preferred_language="en",
        role=UserRole.FARMER,
    ))

    yield {
        "repo": repo,
        "analytics_srv": analytics_srv,
        "farm_srv": farm_srv,
        "plot_srv": plot_srv,
        "cycle_srv": cycle_srv,
        "calendar_srv": calendar_srv,
        "farmer_a": profile_a,
        "farmer_b": profile_b,
        "db_path": db_path,
    }

    try:
        os.remove(db_path)
    except Exception:
        pass


def test_farmer_dashboard_empty_farms(test_setup):
    """Verify metrics when farmer has no farms yet."""
    analytics = test_setup["analytics_srv"]
    fa = test_setup["farmer_a"]

    overview = analytics.get_farmer_overview(farmer_id=fa.farmer_id)
    assert overview.farmer.farmer_id == fa.farmer_id
    assert overview.active_farm is None
    assert overview.metrics["farmer_active_farms"].value == 0
    assert overview.metrics["farmer_total_acres"].value == 0.0
    assert overview.metrics["farmer_active_crops"].value == 0
    assert overview.metrics["farmer_due_tasks"].value == 0


def test_farmer_dashboard_with_farms_and_phenology(test_setup):
    """Verify acreage calculation, crop phenology progress, and ICAR stage insights."""
    analytics = test_setup["analytics_srv"]
    farm_srv = test_setup["farm_srv"]
    plot_srv = test_setup["plot_srv"]
    cycle_srv = test_setup["cycle_srv"]
    calendar_srv = test_setup["calendar_srv"]
    repo = test_setup["repo"]
    fa = test_setup["farmer_a"]

    # Create Farm 1 (5.0 acres)
    f1 = farm_srv.create_farm(fa.farmer_id, FarmCreate(
        farm_name="Karnal Green Fields",
        state="Haryana",
        district="Karnal",
        total_area=5.0,
        area_unit=AreaUnit.ACRE,
        ownership_type=OwnershipType.OWNED,
        latitude=29.6857,
        longitude=76.9905,
    ))

    # Create Farm 2 (3.5 acres)
    f2 = farm_srv.create_farm(fa.farmer_id, FarmCreate(
        farm_name="Karnal Orchard",
        state="Haryana",
        district="Karnal",
        total_area=3.5,
        area_unit=AreaUnit.ACRE,
        ownership_type=OwnershipType.LEASED,
        latitude=29.6900,
        longitude=77.0000,
    ))

    # Add plot and active wheat crop cycle (sown 21 days ago -> Crown Root Initiation)
    plot1 = plot_srv.create_plot(
        farm_id=f1.farm_id,
        farmer_id=fa.farmer_id,
        plot_data=PlotCreate(
            plot_name="Plot Alpha",
            area=3.0,
            area_unit=AreaUnit.ACRE,
        ),
    )

    today = date.today()
    sowing_date_21d = today - timedelta(days=21)
    cycle = cycle_srv.create_crop_cycle(
        plot_id=plot1.plot_id,
        farmer_id=fa.farmer_id,
        cycle_data=CropCycleCreate(
            crop_name="Wheat",
            variety_name="HD-2967",
            season=CropSeason.RABI,
            sowing_date=sowing_date_21d.isoformat(),
            target_yield_quintals=22.0,
        ),
    )

    # Schedule tasks
    calendar_srv.create_activity(
        farm_id=f1.farm_id,
        farmer_id=fa.farmer_id,
        activity_data=FarmActivityCreate(
            activity_title="Apply CRI First Irrigation",
            activity_type=ActivityType.IRRIGATION,
            scheduled_date=today.isoformat(),
        ),
    )

    # Record an advisory case
    repo.create_advisory_case(
        farmer_name=fa.full_name,
        crop="Wheat",
        location="Karnal, Haryana",
        farmer_id=fa.farmer_id,
        farm_id=f1.farm_id,
        state="Haryana",
        district="Karnal",
        symptoms="Yellow striping observed on lower leaf blades",
        risk="High",
        field_visit_required=True,
    )

    # Fetch overview
    overview = analytics.get_farmer_overview(farmer_id=fa.farmer_id, farm_id=f1.farm_id)

    # Assert metrics
    assert overview.metrics["farmer_active_farms"].value == 2
    assert overview.metrics["farmer_total_acres"].value == 8.5
    assert overview.metrics["farmer_active_crops"].value == 1
    assert overview.metrics["farmer_due_tasks"].value == 1
    assert overview.metrics["farmer_crop_health_cases"].value == 1

    # Assert active crop stages
    assert len(overview.active_crop_stages) == 1
    stage_data = overview.active_crop_stages[0]
    assert stage_data["crop_name"] == "Wheat"
    assert stage_data["days_since_sowing"] == 21
    assert stage_data["current_stage"] in (
        CropGrowthStage.SOWING_GERMINATION.value,
        CropGrowthStage.SEEDLING_EMERGENCE.value,
    )

    # Assert insights include ICAR CRI irrigation warning
    insights = overview.insights
    cri_insight = next((ins for ins in insights if "CRI" in ins.title or "Crown Root" in ins.title), None)
    assert cri_insight is not None
    assert cri_insight.severity.value == "recommendation"
    assert "yield loss" in cri_insight.description.lower() or "irrigation" in cri_insight.suggested_action.lower()


def test_farmer_dashboard_tenant_isolation(test_setup):
    """Verify Farmer A cannot view Farmer B's farm metrics."""
    analytics = test_setup["analytics_srv"]
    farm_srv = test_setup["farm_srv"]
    fa = test_setup["farmer_a"]
    fb = test_setup["farmer_b"]

    # Create farm for Farmer B
    farm_b = farm_srv.create_farm(fb.farmer_id, FarmCreate(
        farm_name="Farmer B Secret Plot",
        state="Punjab",
        district="Ludhiana",
        total_area=10.0,
        area_unit=AreaUnit.ACRE,
    ))

    # Farmer A attempts to access Farmer B's farm
    with pytest.raises(PermissionError) as exc_info:
        analytics.get_farmer_overview(farmer_id=fa.farmer_id, farm_id=farm_b.farm_id)

    assert "Access denied" in str(exc_info.value)
