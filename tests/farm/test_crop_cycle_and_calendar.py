"""Unit tests for CropCycleService, SoilIrrigationService, and CropCalendarService."""

import os
import tempfile
from datetime import datetime, timedelta
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycleCreate,
    CropCycleStatus,
    CropGrowthStage,
    CropSeason,
    CropStageTransitionRequest,
    FarmActivityCreate,
    FarmCreate,
    IrrigationMethod,
    IrrigationProfileCreate,
    PlotCreate,
    PumpingSource,
    SoilProfileCreate,
    SoilType,
    UserRegistrationRequest,
    WaterAvailability,
)
from backend.app.services.auth.auth_service import AuthService
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.plot_service import PlotService
from backend.app.services.farm.soil_irrigation_service import SoilIrrigationService


@pytest.fixture
def cycle_calendar_fixture():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    repo = FarmDatabaseRepository(db_path=temp_db.name)

    auth_srv = AuthService(repo=repo)
    farm_srv = FarmService(repo=repo)
    plot_srv = PlotService(repo=repo)
    cycle_srv = CropCycleService(repo=repo)
    soil_srv = SoilIrrigationService(repo=repo)
    cal_srv = CropCalendarService(repo=repo)

    _, farmer, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="kuldeep_farmer",
        password="Password123",
        full_name="Kuldeep Yadav",
    ))

    farm = farm_srv.create_farm(farmer.farmer_id, FarmCreate(
        farm_name="Main Farm",
        state="Haryana",
        district="Karnal",
        total_area=10.0,
    ))

    plot = plot_srv.create_plot(farm.farm_id, farmer.farmer_id, PlotCreate(
        plot_name="Plot A",
        area=5.0,
    ))

    yield {
        "repo": repo,
        "cycle_srv": cycle_srv,
        "soil_srv": soil_srv,
        "cal_srv": cal_srv,
        "farmer": farmer,
        "farm": farm,
        "plot": plot,
    }

    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_crop_stage_estimation(cycle_calendar_fixture):
    """Verify empirical growth stage estimator based on days elapsed since sowing."""
    f = cycle_calendar_fixture
    cycle_srv = f["cycle_srv"]

    today = datetime.now().date()

    # Sown 5 days ago -> SOWING_GERMINATION
    d5 = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    stg, days, _ = cycle_srv.estimate_growth_stage("Wheat", d5)
    assert days == 5
    assert stg == CropGrowthStage.SOWING_GERMINATION

    # Sown 21 days ago -> SEEDLING_EMERGENCE (CRI stage)
    d21 = (today - timedelta(days=21)).strftime("%Y-%m-%d")
    stg, days, _ = cycle_srv.estimate_growth_stage("Wheat", d21)
    assert days == 21
    assert stg == CropGrowthStage.SEEDLING_EMERGENCE

    # Sown 50 days ago -> VEGETATIVE_TILLERING
    d50 = (today - timedelta(days=50)).strftime("%Y-%m-%d")
    stg, days, _ = cycle_srv.estimate_growth_stage("Wheat", d50)
    assert days == 50
    assert stg == CropGrowthStage.VEGETATIVE_TILLERING

    # Sown 80 days ago -> FLOWERING_REPRODUCTIVE
    d80 = (today - timedelta(days=80)).strftime("%Y-%m-%d")
    stg, days, _ = cycle_srv.estimate_growth_stage("Wheat", d80)
    assert days == 80
    assert stg == CropGrowthStage.FLOWERING_REPRODUCTIVE


def test_crop_cycle_lifecycle_and_historical_preservation(cycle_calendar_fixture):
    """Verify stage transitions, farmer confirmation, and harvest completion preserving history."""
    f = cycle_calendar_fixture
    cycle_srv = f["cycle_srv"]
    plot = f["plot"]
    farmer = f["farmer"]

    cycle = cycle_srv.create_crop_cycle(plot.plot_id, farmer.farmer_id, CropCycleCreate(
        crop_name="Wheat",
        variety_name="HD-2967",
        season=CropSeason.RABI,
        sowing_date="2026-10-15",
        target_yield_quintals=24.0,
    ))

    assert cycle.cycle_id.startswith("cyc_")
    assert cycle.status == CropCycleStatus.ACTIVE
    assert cycle.stage_confirmed_by_farmer is False

    # Farmer advances and confirms stage
    updated = cycle_srv.transition_stage(
        cycle.cycle_id,
        farmer.farmer_id,
        CropStageTransitionRequest(
            new_stage=CropGrowthStage.FLOWERING_REPRODUCTIVE,
            notes="Flag leaves healthy, heading observed across 60% of plot.",
        ),
    )
    assert updated.current_stage == CropGrowthStage.FLOWERING_REPRODUCTIVE
    assert updated.stage_confirmed_by_farmer is True

    # Complete harvest
    harvested = cycle_srv.complete_harvest(
        cycle.cycle_id,
        farmer.farmer_id,
        harvest_date="2026-11-09",
        actual_yield_quintals=23.5,
        notes="Bumper harvest, sold directly to FCI mandi.",
    )
    assert harvested.status == CropCycleStatus.COMPLETED
    assert harvested.current_stage == CropGrowthStage.HARVESTED
    assert harvested.actual_yield_quintals == 23.5

    # Verify completed cycle is preserved in history
    history = cycle_srv.list_crop_cycles_by_farmer(farmer.farmer_id, status="completed")
    assert len(history) == 1
    assert history[0].crop_name == "Wheat"
    assert history[0].actual_yield_quintals == 23.5

    # Verify plot is now free to start a new cycle
    new_cycle = cycle_srv.create_crop_cycle(plot.plot_id, farmer.farmer_id, CropCycleCreate(
        crop_name="Tomato",
        season=CropSeason.ZAID,
        sowing_date="2026-11-10",
    ))
    assert new_cycle.status == CropCycleStatus.ACTIVE
    assert new_cycle.crop_name == "Tomato"


def test_soil_health_and_icar_diagnostics(cycle_calendar_fixture):
    """Verify soil fertility benchmarks and ICAR diagnostic categories."""
    f = cycle_calendar_fixture
    soil_srv = f["soil_srv"]
    plot = f["plot"]
    farmer = f["farmer"]

    # Acidic and nutrient-deficient soil test
    soil_low = soil_srv.save_soil_profile(plot.plot_id, farmer.farmer_id, SoilProfileCreate(
        soil_type=SoilType.RED_LOAM,
        ph_level=5.5,
        organic_carbon_pct=0.35,
        nitrogen_kg_ha=190.0,
        phosphorus_kg_ha=8.0,
        potassium_kg_ha=95.0,
        is_lab_tested=True,
        soil_health_card_number="SHC-2026-HR-8841",
    ))

    diag_low = soil_srv.diagnose_soil_fertility(soil_low)
    assert diag_low["ph_category"] == "Strongly Acidic"
    assert "Low" in diag_low["organic_carbon_category"]
    assert "Low" in diag_low["nitrogen_category"]
    assert "Low" in diag_low["phosphorus_category"]
    assert "Low" in diag_low["potassium_category"]
    assert len(diag_low["recommendations"]) >= 4

    # Optimal / fertile soil test
    soil_opt = soil_srv.save_soil_profile(plot.plot_id, farmer.farmer_id, SoilProfileCreate(
        soil_type=SoilType.ALLUVIAL,
        ph_level=7.2,
        organic_carbon_pct=0.85,
        nitrogen_kg_ha=380.0,
        phosphorus_kg_ha=20.0,
        potassium_kg_ha=230.0,
        is_lab_tested=True,
    ))
    diag_opt = soil_srv.diagnose_soil_fertility(soil_opt)
    assert "Neutral" in diag_opt["ph_category"]
    assert "High" in diag_opt["organic_carbon_category"]
    assert "Medium" in diag_opt["nitrogen_category"]
    assert "Medium" in diag_opt["phosphorus_category"]
    assert "Medium" in diag_opt["potassium_category"]


def test_crop_calendar_and_stage_tasks(cycle_calendar_fixture):
    """Verify task creation, due date updating, and stage-grounded task generation."""
    f = cycle_calendar_fixture
    cal_srv = f["cal_srv"]
    cycle_srv = f["cycle_srv"]
    farm = f["farm"]
    plot = f["plot"]
    farmer = f["farmer"]

    # 1. Custom task due today
    today_str = datetime.now().strftime("%Y-%m-%d")
    task_due = cal_srv.create_activity(farm.farm_id, farmer.farmer_id, FarmActivityCreate(
        activity_type=ActivityType.IRRIGATION,
        activity_title="Urgent Morning Irrigation",
        scheduled_date=today_str,
    ))
    assert task_due.status == ActivityStatus.DUE

    # 2. Complete task
    completed = cal_srv.complete_activity(
        task_due.activity_id,
        farmer.farmer_id,
        cost_incurred=450.0,
        notes="Electricity pump ran for 4 hours.",
    )
    assert completed.status == ActivityStatus.COMPLETED
    assert completed.cost_incurred == 450.0

    # 3. Grounded stage recommendations for wheat at CRI stage
    cycle = cycle_srv.create_crop_cycle(plot.plot_id, farmer.farmer_id, CropCycleCreate(
        crop_name="Wheat",
        season=CropSeason.RABI,
        sowing_date="2026-10-20",
    ))
    # Transition to CRI stage
    cycle_srv.transition_stage(cycle.cycle_id, farmer.farmer_id, CropStageTransitionRequest(
        new_stage=CropGrowthStage.SEEDLING_EMERGENCE,
    ))
    cycle_refreshed = cycle_srv.get_crop_cycle(cycle.cycle_id, farmer.farmer_id)

    # Generate tasks
    tasks = cal_srv.generate_grounded_stage_activities(cycle_refreshed, farmer.farmer_id)
    assert len(tasks) >= 2
    task_titles = [t.activity_title.lower() for t in tasks]
    assert any("cri" in title or "crown root" in title for title in task_titles)
    assert any("urea" in title or "top dressing" in title for title in task_titles)

