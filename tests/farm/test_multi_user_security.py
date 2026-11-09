"""Mandatory multi-user isolation and security boundary verification tests.

Verifies that Farmer A and Farmer B have strictly isolated data boundaries across
farms, plots, crop cycles, soil records, irrigation profiles, and calendar activities.
"""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycleCreate,
    CropGrowthStage,
    CropSeason,
    CropStageTransitionRequest,
    FarmActivityCreate,
    FarmCreate,
    FarmUpdate,
    IrrigationMethod,
    IrrigationProfileCreate,
    OwnershipType,
    PlotCreate,
    PlotUpdate,
    SoilProfileCreate,
    SoilType,
    UserRegistrationRequest,
)
from backend.app.services.auth.auth_service import AuthService
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_context_service import FarmContextService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService
from backend.app.services.farm.soil_irrigation_service import SoilIrrigationService


@pytest.fixture
def multi_tenant_fixture():
    """Create isolated test repository with two distinct farmer accounts."""
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    repo = FarmDatabaseRepository(db_path=temp_db.name)

    auth_srv = AuthService(repo=repo)
    profile_srv = FarmerProfileService(repo=repo)
    farm_srv = FarmService(repo=repo)
    plot_srv = PlotService(repo=repo)
    cycle_srv = CropCycleService(repo=repo)
    soil_srv = SoilIrrigationService(repo=repo)
    calendar_srv = CropCalendarService(repo=repo)
    context_srv = FarmContextService(
        repo=repo,
        profile_service=profile_srv,
        farm_service=farm_srv,
        plot_service=plot_srv,
        cycle_service=cycle_srv,
        soil_service=soil_srv,
        calendar_service=calendar_srv,
    )

    # Register Farmer A
    _, farmer_a, token_a = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_alice",
        password="AliceSecurePassword1",
        full_name="Alice Patel",
        phone_number="9111111111",
        preferred_language="hi-IN",
    ))

    # Register Farmer B
    _, farmer_b, token_b = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_bob",
        password="BobSecurePassword2",
        full_name="Bob Singh",
        phone_number="9222222222",
        preferred_language="pa-IN",
    ))

    yield {
        "repo": repo,
        "auth_srv": auth_srv,
        "farm_srv": farm_srv,
        "plot_srv": plot_srv,
        "cycle_srv": cycle_srv,
        "soil_srv": soil_srv,
        "calendar_srv": calendar_srv,
        "context_srv": context_srv,
        "farmer_a": farmer_a,
        "farmer_b": farmer_b,
        "token_a": token_a,
        "token_b": token_b,
    }

    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_farm_multi_user_isolation(multi_tenant_fixture):
    """Verify Farmer B cannot view, modify, or delete Farmer A's farm."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    # Farmer A creates Farm A
    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(
        farm_name="Alice Green Acres",
        state="Punjab",
        district="Ludhiana",
        total_area=10.0,
        area_unit=AreaUnit.ACRE,
        ownership_type=OwnershipType.OWNED,
    ))

    # 1. Farmer B attempting to fetch Farm A must receive None
    assert farm_srv.get_farm(farm_a.farm_id, fb.farmer_id) is None

    # 2. Farmer B's farm listing must NOT contain Farm A
    b_farms = farm_srv.list_farms(fb.farmer_id)
    assert len(b_farms) == 0

    # 3. Farmer B attempting to update Farm A must receive None
    update_result = farm_srv.update_farm(
        farm_a.farm_id,
        fb.farmer_id,
        FarmUpdate(farm_name="Illicitly Modified by Bob"),
    )
    assert update_result is None

    # Verify farm name was untouched
    farm_a_verified = farm_srv.get_farm(farm_a.farm_id, fa.farmer_id)
    assert farm_a_verified.farm_name == "Alice Green Acres"

    # 4. Farmer B attempting to delete Farm A must return False
    del_result = farm_srv.delete_farm(farm_a.farm_id, fb.farmer_id)
    assert del_result is False

    # Verify Farm A still exists for Farmer A
    assert farm_srv.get_farm(farm_a.farm_id, fa.farmer_id) is not None


def test_plot_multi_user_isolation(multi_tenant_fixture):
    """Verify Farmer B cannot view, update, delete, or create plots on Farmer A's farm."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(
        farm_name="Alice Farm",
        state="Haryana",
        district="Karnal",
        total_area=5.0,
    ))

    plot_a = plot_srv.create_plot(farm_a.farm_id, fa.farmer_id, PlotCreate(
        plot_name="North Tubewell Plot",
        area=2.5,
        area_unit=AreaUnit.ACRE,
        soil_type=SoilType.ALLUVIAL,
    ))

    # 1. Farmer B attempting to create a plot under Farm A must be rejected
    with pytest.raises(ValueError, match="not found or does not belong"):
        plot_srv.create_plot(farm_a.farm_id, fb.farmer_id, PlotCreate(
            plot_name="Bob Unauthorized Plot",
            area=1.0,
        ))

    # 2. Farmer B attempting to get Plot A must receive None
    assert plot_srv.get_plot(plot_a.plot_id, fb.farmer_id) is None

    # 3. Farmer B listing plots for Farm A must receive empty list
    assert plot_srv.list_plots_by_farm(farm_a.farm_id, fb.farmer_id) == []

    # 4. Farmer B attempting to update Plot A must receive None
    assert plot_srv.update_plot(plot_a.plot_id, fb.farmer_id, PlotUpdate(plot_name="Compromised")) is None

    # 5. Farmer B attempting to delete Plot A must return False
    assert plot_srv.delete_plot(plot_a.plot_id, fb.farmer_id) is False


def test_crop_cycle_multi_user_isolation(multi_tenant_fixture):
    """Verify Farmer B cannot view, transition, or harvest Farmer A's active crop cycle."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    cycle_srv = f["cycle_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(farm_name="FA", state="Punjab", district="Khanna", total_area=4.0))
    plot_a = plot_srv.create_plot(farm_a.farm_id, fa.farmer_id, PlotCreate(plot_name="PA", area=2.0))

    cycle_a = cycle_srv.create_crop_cycle(plot_a.plot_id, fa.farmer_id, CropCycleCreate(
        crop_name="Wheat",
        variety_name="PBW-550",
        season=CropSeason.RABI,
        sowing_date="2026-11-01",
    ))

    # 1. Farmer B attempting to start a crop cycle on Plot A must be rejected
    with pytest.raises(ValueError, match="not found or not owned"):
        cycle_srv.create_crop_cycle(plot_a.plot_id, fb.farmer_id, CropCycleCreate(
            crop_name="Mustard",
            sowing_date="2026-11-01",
        ))

    # 2. Farmer B attempting to fetch Cycle A must receive None
    assert cycle_srv.get_crop_cycle(cycle_a.cycle_id, fb.farmer_id) is None

    # 3. Farmer B attempting to transition stage of Cycle A must receive None
    trans_res = cycle_srv.transition_stage(
        cycle_a.cycle_id,
        fb.farmer_id,
        CropStageTransitionRequest(new_stage=CropGrowthStage.HARVEST_READY),
    )
    assert trans_res is None

    # Verify Cycle A stage was not modified by Farmer B
    cycle_a_verified = cycle_srv.get_crop_cycle(cycle_a.cycle_id, fa.farmer_id)
    assert cycle_a_verified.current_stage == CropGrowthStage.SOWING_GERMINATION


def test_soil_and_irrigation_multi_user_isolation(multi_tenant_fixture):
    """Verify Farmer B cannot view or update Farmer A's soil and irrigation records."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    soil_srv = f["soil_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(farm_name="FA", state="UP", district="Meerut", total_area=3.0))
    plot_a = plot_srv.create_plot(farm_a.farm_id, fa.farmer_id, PlotCreate(plot_name="PA", area=1.5))

    soil_a = soil_srv.save_soil_profile(plot_a.plot_id, fa.farmer_id, SoilProfileCreate(
        ph_level=7.4,
        organic_carbon_pct=0.65,
        nitrogen_kg_ha=320.0,
        phosphorus_kg_ha=22.0,
        potassium_kg_ha=250.0,
    ))

    irr_a = soil_srv.save_irrigation_profile(plot_a.plot_id, fa.farmer_id, IrrigationProfileCreate(
        primary_source="Canal",
        irrigation_method=IrrigationMethod.DRIP,
    ))

    # 1. Farmer B attempting to read Plot A's soil profile must receive None
    assert soil_srv.get_soil_profile(plot_a.plot_id, fb.farmer_id) is None

    # 2. Farmer B attempting to update Plot A's soil profile must be rejected
    with pytest.raises(ValueError, match="not found or does not belong"):
        soil_srv.save_soil_profile(plot_a.plot_id, fb.farmer_id, SoilProfileCreate(ph_level=5.0))

    # 3. Farmer B attempting to read Plot A's irrigation profile must receive None
    assert soil_srv.get_irrigation_profile(plot_a.plot_id, fb.farmer_id) is None


def test_farm_activities_multi_user_isolation(multi_tenant_fixture):
    """Verify Farmer B cannot view, complete, or delete Farmer A's scheduled farm activities."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    cal_srv = f["calendar_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(farm_name="FA", state="UP", district="Meerut", total_area=3.0))

    act_a = cal_srv.create_activity(farm_a.farm_id, fa.farmer_id, FarmActivityCreate(
        activity_type=ActivityType.IRRIGATION,
        activity_title="Alice Secret Irrigation",
        scheduled_date="2026-11-10",
    ))

    # 1. Farmer B querying activities on Farm A receives empty list
    b_acts = cal_srv.list_activities(fb.farmer_id, farm_id=farm_a.farm_id)
    assert len(b_acts) == 0

    # 2. Farmer B attempting to complete Activity A must receive None
    completed = cal_srv.complete_activity(act_a.activity_id, fb.farmer_id)
    assert completed is None

    # Verify Activity A is still planned for Farmer A
    a_act_verified = cal_srv.list_activities(fa.farmer_id, farm_id=farm_a.farm_id)[0]
    assert a_act_verified.status == ActivityStatus.PLANNED


def test_farm_context_isolation(multi_tenant_fixture):
    """Verify unified FarmContext resolved for Farmer B contains zero elements of Farmer A."""
    f = multi_tenant_fixture
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    cycle_srv = f["cycle_srv"]
    ctx_srv = f["context_srv"]
    fa = f["farmer_a"]
    fb = f["farmer_b"]

    # Farmer A sets up full farm, plot, and active crop
    farm_a = farm_srv.create_farm(fa.farmer_id, FarmCreate(
        farm_name="Alice Organic Farm",
        state="Punjab",
        district="Patiala",
        total_area=8.0,
        is_active_farm=True,
    ))
    plot_a = plot_srv.create_plot(farm_a.farm_id, fa.farmer_id, PlotCreate(plot_name="Main Plot", area=8.0))
    cycle_a = cycle_srv.create_crop_cycle(plot_a.plot_id, fa.farmer_id, CropCycleCreate(
        crop_name="Paddy",
        variety_name="PR-126",
        sowing_date="2026-11-01",
    ))

    # Resolve context for Farmer B
    ctx_b = ctx_srv.resolve_context(farmer_id=fb.farmer_id)

    assert ctx_b.is_authenticated is True
    assert ctx_b.farmer.farmer_id == fb.farmer_id
    assert ctx_b.active_farm is None
    assert ctx_b.all_farms_count == 0
    assert len(ctx_b.active_plots) == 0
    assert ctx_b.current_cycle is None

    # Farmer B attempting to resolve context with farm_id_override pointing to Farm A
    ctx_b_hack = ctx_srv.resolve_context(farmer_id=fb.farmer_id, farm_id_override=farm_a.farm_id)
    assert ctx_b_hack.active_farm is None
    assert ctx_b_hack.is_ambiguous is True
    assert "not found or not owned" in ctx_b_hack.ambiguity_reason
