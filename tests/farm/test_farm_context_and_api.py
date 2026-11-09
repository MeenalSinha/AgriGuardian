"""Integration tests for FarmContextService and Phase 6 FastAPI REST Endpoints."""

import os
import tempfile
import pytest

from backend.app.main import (
    activate_farm,
    create_crop_cycle,
    create_farm,
    create_plot,
    get_current_user_profile,
    get_resolved_farm_context,
    list_farmer_farms,
    login_farmer,
    register_farmer,
)
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    CropCycleCreate,
    CropSeason,
    FarmCreate,
    PlotCreate,
    UserLoginRequest,
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
def context_fixture():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    repo = FarmDatabaseRepository(db_path=temp_db.name)

    auth_srv = AuthService(repo=repo)
    profile_srv = FarmerProfileService(repo=repo)
    farm_srv = FarmService(repo=repo)
    plot_srv = PlotService(repo=repo)
    cycle_srv = CropCycleService(repo=repo)
    soil_srv = SoilIrrigationService(repo=repo)
    cal_srv = CropCalendarService(repo=repo)
    context_srv = FarmContextService(
        repo=repo,
        profile_service=profile_srv,
        farm_service=farm_srv,
        plot_service=plot_srv,
        cycle_service=cycle_srv,
        soil_service=soil_srv,
        calendar_service=cal_srv,
    )

    _, farmer, token = auth_srv.register_farmer(UserRegistrationRequest(
        username="context_farmer",
        password="Password123",
        full_name="Rajesh Verma",
        preferred_language="hi-IN",
    ))

    yield {
        "repo": repo,
        "auth_srv": auth_srv,
        "farm_srv": farm_srv,
        "plot_srv": plot_srv,
        "cycle_srv": cycle_srv,
        "context_srv": context_srv,
        "farmer": farmer,
        "token": token,
    }

    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_unauthenticated_and_ambiguous_context(context_fixture):
    """Verify unauthenticated context and multi-farm ambiguity flagging."""
    f = context_fixture
    ctx_srv = f["context_srv"]
    farm_srv = f["farm_srv"]
    farmer_id = f["farmer"].farmer_id

    # 1. Unauthenticated context
    guest_ctx = ctx_srv.resolve_context(farmer_id=None)
    assert guest_ctx.is_authenticated is False
    assert guest_ctx.active_farm is None

    # 2. Add two farms, neither set active
    f1 = farm_srv.create_farm(farmer_id, FarmCreate(farm_name="Farm 1", state="Punjab", district="Khanna", total_area=4.0, is_active_farm=False))
    f2 = farm_srv.create_farm(farmer_id, FarmCreate(farm_name="Farm 2", state="Punjab", district="Ludhiana", total_area=6.0, is_active_farm=False))

    # De-activate f1 (which auto-activated as first)
    with f["repo"].get_connection() as conn:
        conn.execute("UPDATE farms SET is_active_farm = 0 WHERE farmer_id = ?", (farmer_id,))

    ambiguous_ctx = ctx_srv.resolve_context(farmer_id=farmer_id)
    assert ambiguous_ctx.is_authenticated is True
    assert ambiguous_ctx.is_ambiguous is True
    assert "no active farm has been designated" in ambiguous_ctx.ambiguity_reason


def test_weather_and_market_context_extraction(context_fixture):
    """Verify farm context correctly informs weather coordinates and market location queries."""
    f = context_fixture
    ctx_srv = f["context_srv"]
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    cycle_srv = f["cycle_srv"]
    farmer_id = f["farmer"].farmer_id

    # Farm with auto-resolved coordinates
    farm = farm_srv.create_farm(farmer_id, FarmCreate(
        farm_name="Model Farm",
        state="Uttar Pradesh",
        district="Meerut",
        total_area=5.0,
        is_active_farm=True,
    ))
    plot = plot_srv.create_plot(farm.farm_id, farmer_id, PlotCreate(plot_name="Plot 1", area=5.0))
    cycle = cycle_srv.create_crop_cycle(plot.plot_id, farmer_id, CropCycleCreate(
        crop_name="Wheat",
        season=CropSeason.RABI,
        sowing_date="2026-11-01",
    ))

    ctx = ctx_srv.resolve_context(farmer_id=farmer_id)
    assert ctx.has_active_farm is True
    assert ctx.has_active_crop is True

    # Weather coordinates extraction
    weather_loc = ctx_srv.get_weather_coordinates(ctx)
    assert weather_loc is not None
    lat, lon, label = weather_loc
    assert lat == pytest.approx(28.98, abs=0.1)
    assert "Meerut" in label

    # Market location query extraction
    mkt_loc = ctx_srv.get_market_location_query(ctx)
    assert mkt_loc is not None
    assert mkt_loc["state"] == "Uttar Pradesh"
    assert mkt_loc["district"] == "Meerut"

    # Agronomic payload
    agro_payload = ctx_srv.get_agronomic_context_payload(ctx)
    assert agro_payload["is_personalized"] is True
    assert agro_payload["crop"] == "Wheat"
    assert agro_payload["farm_name"] == "Model Farm"


def test_fastapi_phase6_route_direct_invocation():
    """Verify FastAPI route functions directly (sub-second in Python 3.14)."""
    # 1. Register new farmer
    username = f"api_test_farmer_{os.getpid()}"
    reg_resp = register_farmer(UserRegistrationRequest(
        username=username,
        password="SecurePassword123",
        full_name="FastAPI Test Farmer",
        phone_number="9876543299",
        preferred_language="hi-IN",
    ))
    assert reg_resp.username == username
    assert reg_resp.access_token is not None

    # 2. Login farmer
    login_resp = login_farmer(UserLoginRequest(
        username=username,
        password="SecurePassword123",
    ))
    assert login_resp.farmer_id == reg_resp.farmer_id

    # 3. Create Farm
    auth_tuple = (None, reg_resp)
    # Wrap in dummy user/profile for direct call
    from backend.app.main import auth_service
    user, profile = auth_service.get_current_farmer(reg_resp.access_token)
    auth_ctx = (user, profile)

    farm = create_farm(
        req=FarmCreate(
            farm_name="REST Farm Alpha",
            state="Punjab",
            district="Khanna",
            total_area=15.0,
        ),
        auth=auth_ctx,
    )
    assert farm.farm_name == "REST Farm Alpha"
    assert farm.farmer_id == profile.farmer_id

    # 4. List Farms
    farms = list_farmer_farms(auth=auth_ctx)
    assert len(farms) >= 1

    # 5. Create Plot
    plot = create_plot(
        farm_id=farm.farm_id,
        req=PlotCreate(
            plot_name="REST Plot 1",
            area=5.0,
        ),
        auth=auth_ctx,
    )
    assert plot.plot_name == "REST Plot 1"

    # 6. Create Crop Cycle
    cycle = create_crop_cycle(
        plot_id=plot.plot_id,
        req=CropCycleCreate(
            crop_name="Wheat",
            season=CropSeason.RABI,
            sowing_date="2026-11-01",
        ),
        auth=auth_ctx,
    )
    assert cycle.crop_name == "Wheat"

    # 7. Resolve Farm Context via API
    ctx = get_resolved_farm_context(auth=auth_ctx)
    assert ctx.is_authenticated is True
    assert ctx.active_farm.farm_id == farm.farm_id
    assert ctx.current_cycle.cycle_id == cycle.cycle_id
