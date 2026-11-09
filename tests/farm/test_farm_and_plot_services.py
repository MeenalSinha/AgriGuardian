"""Unit tests for FarmService, PlotService, and FarmerProfileService."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository, normalize_area_to_acres
from backend.app.schemas.farm import (
    AreaUnit,
    CropCycleCreate,
    CropSeason,
    FarmCreate,
    FarmerProfileUpdate,
    FarmUpdate,
    OwnershipType,
    PlotCreate,
    PlotUpdate,
    SoilType,
    UserRegistrationRequest,
)
from backend.app.services.auth.auth_service import AuthService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService


@pytest.fixture
def farm_fixture():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    repo = FarmDatabaseRepository(db_path=temp_db.name)

    auth_srv = AuthService(repo=repo)
    profile_srv = FarmerProfileService(repo=repo)
    farm_srv = FarmService(repo=repo)
    plot_srv = PlotService(repo=repo)
    cycle_srv = CropCycleService(repo=repo)

    _, farmer, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="suresh_farmer",
        password="Password123",
        full_name="Suresh Kumar",
        phone_number="9876543210",
        preferred_language="hi-IN",
    ))

    yield {
        "repo": repo,
        "profile_srv": profile_srv,
        "farm_srv": farm_srv,
        "plot_srv": plot_srv,
        "cycle_srv": cycle_srv,
        "farmer": farmer,
    }

    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_area_normalization_to_acres():
    """Verify conversion of regional units to standard acres."""
    assert normalize_area_to_acres(10.0, AreaUnit.ACRE) == 10.0
    assert normalize_area_to_acres(2.0, AreaUnit.HECTARE) == pytest.approx(4.9421, abs=0.01)
    assert normalize_area_to_acres(8.0, AreaUnit.BIGHA) == pytest.approx(5.0, abs=0.01)
    assert normalize_area_to_acres(40.0, AreaUnit.GUNTHA) == pytest.approx(1.0, abs=0.01)


def test_multi_farm_management(farm_fixture):
    """Verify creating multiple farms, area normalization, and active farm switching."""
    f = farm_fixture
    farm_srv = f["farm_srv"]
    farmer_id = f["farmer"].farmer_id

    # Create First Farm (should auto-activate)
    farm1 = farm_srv.create_farm(farmer_id, FarmCreate(
        farm_name="East Ridge Farm",
        state="Uttar Pradesh",
        district="Meerut",
        total_area=4.0,
        area_unit=AreaUnit.HECTARE,
        ownership_type=OwnershipType.OWNED,
    ))
    assert farm1.is_active_farm is True
    assert farm1.total_area_acres == pytest.approx(9.884, abs=0.05)
    # Verify auto-resolved coordinates for Meerut
    assert farm1.latitude is not None and farm1.longitude is not None
    assert farm1.latitude == pytest.approx(28.98, abs=0.1)

    # Create Second Farm (should not be active by default)
    farm2 = farm_srv.create_farm(farmer_id, FarmCreate(
        farm_name="Canal View Farm",
        state="Punjab",
        district="Ludhiana",
        total_area=12.0,
        area_unit=AreaUnit.BIGHA,
        ownership_type=OwnershipType.LEASED,
        is_active_farm=False,
    ))
    assert farm2.is_active_farm is False
    assert farm2.total_area_acres == pytest.approx(7.5, abs=0.1)

    # Verify active farm query
    active = farm_srv.get_active_farm(farmer_id)
    assert active.farm_id == farm1.farm_id

    # Switch active farm to farm2
    activated = farm_srv.set_active_farm(farm2.farm_id, farmer_id)
    assert activated.is_active_farm is True

    # Verify farm1 is no longer active
    farm1_refreshed = farm_srv.get_farm(farm1.farm_id, farmer_id)
    assert farm1_refreshed.is_active_farm is False


def test_plot_management_and_validation(farm_fixture):
    """Verify plot creation, area calculation, and listing."""
    f = farm_fixture
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    farmer_id = f["farmer"].farmer_id

    farm = farm_srv.create_farm(farmer_id, FarmCreate(
        farm_name="Central Farm",
        state="Maharashtra",
        district="Nashik",
        total_area=6.0,
    ))

    plot1 = plot_srv.create_plot(farm.farm_id, farmer_id, PlotCreate(
        plot_name="Grape Orchard Block A",
        area=3.0,
        area_unit=AreaUnit.ACRE,
        soil_type=SoilType.BLACK_COTTON,
        irrigation_source="Drip Well",
    ))
    assert plot1.plot_id.startswith("plt_")
    assert plot1.area_acres == 3.0
    assert plot1.soil_type == SoilType.BLACK_COTTON

    plot2 = plot_srv.create_plot(farm.farm_id, farmer_id, PlotCreate(
        plot_name="Vegetable Nursery Block",
        area=1.0,
        area_unit=AreaUnit.HECTARE,
        soil_type=SoilType.RED_LOAM,
    ))
    assert plot2.area_acres == pytest.approx(2.47, abs=0.05)

    plots = plot_srv.list_plots_by_farm(farm.farm_id, farmer_id)
    assert len(plots) == 2


def test_progressive_profile_completion_scoring(farm_fixture):
    """Verify progressive onboarding percentage dynamically reflects farm, plot, and crop additions."""
    f = farm_fixture
    profile_srv = f["profile_srv"]
    farm_srv = f["farm_srv"]
    plot_srv = f["plot_srv"]
    cycle_srv = f["cycle_srv"]
    farmer_id = f["farmer"].farmer_id

    # Initial profile completion
    p0 = profile_srv.get_profile(farmer_id)
    initial_score = p0.profile_completion_pct
    assert initial_score >= 25.0

    # Step 1: Add farm (+20%)
    farm = farm_srv.create_farm(farmer_id, FarmCreate(farm_name="F1", state="UP", district="Meerut", total_area=5.0))
    p1 = profile_srv.get_profile(farmer_id)
    assert p1.profile_completion_pct > initial_score

    # Step 2: Add plot (+20%)
    plot = plot_srv.create_plot(farm.farm_id, farmer_id, PlotCreate(plot_name="P1", area=2.0))
    p2 = profile_srv.get_profile(farmer_id)
    assert p2.profile_completion_pct > p1.profile_completion_pct

    # Step 3: Add active crop cycle (+20%)
    cycle_srv.create_crop_cycle(plot.plot_id, farmer_id, CropCycleCreate(
        crop_name="Wheat",
        season=CropSeason.RABI,
        sowing_date="2026-11-01",
    ))
    p3 = profile_srv.get_profile(farmer_id)
    assert p3.profile_completion_pct > p2.profile_completion_pct
    assert p3.profile_completion_pct >= 80.0
