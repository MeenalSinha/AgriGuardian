"""Unit tests for Smart Farm Reminders and weather conflict risk analysis."""

import os
import tempfile
from datetime import datetime, timedelta
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycle,
    CropCycleStatus,
    CropGrowthStage,
    CropSeason,
    Farm,
    FarmActivity,
    FarmerProfile,
    LiteracyPreference,
    OwnershipType,
    Plot,
    SoilType,
    UserRecord,
    UserRole,
)
from backend.app.schemas.notification import (
    FarmReminderCreate,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.services.notifications.reminder_service import ReminderService


@pytest.fixture
def reminder_fixture():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    # 1. User & Farmer
    user = UserRecord(
        user_id="usr_rem_01",
        username="farmer_rem_01",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-09T00:00:00",
    )
    repo.create_user(user)

    farmer = FarmerProfile(
        farmer_id="fmr_rem_01",
        user_id="usr_rem_01",
        full_name="Kisan Ramesh",
        phone_number="9876543210",
        state="Haryana",
        district="Karnal",
        preferred_language="hi-IN",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farmer_profile(farmer)

    # 2. Farm & Plot
    farm = Farm(
        farm_id="frm_rem_01",
        farmer_id="fmr_rem_01",
        farm_name="Green Valley",
        state="Haryana",
        district="Karnal",
        total_area=5.0,
        total_area_acres=5.0,
        area_unit=AreaUnit.ACRE,
        ownership_type=OwnershipType.OWNED,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farm(farm)

    plot = Plot(
        plot_id="plt_rem_01",
        farm_id="frm_rem_01",
        farmer_id="fmr_rem_01",
        plot_name="North Field",
        area=5.0,
        area_acres=5.0,
        area_unit=AreaUnit.ACRE,
        soil_type=SoilType.ALLUVIAL,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_plot(plot)

    # 3. Active Crop Cycle
    today_str = datetime.now().strftime("%Y-%m-%d")
    cycle = CropCycle(
        cycle_id="cyc_rem_01",
        plot_id="plt_rem_01",
        farm_id="frm_rem_01",
        farmer_id="fmr_rem_01",
        crop_name="Wheat",
        season=CropSeason.RABI,
        sowing_date=today_str,
        current_stage=CropGrowthStage.VEGETATIVE_TILLERING,
        status=CropCycleStatus.ACTIVE,
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_crop_cycle(cycle)

    # 4. Planned Activity
    act = FarmActivity(
        activity_id="act_rem_01",
        cycle_id="cyc_rem_01",
        farm_id="frm_rem_01",
        farmer_id="fmr_rem_01",
        activity_type=ActivityType.IRRIGATION,
        activity_title="Crown Root Initiation Irrigation",
        scheduled_date=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
        status=ActivityStatus.PLANNED,
        notes="First Crown Root Initiation (CRI) irrigation",
        created_at="2026-11-09T00:00:00",
        updated_at="2026-11-09T00:00:00",
    )
    repo.create_farm_activity(act)

    service = ReminderService(repo=repo)

    yield {
        "repo": repo,
        "service": service,
        "farmer": farmer,
        "cycle": cycle,
        "activity": act,
    }

    try:
        os.remove(path)
    except Exception:
        pass


def test_custom_reminder_lifecycle(reminder_fixture):
    """Verify reminder creation, listing, snoozing, and completion."""
    service = reminder_fixture["service"]
    farmer = reminder_fixture["farmer"]

    # 1. Create reminder
    due_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    req = FarmReminderCreate(
        activity_type=ReminderActivityType.FERTILIZATION,
        title_en="Apply Urea Top-Dressing",
        description_en="Apply 45kg urea per acre before next irrigation.",
        scheduled_date=due_date,
    )
    rem = service.create_reminder(farmer.farmer_id, req)
    assert rem.reminder_id is not None
    assert rem.title_en == "Apply Urea Top-Dressing"
    assert rem.status == ReminderStatus.SCHEDULED

    # 2. List reminders
    rems = service.list_reminders(farmer.farmer_id)
    assert len(rems) == 1

    # 3. Snooze reminder by 2 days
    ok_snooze = service.snooze_reminder(rem.reminder_id, snooze_days=2, farmer_id=farmer.farmer_id)
    assert ok_snooze is True
    updated = service.list_reminders(farmer.farmer_id)[0]
    assert updated.status == ReminderStatus.SNOOZED
    assert updated.snooze_until is not None

    # 4. Complete reminder
    ok_comp = service.complete_reminder(rem.reminder_id, farmer_id=farmer.farmer_id)
    assert ok_comp is True
    completed = service.list_reminders(farmer.farmer_id)[0]
    assert completed.status == ReminderStatus.COMPLETED
    assert completed.completed_at is not None


def test_auto_generate_reminders_from_crop_cycles(reminder_fixture):
    """Verify automatic generation of smart reminders from Phase 6 farm activities."""
    service = reminder_fixture["service"]
    farmer = reminder_fixture["farmer"]

    generated = service.generate_reminders_from_crop_cycles(farmer.farmer_id)
    assert len(generated) == 1
    assert "Irrigation" in generated[0].title_en
    assert generated[0].activity_type == ReminderActivityType.IRRIGATION

    # Repeated generation is idempotent (no duplicates)
    repeat = service.generate_reminders_from_crop_cycles(farmer.farmer_id)
    assert len(repeat) == 0


def test_weather_spray_conflict_flagging(reminder_fixture):
    """Verify weather risk flag is attached when rain threatens scheduled pesticide spray."""
    service = reminder_fixture["service"]
    repo = reminder_fixture["repo"]
    farmer = reminder_fixture["farmer"]

    req = FarmReminderCreate(
        activity_type=ReminderActivityType.PEST_SPRAY,
        title_en="Spray Mancozeb for Blight Control",
        description_en="Foliar fungicide application.",
        scheduled_date=datetime.now().strftime("%Y-%m-%d"),
    )
    rem = service.create_reminder(farmer.farmer_id, req)

    # Attach weather risk warning
    repo.update_farm_reminder_weather_risk(
        reminder_id=rem.reminder_id,
        risk_flag=True,
        reason_en="Heavy rain forecast (18mm) within 12h. High wash-off risk.",
        reason_hi="अगले 12 घंटों में भारी बारिश (18मिमी) की संभावना। दवा धुलने का खतरा।",
    )

    flagged = repo.get_farm_reminder(rem.reminder_id)
    assert flagged.weather_risk_flag is True
    assert "Heavy rain" in flagged.weather_risk_reason_en
    assert "भारी बारिश" in flagged.weather_risk_reason_hi


def test_reminder_crud_operations(reminder_fixture):
    """Verify GET, PUT update, and DELETE for farm reminders."""
    service = reminder_fixture["service"]
    farmer = reminder_fixture["farmer"]

    # 1. Create
    req = FarmReminderCreate(
        activity_type=ReminderActivityType.WEEDING,
        title_en="Manual Weeding Field A",
        description_en="Remove broadleaf weeds.",
        scheduled_date="2026-11-15",
    )
    rem = service.create_reminder(farmer.farmer_id, req)
    assert rem.reminder_id is not None

    # 2. Get by ID
    fetched = service.get_reminder(rem.reminder_id, farmer_id=farmer.farmer_id)
    assert fetched is not None
    assert fetched.title_en == "Manual Weeding Field A"

    # Isolation check: other farmer cannot get
    assert service.get_reminder(rem.reminder_id, farmer_id="fmr_other") is None

    # 3. Update
    from backend.app.schemas.notification import FarmReminderUpdate
    upd = FarmReminderUpdate(
        title_en="Mechanical Weeding Field A",
        scheduled_date="2026-11-16",
    )
    updated = service.update_reminder(rem.reminder_id, upd, farmer_id=farmer.farmer_id)
    assert updated is not None
    assert updated.title_en == "Mechanical Weeding Field A"
    assert updated.scheduled_date == "2026-11-16"

    # 4. Delete
    deleted = service.delete_reminder(rem.reminder_id, farmer_id=farmer.farmer_id)
    assert deleted is True
    assert service.get_reminder(rem.reminder_id) is None


def test_genuine_weather_risk_conflict_evaluation(reminder_fixture):
    """Verify real microclimate forecast cross-referencing detects spray conflicts."""
    repo = reminder_fixture["repo"]
    farmer = reminder_fixture["farmer"]
    today_str = datetime.now().strftime("%Y-%m-%d")

    from unittest.mock import MagicMock
    from backend.app.schemas.location import LocationRecord
    from backend.app.schemas.weather import DailyForecastPoint, WeatherProvenance, WeatherReport

    mock_weather = MagicMock()
    mock_weather.get_weather_report.return_value = WeatherReport(
        location=LocationRecord(
            location_id="LOC-TEST",
            display_name="Karnal, Haryana",
            state="Haryana",
            district="Karnal",
            latitude=29.68,
            longitude=76.99,
            precision="district_centroid",
        ),
        daily=[
            DailyForecastPoint(
                date_str=today_str,
                min_temperature_c=18.0,
                max_temperature_c=28.0,
                precipitation_sum_mm=8.5,
                max_wind_speed_kmh=15.0,
                max_wind_gusts_kmh=28.0,
            )
        ],
        provenance=WeatherProvenance(
            provider="Open-Meteo NWP",
            is_live=True,
            is_unavailable=False,
        ),
    )

    mock_bus = MagicMock()
    service = ReminderService(repo=repo, event_bus=mock_bus, weather_service=mock_weather)

    req = FarmReminderCreate(
        farm_id="frm_rem_01",
        activity_type=ReminderActivityType.PEST_SPRAY,
        title_en="Fungicide Spray for Rust",
        description_en="Propiconazole application.",
        scheduled_date=today_str,
    )
    rem = service.create_reminder(farmer.farmer_id, req)

    # Check that weather conflict was automatically detected
    assert rem.weather_risk_flag is True
    assert "8.5mm rain" in rem.weather_risk_reason_en
    assert "gusts 28.0 km/h" in rem.weather_risk_reason_en
    assert "Open-Meteo NWP" in rem.weather_risk_reason_en
    assert mock_bus.emit_spray_weather_conflict_event.called
