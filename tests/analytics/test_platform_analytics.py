"""Unit and integration tests for Platform Administrator Analytics Service."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.analytics import MetricStatus
from backend.app.schemas.farm import (
    AreaUnit,
    FarmCreate,
    OwnershipType,
    UserRegistrationRequest,
    UserRole,
)
from backend.app.services.analytics.platform_analytics import PlatformAnalyticsService
from backend.app.services.auth.auth_service import AuthService
from backend.app.services.farm.farm_service import FarmService


@pytest.fixture
def admin_setup():
    """Create isolated SQLite database with users and farms."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=db_path)
    auth_srv = AuthService(repo=repo)
    farm_srv = FarmService(repo=repo)
    analytics = PlatformAnalyticsService(farm_db=repo)

    # Create 2 farmers
    _, f1, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_one",
        password="ValidPassword123!",
        full_name="Farmer One",
        role=UserRole.FARMER,
    ))
    _, f2, _ = auth_srv.register_farmer(UserRegistrationRequest(
        username="farmer_two",
        password="ValidPassword123!",
        full_name="Farmer Two",
        role=UserRole.FARMER,
    ))

    # Add 2 farms
    farm_srv.create_farm(f1.farmer_id, FarmCreate(
        farm_name="Farm 1",
        state="Haryana",
        district="Karnal",
        total_area=4.0,
        area_unit=AreaUnit.ACRE,
    ))
    farm_srv.create_farm(f2.farmer_id, FarmCreate(
        farm_name="Farm 2",
        state="Punjab",
        district="Patiala",
        total_area=6.0,
        area_unit=AreaUnit.ACRE,
    ))

    # Create 1 advisory case
    repo.create_advisory_case(
        farmer_name="Farmer One",
        crop="Wheat",
        location="Karnal, Haryana",
        farmer_id=f1.farmer_id,
        state="Haryana",
        district="Karnal",
        symptoms="Rust spots",
        risk="High",
    )

    yield {
        "repo": repo,
        "analytics": analytics,
        "db_path": db_path,
    }

    try:
        os.remove(db_path)
    except Exception:
        pass


def test_admin_dashboard_metrics(admin_setup):
    """Verify platform-wide aggregation of users, farms, advisories, and uptime."""
    analytics = admin_setup["analytics"]
    overview = analytics.get_admin_overview(admin_id="adm_test", admin_name="Admin Test")

    assert overview.admin_id == "adm_test"
    assert overview.metrics["admin_total_users"].value == 2
    assert overview.metrics["admin_active_users_30d"].value == 2
    assert overview.metrics["admin_platform_farms"].value == 2
    assert overview.metrics["admin_total_advisories"].value == 1
    assert overview.metrics["admin_system_uptime"].status == MetricStatus.OPTIMAL
    assert overview.metrics["admin_api_error_rate"].status == MetricStatus.OPTIMAL


def test_admin_integration_health_checks(admin_setup):
    """Verify integration health records for all core external and internal subsystems."""
    analytics = admin_setup["analytics"]
    overview = analytics.get_admin_overview()

    service_names = {h.service_name for h in overview.integration_health}
    assert "Gnani AI Voice Services" in service_names
    assert "PyTorch MobileNetV3 Classifier" in service_names
    assert "Open-Meteo Agro-Meteorology API" in service_names
    assert "AGMARKNET Mandi Price Registry" in service_names
    assert "Agronomic Evidence Knowledge Base" in service_names

    # Check that each integration record has status, latency, and details
    for h in overview.integration_health:
        assert h.status in ("active", "degraded", "mock_mode")
        assert h.latency_ms >= 0.0
        assert len(h.details) > 0


def test_admin_model_and_data_quality_telemetry(admin_setup):
    """Verify model telemetry and data quality audits."""
    analytics = admin_setup["analytics"]
    overview = analytics.get_admin_overview()

    # Model telemetry
    mp = overview.model_performance
    assert "MobileNetV3" in mp.model_name
    assert mp.avg_latency_ms > 0
    assert mp.device is not None

    # Data quality audit
    dq = overview.data_quality
    assert dq.overall_quality_pct > 90.0
    assert dq.missing_market_records == 0
    assert dq.stale_weather_records == 0
