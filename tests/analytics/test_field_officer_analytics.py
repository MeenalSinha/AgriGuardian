"""Unit and integration tests for Field Officer Analytics and Case Review Service."""

from datetime import datetime
import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.analytics import MetricStatus
from backend.app.schemas.farm import UserRecord, UserRole
from backend.app.services.analytics.field_officer_analytics import FieldOfficerAnalyticsService


@pytest.fixture
def officer_setup():
    """Create isolated SQLite database with test cases and officer assignments."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=db_path)
    analytics = FieldOfficerAnalyticsService(farm_db=repo)

    now_str = datetime.now().isoformat()
    repo.create_user(UserRecord(
        user_id="off_karnal",
        username="officer_karnal",
        password_hash="hash",
        salt="salt",
        role=UserRole.FIELD_OFFICER,
        created_at=now_str,
    ))
    repo.create_user(UserRecord(
        user_id="off_meerut",
        username="officer_meerut",
        password_hash="hash",
        salt="salt",
        role=UserRole.FIELD_OFFICER,
        created_at=now_str,
    ))

    yield {
        "repo": repo,
        "analytics": analytics,
        "db_path": db_path,
    }

    try:
        os.remove(db_path)
    except Exception:
        pass


def test_officer_unassigned_territory(officer_setup):
    """Verify an officer with no assigned territory receives informative status."""
    analytics = officer_setup["analytics"]
    overview = analytics.get_officer_overview(officer_id="off_unassigned", officer_name="Unassigned Officer")

    assert overview.officer_id == "off_unassigned"
    assert len(overview.assigned_territory) == 0
    assert overview.metrics["officer_assigned_districts"].is_available is False
    assert overview.metrics["officer_registered_cases"].value == 0
    assert "No geographic territory" in overview.metrics["officer_assigned_districts"].missing_reason


def test_officer_territory_scoping_and_privacy_masking(officer_setup):
    """Verify territory scoping isolates district cases and masks small cohorts."""
    repo = officer_setup["repo"]
    analytics = officer_setup["analytics"]

    # Assign Officer 1 to Karnal
    repo.assign_field_officer_territory(officer_id="off_karnal", state="Haryana", district="Karnal")

    # Assign Officer 2 to Meerut
    repo.assign_field_officer_territory(officer_id="off_meerut", state="Uttar Pradesh", district="Meerut")

    # Create 3 cases in Karnal (cohort size = 3 -> privacy NOT masked)
    c_k1 = repo.create_advisory_case(
        farmer_name="Ramesh Kumar",
        crop="Wheat",
        location="Karnal, Haryana",
        farmer_id="fmr_1",
        state="Haryana",
        district="Karnal",
        symptoms="Yellow rust spots",
        risk="High",
        field_visit_required=True,
    )
    c_k2 = repo.create_advisory_case(
        farmer_name="Sita Devi",
        crop="Wheat",
        location="Karnal, Haryana",
        farmer_id="fmr_2",
        state="Haryana",
        district="Karnal",
        symptoms="Leaf blight",
        risk="Medium",
        field_visit_required=False,
    )
    c_k3 = repo.create_advisory_case(
        farmer_name="Anil Yadav",
        crop="Rice",
        location="Karnal, Haryana",
        farmer_id="fmr_3",
        state="Haryana",
        district="Karnal",
        symptoms="Stem borer holes",
        risk="Low",
        field_visit_required=False,
    )

    # Create 1 case in Meerut (cohort size = 1 -> privacy MUST be masked)
    c_m1 = repo.create_advisory_case(
        farmer_name="Secret Farmer",
        crop="Sugarcane",
        location="Meerut, Uttar Pradesh",
        farmer_id="fmr_4",
        state="Uttar Pradesh",
        district="Meerut",
        symptoms="Red rot",
        risk="High",
        field_visit_required=True,
    )

    # 1. Officer 1 (Karnal) should see exactly 3 cases, none from Meerut
    ov_karnal = analytics.get_officer_overview(officer_id="off_karnal", officer_name="Karnal Officer")
    assert ov_karnal.metrics["officer_assigned_districts"].value == 1
    assert ov_karnal.metrics["officer_registered_cases"].value == 3
    assert ov_karnal.metrics["officer_high_risk_cases"].value == 1
    assert ov_karnal.risk_distribution == {"Low": 1, "Medium": 1, "High": 1}

    # Karnal cluster has 3 cases -> NOT masked
    cluster_karnal = next((c for c in ov_karnal.village_clusters if c["location"] == "Karnal"), None)
    assert cluster_karnal is not None
    assert cluster_karnal["case_count"] == 3
    assert cluster_karnal["privacy_masked"] is False
    assert "Ramesh Kumar" in cluster_karnal["sample_farmers"]

    # 2. Officer 2 (Meerut) should see exactly 1 case, and it MUST be privacy masked (<3)
    ov_meerut = analytics.get_officer_overview(officer_id="off_meerut", officer_name="Meerut Officer")
    assert ov_meerut.metrics["officer_registered_cases"].value == 1
    cluster_meerut = next((c for c in ov_meerut.village_clusters if c["location"] == "Meerut"), None)
    assert cluster_meerut is not None
    assert cluster_meerut["case_count"] == 1
    assert cluster_meerut["privacy_masked"] is True
    # Individual name suppressed in sample farmers
    assert "Confidential Farmer" in cluster_meerut["sample_farmers"]
    assert "Secret Farmer" not in cluster_meerut["sample_farmers"]


def test_officer_follow_up_jurisdiction_enforcement(officer_setup):
    """Verify officer cannot record follow-up on a case outside their assigned district."""
    repo = officer_setup["repo"]
    analytics = officer_setup["analytics"]

    repo.assign_field_officer_territory(officer_id="off_karnal", state="Haryana", district="Karnal")

    # Case in Meerut
    c_m1 = repo.create_advisory_case(
        farmer_name="Chaudhary Ji",
        crop="Sugarcane",
        location="Meerut, Uttar Pradesh",
        state="Uttar Pradesh",
        district="Meerut",
        symptoms="Red rot",
        risk="High",
        field_visit_required=True,
    )

    # Karnal officer attempts to audit Meerut case -> MUST raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        analytics.record_case_follow_up(
            case_id=c_m1.case_id,
            officer_id="off_karnal",
            officer_name="Karnal Officer",
            notes="Attempted inspection outside my area.",
            new_status="in_progress",
            is_admin=False,
        )

    assert "Access Denied" in str(exc_info.value)
    assert "outside officer 'off_karnal' assigned territory" in str(exc_info.value)


def test_officer_follow_up_lifecycle_and_audit_trail(officer_setup):
    """Verify successful follow-up creates chronological audit trail and updates status."""
    repo = officer_setup["repo"]
    analytics = officer_setup["analytics"]

    repo.assign_field_officer_territory(officer_id="off_karnal", state="Haryana", district="Karnal")

    case = repo.create_advisory_case(
        farmer_name="Ramesh Kumar",
        crop="Wheat",
        location="Karnal, Haryana",
        state="Haryana",
        district="Karnal",
        symptoms="Yellow rust",
        risk="High",
        field_visit_required=True,
        status="pending",
    )

    # Record first follow-up: in_progress
    rec1 = analytics.record_case_follow_up(
        case_id=case.case_id,
        officer_id="off_karnal",
        officer_name="Officer Sharma",
        notes="Conducted plot visit. Confirmed Pucciniaceae stripe rust.",
        action_taken="Prescribed Propiconazole 25% EC @ 1ml/L.",
        new_status="in_progress",
    )
    assert rec1.case_id == case.case_id
    assert rec1.status_transition == "in_progress"

    # Verify case status updated in DB
    updated_case = repo.get_advisory_case(case.case_id)
    assert updated_case.status == "in_progress"

    # Record second follow-up: resolved
    rec2 = analytics.record_case_follow_up(
        case_id=case.case_id,
        officer_id="off_karnal",
        officer_name="Officer Sharma",
        notes="Re-inspected plot after 7 days. Foliar lesions arrested; new growth clean.",
        action_taken="Advised normal irrigation schedule.",
        new_status="resolved",
    )
    assert rec2.status_transition == "resolved"

    # Verify audit trail
    history = analytics.list_case_follow_ups(case.case_id, officer_id="off_karnal")
    assert len(history) == 2
    assert history[0].notes == rec1.notes
    assert history[1].notes == rec2.notes
