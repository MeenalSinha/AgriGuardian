"""Direct FastAPI security tests proving strict RBAC and territory boundaries."""

from datetime import datetime
import uuid
from fastapi import HTTPException
import pytest

from backend.app.main import (
    assign_officer_territory,
    auth_service,
    get_admin_dashboard_overview,
    get_current_user_auth,
    get_farmer_dashboard_overview,
    get_field_officer_cases,
    get_field_officer_dashboard_overview,
    get_metrics_catalog,
    get_my_officer_territory,
    record_case_follow_up,
    require_roles,
    system,
)
from backend.app.schemas.analytics import CaseFollowUpCreate, OfficerTerritoryAssignRequest
from backend.app.schemas.farm import UserRecord, UserRegistrationRequest, UserRole


@pytest.fixture
def auth_fixture():
    """Provision isolated Farmer, Officer, and Admin users and test cases in system DB."""
    rand_sfx = uuid.uuid4().hex[:6]
    now_str = datetime.now().isoformat()

    # 1. Register Farmer
    _, farmer_profile, farmer_token = auth_service.register_farmer(UserRegistrationRequest(
        username=f"farmer_sec_{rand_sfx}",
        password="ValidPassword123!",
        full_name="Farmer Security",
        role=UserRole.FARMER,
    ))
    farmer_user = auth_service.repo.get_user_by_username(f"farmer_sec_{rand_sfx}")

    # 2. Register Field Officer
    officer_user = UserRecord(
        user_id=f"usr_off_{rand_sfx}",
        username=f"officer_sec_{rand_sfx}",
        password_hash="dummy_hash",
        salt="dummy_salt",
        role=UserRole.FIELD_OFFICER,
        created_at=now_str,
    )
    auth_service.repo.create_user(officer_user)
    officer_token = auth_service.create_access_token(officer_user)

    # 3. Register Administrator
    admin_user = UserRecord(
        user_id=f"usr_adm_{rand_sfx}",
        username=f"admin_sec_{rand_sfx}",
        password_hash="dummy_hash",
        salt="dummy_salt",
        role=UserRole.ADMINISTRATOR,
        created_at=now_str,
    )
    auth_service.repo.create_user(admin_user)
    admin_token = auth_service.create_access_token(admin_user)

    # Assign Officer to Karnal only
    system.analytics_orchestrator.assign_officer_territory(
        officer_id=officer_user.user_id,
        state="Haryana",
        district="Karnal",
    )

    # Create one case in Karnal and one in Meerut
    c_karnal = system.analytics_orchestrator.db.create_advisory_case(
        farmer_name="Farmer Security",
        crop="Wheat",
        location="Karnal, Haryana",
        farmer_id=farmer_profile.farmer_id,
        state="Haryana",
        district="Karnal",
        symptoms="Rust",
        risk="High",
    )
    c_meerut = system.analytics_orchestrator.db.create_advisory_case(
        farmer_name="Other Farmer",
        crop="Sugarcane",
        location="Meerut, Uttar Pradesh",
        state="Uttar Pradesh",
        district="Meerut",
        symptoms="Red Rot",
        risk="High",
    )

    return {
        "farmer_user": farmer_user,
        "officer_user": officer_user,
        "admin_user": admin_user,
        "farmer_token": farmer_token,
        "officer_token": officer_token,
        "admin_token": admin_token,
        "c_karnal": c_karnal,
        "c_meerut": c_meerut,
    }


def test_public_metrics_catalog():
    """Verify metrics catalog endpoint is publicly accessible and complete."""
    catalog = get_metrics_catalog()
    assert "farmer_active_farms" in catalog
    assert "admin_total_users" in catalog
    assert "officer_registered_cases" in catalog
    assert len(catalog) >= 16


def test_unauthenticated_request_fails():
    """Verify missing bearer token raises HTTP 401 Unauthorized."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user_auth(authorization=None)
    assert exc_info.value.status_code == 401
    assert "Authorization bearer token required" in exc_info.value.detail


def test_bearer_token_validation(auth_fixture):
    """Verify valid token resolves into corresponding UserRecord."""
    officer_token = auth_fixture["officer_token"]
    user = get_current_user_auth(authorization=f"Bearer {officer_token}")
    assert user.user_id == auth_fixture["officer_user"].user_id
    assert user.role == UserRole.FIELD_OFFICER


def test_rbac_dependency_enforcement(auth_fixture):
    """Verify role-based dependency raises HTTP 403 Forbidden on role mismatch."""
    farmer = auth_fixture["farmer_user"]
    officer = auth_fixture["officer_user"]
    admin = auth_fixture["admin_user"]

    # Farmer cannot access Field Officer dependency
    officer_dep = require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])
    with pytest.raises(HTTPException) as exc_farmer:
        officer_dep(farmer)
    assert exc_farmer.value.status_code == 403
    assert "Access forbidden" in exc_farmer.value.detail

    # Officer cannot access Admin dependency
    admin_dep = require_roles([UserRole.ADMINISTRATOR])
    with pytest.raises(HTTPException) as exc_officer:
        admin_dep(officer)
    assert exc_officer.value.status_code == 403

    # Admin CAN access Admin dependency
    validated_admin = admin_dep(admin)
    assert validated_admin.user_id == admin.user_id


def test_farmer_dashboard_endpoint(auth_fixture):
    """Verify farmer overview returns structured data for authenticated farmer."""
    farmer = auth_fixture["farmer_user"]
    overview = get_farmer_dashboard_overview(user=farmer)
    assert overview.farmer.farmer_name == "Farmer Security"
    assert "farmer_active_farms" in overview.metrics


def test_field_officer_dashboard_and_cases(auth_fixture):
    """Verify officer overview and cases are scoped to Karnal only."""
    officer = auth_fixture["officer_user"]
    overview = get_field_officer_dashboard_overview(user=officer)
    assert overview.officer_id == officer.user_id
    assert overview.metrics["officer_assigned_districts"].value >= 1

    cases = get_field_officer_cases(user=officer)
    # Must only include Karnal cases, never Meerut
    assert any(c.case_id == auth_fixture["c_karnal"].case_id for c in cases)
    assert not any(c.case_id == auth_fixture["c_meerut"].case_id for c in cases)


def test_follow_up_territory_permission_api(auth_fixture):
    """Verify officer cannot submit follow-up on a case outside their assigned district."""
    officer = auth_fixture["officer_user"]
    admin = auth_fixture["admin_user"]
    c_karnal = auth_fixture["c_karnal"]
    c_meerut = auth_fixture["c_meerut"]

    # 1. Officer audits Karnal case -> Success
    res_ok = record_case_follow_up(
        case_id=c_karnal.case_id,
        req=CaseFollowUpCreate(
            notes="Field observation in assigned district.",
            new_status="in_progress",
        ),
        user=officer,
    )
    assert res_ok.case_id == c_karnal.case_id
    assert res_ok.status_transition == "in_progress"

    # 2. Officer attempts to audit Meerut case -> 403 Forbidden!
    with pytest.raises(HTTPException) as exc_bad:
        record_case_follow_up(
            case_id=c_meerut.case_id,
            req=CaseFollowUpCreate(
                notes="Attempting unauthorized cross-district update.",
                new_status="in_progress",
            ),
            user=officer,
        )
    assert exc_bad.value.status_code == 403
    assert "Access Denied" in exc_bad.value.detail

    # 3. Administrator CAN audit Meerut case -> Success (Admin global override)
    res_admin = record_case_follow_up(
        case_id=c_meerut.case_id,
        req=CaseFollowUpCreate(
            notes="Platform Admin administrative override review.",
            new_status="reviewed",
        ),
        user=admin,
    )
    assert res_admin.case_id == c_meerut.case_id


def test_admin_dashboard_endpoint(auth_fixture):
    """Verify administrator overview returns platform governance metrics."""
    admin = auth_fixture["admin_user"]
    overview = get_admin_dashboard_overview(user=admin)
    assert overview.admin_id == admin.user_id
    assert overview.metrics["admin_total_users"].value >= 3
    assert len(overview.integration_health) >= 5


def test_public_registration_blocks_privileged_role_escalation():
    """Verify that attempting to self-register as an administrator or officer is strictly blocked."""
    rand_sfx = uuid.uuid4().hex[:6]
    # Attempting to supply ADMINISTRATOR role in public registration
    user, profile, _ = auth_service.register_farmer(UserRegistrationRequest(
        username=f"hacker_adm_{rand_sfx}",
        password="ValidPassword123!",
        full_name="Attacker Attempting Escalation",
        role=UserRole.ADMINISTRATOR,
    ))
    # Must be forced to ordinary FARMER!
    assert user.role == UserRole.FARMER
    fetched = auth_service.repo.get_user_by_username(f"hacker_adm_{rand_sfx}")
    assert fetched.role == UserRole.FARMER
