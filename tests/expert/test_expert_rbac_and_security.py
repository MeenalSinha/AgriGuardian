"""Security, role-based access control (RBAC), and authentication tests for Expert APIs."""

import os
import tempfile
import uuid
import pytest
from fastapi import HTTPException

from backend.app.main import (
    admin_provision_expert,
    admin_verify_expert,
    auth_service,
    claim_case,
    get_case_dossier,
    get_current_user_auth,
    get_my_expert_profile,
    list_experts,
    list_review_cases,
    require_roles,
    submit_expert_decision,
    system,
)
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.expert import (
    ExpertAccountStatus,
    ExpertDecisionSubmission,
    ExpertProfile,
    ExpertProvisionRequest,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
    ExpertVerificationAction,
    DiagnosisDecision,
    RecommendationDecision,
)
from backend.app.schemas.farm import UserRecord, UserRegistrationRequest, UserRole


@pytest.fixture
def rbac_fixture():
    """Setup isolated database, users, and tokens across roles."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    old_repo = system.farm_context.repo
    system.farm_context.repo = repo
    auth_service.repo = repo
    system.expert_review_service.repo = repo
    system.expert_review_service.assignment_service.repo = repo

    rand_sfx = uuid.uuid4().hex[:6]

    # 1. Admin user
    u_admin, tok_admin = auth_service.provision_privileged_user(
        f"admin_sec_{rand_sfx}", "AdminPass123!", UserRole.ADMINISTRATOR
    )

    # 2. Expert user & profile
    u_expert, tok_expert = auth_service.provision_privileged_user(
        f"expert_sec_{rand_sfx}", "ExpertPass123!", UserRole.AGRONOMIST
    )
    exp_prof = ExpertProfile(
        expert_id=f"exp_{rand_sfx}",
        user_id=u_expert.user_id,
        full_name="Dr. Security Expert",
        institution="ICAR-IARI",
        designation="Senior Agronomist",
        specializations=ExpertSpecializationItem(
            crop_types=["Wheat", "Tomato"],
            domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
            states=["Haryana"],
            languages=["hi-IN", "en-IN"],
        ),
        verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
    )
    repo.create_expert_profile(exp_prof)

    # 3. Field Officer
    u_officer, tok_officer = auth_service.provision_privileged_user(
        f"officer_sec_{rand_sfx}", "OfficerPass123!", UserRole.FIELD_OFFICER
    )

    # 4. Farmer
    _, fmr_profile, tok_farmer = auth_service.register_farmer(
        UserRegistrationRequest(
            username=f"farmer_sec_{rand_sfx}",
            password="FarmerPass123!",
            full_name="Farmer Security",
        )
    )
    u_farmer = repo.get_user_by_username(f"farmer_sec_{rand_sfx}")

    yield {
        "repo": repo,
        "admin": u_admin,
        "expert": u_expert,
        "officer": u_officer,
        "farmer": u_farmer,
        "tokens": {
            "admin": tok_admin,
            "expert": tok_expert,
            "officer": tok_officer,
            "farmer": tok_farmer,
        },
        "expert_profile": exp_prof,
    }

    system.farm_context.repo = old_repo
    auth_service.repo = old_repo
    try:
        os.remove(path)
    except Exception:
        pass


def test_unauthenticated_requests_blocked():
    """Verify missing authorization header raises HTTP 401 Unauthorized."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user_auth(authorization=None)
    assert exc_info.value.status_code == 401
    assert "Authorization bearer token required" in exc_info.value.detail


def test_farmer_cannot_access_expert_or_admin_endpoints(rbac_fixture):
    """Verify Farmer role is strictly rejected with HTTP 403 on agronomist/admin operations."""
    farmer = rbac_fixture["farmer"]

    # Expert endpoint role check
    check_expert = require_roles([UserRole.AGRONOMIST])
    with pytest.raises(HTTPException) as exc_info:
        check_expert(user=farmer)
    assert exc_info.value.status_code == 403
    assert "Requires one of" in exc_info.value.detail

    # Admin endpoint role check
    check_admin = require_roles([UserRole.ADMINISTRATOR])
    with pytest.raises(HTTPException) as exc_info:
        check_admin(user=farmer)
    assert exc_info.value.status_code == 403


def test_field_officer_cannot_act_as_expert(rbac_fixture):
    """Verify Field Officer cannot claim cases or submit expert prescriptions."""
    officer = rbac_fixture["officer"]

    # Role check for expert actions
    check_expert = require_roles([UserRole.AGRONOMIST])
    with pytest.raises(HTTPException) as exc_info:
        check_expert(user=officer)
    assert exc_info.value.status_code == 403


def test_agronomist_cannot_verify_other_experts_or_provision(rbac_fixture):
    """Verify Agronomist cannot access administrator-only provisioning and verification."""
    expert = rbac_fixture["expert"]

    check_admin = require_roles([UserRole.ADMINISTRATOR])
    with pytest.raises(HTTPException) as exc_info:
        check_admin(user=expert)
    assert exc_info.value.status_code == 403


def test_admin_provision_expert_endpoint(rbac_fixture):
    """Verify administrator successfully provisions a new verified expert account."""
    admin = rbac_fixture["admin"]
    req = ExpertProvisionRequest(
        username="dr_anita_pau",
        password="SecurePass2026!",
        full_name="Dr. Anita Kaur",
        email="anita@pau.edu",
        institution="Punjab Agricultural University",
        designation="Professor of Entomology",
        experience_years=18,
        crop_types=["Rice", "Cotton"],
        domains=["entomology"],
        states=["Punjab"],
        initial_credential_institution="PAU Ludhiana",
        initial_credential_degree="Ph.D. Agricultural Entomology",
        initial_credential_reg_num="PAU-ENT-2010-884",
    )

    created_prof = admin_provision_expert(req, admin=admin)
    assert created_prof.full_name == "Dr. Anita Kaur"
    assert created_prof.verification_status == ExpertAccountStatus.ACTIVE_VERIFIED
    assert created_prof.verified_by_admin_id == admin.user_id
    assert len(created_prof.credentials) == 1
    assert created_prof.credentials[0].is_verified is True


def test_agronomist_retrieves_own_profile(rbac_fixture):
    """Verify authenticated agronomist retrieves their profile and credentials."""
    expert = rbac_fixture["expert"]
    profile = get_my_expert_profile(user=expert)
    assert profile.full_name == "Dr. Security Expert"
    assert profile.institution == "ICAR-IARI"
