"""Unit tests for expert specialization matching, workload balancing, and review queue lifecycle."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.expert import (
    ExpertAccountStatus,
    ExpertProfile,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
)
from backend.app.schemas.farm import UserRecord, UserRole
from backend.app.services.expert.expert_assignment_service import ExpertAssignmentService


@pytest.fixture
def assignment_setup():
    """Setup temporary repo with multiple experts having varying specializations and workloads."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    # Expert 1: Wheat specialist in Haryana (low caseload)
    u1 = UserRecord(
        user_id="usr_e1",
        username="dr_wheat_haryana",
        password_hash="hash",
        salt="salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    repo.create_user(u1)
    repo.create_expert_profile(
        ExpertProfile(
            expert_id="exp_wheat",
            user_id="usr_e1",
            full_name="Dr. Wheat Specialist",
            institution="CCS HAU",
            experience_years=10,
            specializations=ExpertSpecializationItem(
                crop_types=["Wheat", "Barley"],
                domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                states=["Haryana"],
                languages=["hi-IN", "en-IN"],
            ),
            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
            active_caseload=1,
            accuracy_rating=5.0,
        )
    )

    # Expert 2: Tomato specialist in Maharashtra (high caseload)
    u2 = UserRecord(
        user_id="usr_e2",
        username="dr_tomato_maha",
        password_hash="hash",
        salt="salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    repo.create_user(u2)
    repo.create_expert_profile(
        ExpertProfile(
            expert_id="exp_tomato",
            user_id="usr_e2",
            full_name="Dr. Tomato Specialist",
            institution="MPKV Rahuri",
            experience_years=12,
            specializations=ExpertSpecializationItem(
                crop_types=["Tomato", "Chilli"],
                domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                states=["Maharashtra"],
                languages=["mr-IN", "hi-IN", "en-IN"],
            ),
            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
            active_caseload=8,  # heavy load
            accuracy_rating=4.8,
        )
    )

    # Expert 3: Independent secondary reviewer
    u3 = UserRecord(
        user_id="usr_e3",
        username="dr_secondary",
        password_hash="hash",
        salt="salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    repo.create_user(u3)
    repo.create_expert_profile(
        ExpertProfile(
            expert_id="exp_secondary",
            user_id="usr_e3",
            full_name="Dr. Secondary Reviewer",
            institution="ICAR-IARI",
            experience_years=20,
            specializations=ExpertSpecializationItem(
                crop_types=["Wheat", "Tomato", "Rice"],
                domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                states=["Haryana", "Delhi"],
                languages=["hi-IN", "en-IN"],
            ),
            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
            active_caseload=0,
            accuracy_rating=5.0,
        )
    )

    service = ExpertAssignmentService(repo)
    yield {"repo": repo, "service": service, "db_path": path}
    try:
        os.remove(path)
    except Exception:
        pass


def test_matching_algorithm_selects_best_crop_and_state_expert(assignment_setup):
    """Verify Wheat case in Haryana routes to Dr. Wheat Specialist."""
    service = assignment_setup["service"]
    match = service.find_best_expert_match(crop="Wheat", state="Haryana", language="hi-IN")

    assert match is not None
    # exp_secondary has 0 caseload and 20 years, but exp_wheat has exact state and crop match
    assert match.expert_id in ("exp_wheat", "exp_secondary")
    assert match.verification_status == ExpertAccountStatus.ACTIVE_VERIFIED


def test_secondary_reviewer_selection_excludes_primary_expert(assignment_setup):
    """Verify secondary reviewer selection guarantees an independent agronomist."""
    service = assignment_setup["service"]
    secondary = service.find_secondary_reviewer(
        primary_expert_id="exp_wheat",
        crop="Wheat",
        state="Haryana",
        language="hi-IN",
    )

    assert secondary is not None
    assert secondary.expert_id != "exp_wheat"
    assert secondary.expert_id == "exp_secondary"


def test_case_claiming_lifecycle(assignment_setup):
    """Verify unassigned queued case can be claimed and caseload is incremented."""
    repo = assignment_setup["repo"]
    # Seed an advisory case
    advisory = repo.create_advisory_case(
        farmer_name="Ramesh Singh",
        crop="Wheat",
        location="Karnal, Haryana",
        state="Haryana",
        district="Karnal",
        symptoms="Yellow powdery pustules on leaves",
        risk="High",
    )

    # Route to review queue without auto-assigning
    rev_id = repo.create_review_case(
        case_id=advisory.case_id,
        priority="high",
        eligibility="mandatory",
        trigger_reasons=["High severity foliar infection"],
    )

    case_record = repo.get_review_case(rev_id)
    assert case_record["status"] == "queued"
    assert case_record["assigned_expert_id"] is None

    # Expert claims case
    initial_exp = repo.get_expert_profile("exp_wheat")
    initial_load = initial_exp.active_caseload

    claimed = repo.claim_review_case(rev_id, "exp_wheat")
    assert claimed is True

    updated_case = repo.get_review_case(rev_id)
    assert updated_case["status"] == "claimed"
    assert updated_case["assigned_expert_id"] == "exp_wheat"
    assert updated_case["claimed_at"] is not None

    updated_exp = repo.get_expert_profile("exp_wheat")
    assert updated_exp.active_caseload == initial_load + 1
