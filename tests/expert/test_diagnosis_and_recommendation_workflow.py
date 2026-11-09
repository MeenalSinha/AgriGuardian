"""Unit tests for complete clinical decision support, dossier assembly, recommendation versioning, and continuous learning."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.expert import (
    BiologicalControlItem,
    CaseReviewStatus,
    ChemicalInterventionItem,
    DiagnosisDecision,
    ExpertAccountStatus,
    ExpertDecisionSubmission,
    ExpertProfile,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
    FarmerBadgeStatus,
    RecommendationDecision,
    SecondaryReviewSubmission,
)
from backend.app.schemas.farm import UserRecord, UserRole
from backend.app.services.expert.expert_review_service import ExpertReviewService


@pytest.fixture
def workflow_setup():
    """Setup isolated environment for clinical decision workflow testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    # 1. Register agronomists
    u_exp1 = UserRecord(
        user_id="usr_exp1",
        username="dr_pathologist",
        password_hash="hash",
        salt="salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    repo.create_user(u_exp1)
    repo.create_expert_profile(
        ExpertProfile(
            expert_id="exp_primary",
            user_id="usr_exp1",
            full_name="Dr. Primary Pathologist",
            institution="ICAR-NCIPM",
            experience_years=14,
            specializations=ExpertSpecializationItem(
                crop_types=["Tomato", "Potato"],
                domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                states=["Haryana"],
                languages=["hi-IN", "en-IN"],
            ),
            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
        )
    )

    u_exp2 = UserRecord(
        user_id="usr_exp2",
        username="dr_senior_reviewer",
        password_hash="hash",
        salt="salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    repo.create_user(u_exp2)
    repo.create_expert_profile(
        ExpertProfile(
            expert_id="exp_secondary",
            user_id="usr_exp2",
            full_name="Dr. Senior Reviewer",
            institution="IARI Pusa",
            experience_years=25,
            specializations=ExpertSpecializationItem(
                crop_types=["Tomato", "Potato"],
                domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
                states=["Haryana"],
                languages=["hi-IN", "en-IN"],
            ),
            verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
        )
    )

    # 2. Seed farmer and advisory case with image analysis
    case = repo.create_advisory_case(
        farmer_name="Baljeet Singh",
        crop="Tomato",
        location="Karnal, Haryana",
        state="Haryana",
        district="Karnal",
        symptoms="Dark concentric target spots on lower leaves",
        risk="High",
        advisory_en="Spray standard bio-pesticide and prune affected leaves.",
        advisory_hi="प्रभावित पत्तियों को हटाएं।",
        image_analysis={"primary_disease": "Tomato Early Blight", "confidence": 0.58, "severity": "High"},
    )

    service = ExpertReviewService(repo)
    yield {
        "repo": repo,
        "service": service,
        "case_id": case.case_id,
        "db_path": path,
    }
    try:
        os.remove(path)
    except Exception:
        pass


def test_dossier_assembly_integrates_all_evidence(workflow_setup):
    """Verify dossier integrates farmer context, image analysis, weather, and immutable original AI advice."""
    service = workflow_setup["service"]
    case_id = workflow_setup["case_id"]

    dossier = service.build_case_dossier(case_id)
    assert dossier.case_id == case_id
    assert dossier.farmer_name == "Baljeet Singh"
    assert dossier.crop_name == "Tomato"
    assert dossier.ai_primary_disease == "Tomato Early Blight"
    assert dossier.ai_confidence == 0.58
    assert "Spray standard bio-pesticide" in dossier.original_ai_recommendation["advisory_en"]
    assert len(dossier.icar_retrieved_facts) >= 2


def test_expert_modifies_diagnosis_and_triggers_knowledge_proposal(workflow_setup):
    """Verify when expert modifies diagnosis, original AI is preserved, revision recorded, and proposal created."""
    service = workflow_setup["service"]
    repo = workflow_setup["repo"]
    case_id = workflow_setup["case_id"]

    # Route case to review
    rev_id, _ = service.route_advisory_case_for_review(case_id, missing_facts=False)
    repo.update_review_case(rev_id, {"assigned_expert_id": "exp_primary", "status": "in_review"})

    submission = ExpertDecisionSubmission(
        review_id=rev_id,
        diagnosis_decision=DiagnosisDecision.MODIFIED,
        revised_disease="Alternaria solani (Early Blight with Alternaria Canker)",
        pathology_notes="Target-board concentric rings indicate severe Alternaria solani with secondary stem canker formation.",
        recommendation_decision=RecommendationDecision.MODIFIED_DOSAGE_TIMING,
        cultural_management=["Destroy lower infected foliage.", "Avoid overhead furrow irrigation."],
        biological_control=[
            BiologicalControlItem(
                agent_name="Trichoderma harzianum",
                target_pest_or_disease="Alternaria solani",
                dosage="2.5 kg/acre",
                timing="Soil drenched at root zone",
            )
        ],
        chemical_control=[
            ChemicalInterventionItem(
                chemical_name="Azoxystrobin 23% SC",
                dosage_per_acre="200 ml in 200L water",
                pre_harvest_interval_days=5,
            )
        ],
        pre_harvest_interval_days=5,
        rationale="Targeted strobilurin fungicide recommended for rapid systemic arrest.",
        farmer_summary_notes="Spray Azoxystrobin at 1 ml per liter of water. Observe 5 days waiting period before harvest.",
    )

    res = service.submit_expert_decision(submission, "exp_primary")
    assert res["status"] in (CaseReviewStatus.APPROVED.value, CaseReviewStatus.MODIFIED.value)

    # 1. Original AI recommendation remains intact
    advisory = repo.get_advisory_case(case_id)
    assert advisory.advisory_en == "Spray standard bio-pesticide and prune affected leaves."
    assert advisory.status == "expert_modified"

    # 2. Knowledge improvement proposal automatically logged
    proposals = repo.list_knowledge_proposals()
    assert len(proposals) == 1
    assert proposals[0].crop == "Tomato"
    assert "Early Blight with Alternaria Canker" in proposals[0].expert_confirmed_disease

    # 3. Farmer verification badge is EXPERT_MODIFIED
    badge_info = service.get_farmer_verification_badge(case_id)
    assert badge_info.badge == FarmerBadgeStatus.EXPERT_MODIFIED
    assert badge_info.expert_name == "Dr. Primary Pathologist"
    assert "Early Blight with Alternaria Canker" in badge_info.confirmed_diagnosis
    assert len(badge_info.audio_tts_text_hi) > 10


def test_high_risk_chemical_routes_to_secondary_dual_review(workflow_setup):
    """Verify high-toxicity chemical intervention requires dual agronomist sign-off."""
    service = workflow_setup["service"]
    repo = workflow_setup["repo"]
    case_id = workflow_setup["case_id"]

    rev_id, _ = service.route_advisory_case_for_review(case_id)
    repo.update_review_case(rev_id, {"assigned_expert_id": "exp_primary", "status": "in_review"})

    submission = ExpertDecisionSubmission(
        review_id=rev_id,
        diagnosis_decision=DiagnosisDecision.CONFIRMED,
        pathology_notes="Confirmed severe Alternaria solani blight.",
        recommendation_decision=RecommendationDecision.REPLACED_FORMULATION,
        chemical_control=[
            ChemicalInterventionItem(
                chemical_name="Mancozeb 75% WP",  # High-risk chemical
                dosage_per_acre="800g per acre",
                pre_harvest_interval_days=15,
            )
        ],
        pre_harvest_interval_days=15,
        rationale="Broad-spectrum dithiocarbamate contact fungicide needed for heavy infestation.",
        farmer_summary_notes="Mancozeb spray required with strict 15-day pre-harvest waiting period.",
    )

    res = service.submit_expert_decision(submission, "exp_primary")
    assert res["requires_secondary_review"] is True
    assert res["status"] == CaseReviewStatus.SECONDARY_REVIEW_REQUIRED.value

    # Check that secondary reviewer was assigned
    rev_case = repo.get_review_case(rev_id)
    assert rev_case["secondary_expert_id"] == "exp_secondary"

    # Secondary reviewer signs off
    sec_sub = SecondaryReviewSubmission(
        review_id=rev_id,
        approved=True,
        secondary_notes="Verified Mancozeb 75% WP dosage at 800g/acre. Pre-harvest interval of 15 days is correct.",
    )
    sec_res = service.submit_secondary_review(sec_sub, "exp_secondary")
    assert sec_res["status"] == CaseReviewStatus.APPROVED.value

    # Finalized badge
    badge = service.get_farmer_verification_badge(case_id)
    assert badge.badge == FarmerBadgeStatus.EXPERT_VERIFIED
