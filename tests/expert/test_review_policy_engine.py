"""Unit tests for Expert Review Policy Engine triggers and SLA priority tiers."""

import pytest
from backend.app.schemas.expert import (
    ReviewEligibility,
    ReviewPriority,
    ReviewTriggerType,
)
from backend.app.services.expert.review_policy_engine import ReviewPolicyEngine


def test_policy_engine_low_confidence_trigger():
    """Verify low model confidence triggers mandatory/recommended review."""
    engine = ReviewPolicyEngine(confidence_threshold=0.65)
    result = engine.evaluate_case(
        case_id="case_low_conf",
        diagnosis_data={"confidence": 0.52, "primary_disease": "Early Blight"},
    )

    assert ReviewTriggerType.LOW_MODEL_CONFIDENCE in result.triggers
    assert result.eligibility in (ReviewEligibility.MANDATORY, ReviewEligibility.RECOMMENDED)
    assert any("below safe autonomous threshold" in r for r in result.reasons)


def test_policy_engine_conflicting_predictions_trigger():
    """Verify conflicting top-2 candidates trigger differential review."""
    engine = ReviewPolicyEngine(conflict_delta_threshold=0.15)
    result = engine.evaluate_case(
        case_id="case_conflict",
        diagnosis_data={
            "confidence": 0.82,
            "top_candidates": [
                {"disease": "Late Blight", "confidence": 0.48},
                {"disease": "Early Blight", "confidence": 0.42},
            ],
        },
    )

    assert ReviewTriggerType.CONFLICTING_PREDICTIONS in result.triggers
    assert result.priority in (ReviewPriority.HIGH, ReviewPriority.CRITICAL)
    assert any("Differential diagnosis conflict" in r for r in result.reasons)


def test_policy_engine_high_severity_and_chemical_triggers():
    """Verify high foliar damage and high-risk chemicals require dual review and critical priority."""
    engine = ReviewPolicyEngine(high_severity_damage_pct=25.0)
    result = engine.evaluate_case(
        case_id="case_chem_sev",
        diagnosis_data={
            "confidence": 0.78,
            "severity": "Critical",
            "damage_pct": 35.0,
        },
        recommendation_data={
            "chemical_control": [
                {"chemical_name": "Mancozeb 75% WP", "dosage": "800g/acre"},
            ],
        },
    )

    assert ReviewTriggerType.HIGH_SEVERITY in result.triggers
    assert ReviewTriggerType.CHEMICAL_RECOMMENDATION in result.triggers
    assert result.eligibility == ReviewEligibility.MANDATORY
    assert result.requires_dual_review is True  # Mancozeb is in HIGH_RISK_CHEMICALS


def test_policy_engine_field_officer_escalation():
    """Verify field officer on-ground escalation triggers mandatory review."""
    engine = ReviewPolicyEngine()
    result = engine.evaluate_case(
        case_id="case_officer_esc",
        diagnosis_data={"confidence": 0.90, "severity": "Medium"},
        field_officer_escalation=True,
    )

    assert ReviewTriggerType.FIELD_OFFICER_ESCALATION in result.triggers
    assert result.eligibility == ReviewEligibility.MANDATORY
    assert any("Field Officer initiated ground escalation" in r for r in result.reasons)


def test_policy_engine_farmer_explicit_request():
    """Verify farmer explicit request flags second opinion review."""
    engine = ReviewPolicyEngine()
    result = engine.evaluate_case(
        case_id="case_farmer_req",
        diagnosis_data={"confidence": 0.95, "severity": "Low"},
        farmer_requested=True,
    )

    assert ReviewTriggerType.FARMER_REQUEST in result.triggers
    assert result.priority == ReviewPriority.MEDIUM
