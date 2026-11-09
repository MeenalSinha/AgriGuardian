"""Policy Engine determining eligibility, priority, and dual-review requirements for Expert Review."""

from typing import Any, Dict, List, Optional
from backend.app.schemas.expert import (
    ExpertReviewPolicyResult,
    ReviewEligibility,
    ReviewPriority,
    ReviewTriggerType,
)
from backend.app.core.logging import get_logger

logger = get_logger("ReviewPolicyEngine")

# High-risk active ingredients requiring mandatory expert / secondary review
HIGH_RISK_CHEMICALS = {
    "mancozeb",
    "chlorothalonil",
    "propiconazole",
    "carbendazim",
    "streptocycline",
    "hexaconazole",
    "metalaxyl",
    "copper oxychloride",
    "monocrotophos",
    "imidacloprid",
}


class ReviewPolicyEngine:
    """Configurable rule and threshold engine evaluating agricultural cases for human agronomist review."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        conflict_delta_threshold: float = 0.15,
        high_severity_damage_pct: float = 25.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.conflict_delta_threshold = conflict_delta_threshold
        self.high_severity_damage_pct = high_severity_damage_pct

    def evaluate_case(
        self,
        case_id: str,
        diagnosis_data: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict[str, Any]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        field_officer_escalation: bool = False,
        farmer_requested: bool = False,
        missing_facts: bool = False,
    ) -> ExpertReviewPolicyResult:
        """Evaluate comprehensive case telemetry against safety and quality policies."""
        triggers: List[ReviewTriggerType] = []
        reasons: List[str] = []
        requires_dual_review = False

        diagnosis_data = diagnosis_data or {}
        recommendation_data = recommendation_data or {}
        weather_data = weather_data or {}

        # 1. Inspect Model Confidence
        confidence = float(diagnosis_data.get("confidence", 1.0))
        if confidence < self.confidence_threshold:
            triggers.append(ReviewTriggerType.LOW_MODEL_CONFIDENCE)
            reasons.append(
                f"Model prediction confidence ({confidence * 100:.1f}%) is below safe autonomous threshold ({self.confidence_threshold * 100:.1f}%)."
            )

        # 2. Inspect Conflicting Top-2 Predictions
        top_candidates = diagnosis_data.get("top_candidates") or diagnosis_data.get("secondary_diseases") or []
        if isinstance(top_candidates, list) and len(top_candidates) >= 2:
            p1 = float(top_candidates[0].get("probability", top_candidates[0].get("confidence", 0.0)))
            p2 = float(top_candidates[1].get("probability", top_candidates[1].get("confidence", 0.0)))
            delta = abs(p1 - p2)
            if delta < self.conflict_delta_threshold:
                triggers.append(ReviewTriggerType.CONFLICTING_PREDICTIONS)
                d1 = top_candidates[0].get("disease", "Candidate 1")
                d2 = top_candidates[1].get("disease", "Candidate 2")
                reasons.append(
                    f"Differential diagnosis conflict between '{d1}' ({p1 * 100:.1f}%) and '{d2}' ({p2 * 100:.1f}%); delta ({delta * 100:.1f}%) < {self.conflict_delta_threshold * 100:.1f}%."
                )

        # 3. Inspect Foliar Damage Severity
        severity = str(diagnosis_data.get("severity", "Low")).lower()
        foliar_damage = float(diagnosis_data.get("affected_area_percentage", diagnosis_data.get("damage_pct", 0.0)))
        if severity in ("high", "critical", "severe") or foliar_damage >= self.high_severity_damage_pct:
            triggers.append(ReviewTriggerType.HIGH_SEVERITY)
            reasons.append(
                f"Elevated crop infection severity: level '{severity.title()}' with estimated damage {foliar_damage:.1f}%."
            )

        # 4. Inspect Chemical Interventions
        chemicals = (
            recommendation_data.get("chemical_control")
            or recommendation_data.get("chemical_treatments")
            or recommendation_data.get("chemicals")
            or []
        )
        has_chemical = False
        if isinstance(chemicals, list) and len(chemicals) > 0:
            has_chemical = True
            triggers.append(ReviewTriggerType.CHEMICAL_RECOMMENDATION)
            chem_names = []
            for c in chemicals:
                c_name = c.get("chemical_name", "") if isinstance(c, dict) else str(c)
                chem_names.append(c_name)
                # Check for high-toxicity or high-risk active ingredients
                c_lower = c_name.lower()
                if any(hr in c_lower for hr in HIGH_RISK_CHEMICALS):
                    requires_dual_review = True
            reasons.append(
                f"Synthetic chemical intervention prescribed ({', '.join(chem_names[:3])}); requires accredited agronomist validation."
            )

        # 5. Missing or Contradictory ICAR Knowledge Base Facts
        if missing_facts or recommendation_data.get("missing_icar_facts", False):
            triggers.append(ReviewTriggerType.MISSING_ICAR_FACTS)
            reasons.append("Retrieved agronomic evidence lacks verified ICAR or CIBRC registration grounding.")

        # 6. Field Officer Ground Escalation
        if field_officer_escalation:
            triggers.append(ReviewTriggerType.FIELD_OFFICER_ESCALATION)
            reasons.append("Field Officer initiated ground escalation following physical farm visit inspection.")

        # 7. Farmer Explicit Request
        if farmer_requested:
            triggers.append(ReviewTriggerType.FARMER_REQUEST)
            reasons.append("Farmer explicitly requested accredited agricultural specialist review and second opinion.")

        # Determine Eligibility
        if (
            ReviewTriggerType.FIELD_OFFICER_ESCALATION in triggers
            or ReviewTriggerType.LOW_MODEL_CONFIDENCE in triggers
            or (ReviewTriggerType.CHEMICAL_RECOMMENDATION in triggers and ReviewTriggerType.HIGH_SEVERITY in triggers)
        ):
            eligibility = ReviewEligibility.MANDATORY
        elif triggers:
            eligibility = ReviewEligibility.RECOMMENDED
        else:
            eligibility = ReviewEligibility.OPTIONAL

        # Determine Priority
        if ReviewTriggerType.FIELD_OFFICER_ESCALATION in triggers and ReviewTriggerType.HIGH_SEVERITY in triggers:
            priority = ReviewPriority.CRITICAL
        elif ReviewTriggerType.LOW_MODEL_CONFIDENCE in triggers and has_chemical:
            priority = ReviewPriority.CRITICAL
        elif ReviewTriggerType.HIGH_SEVERITY in triggers or ReviewTriggerType.CONFLICTING_PREDICTIONS in triggers:
            priority = ReviewPriority.HIGH
        elif ReviewTriggerType.CHEMICAL_RECOMMENDATION in triggers or ReviewTriggerType.FARMER_REQUEST in triggers:
            priority = ReviewPriority.MEDIUM
        else:
            priority = ReviewPriority.LOW

        return ExpertReviewPolicyResult(
            case_id=case_id,
            eligibility=eligibility,
            priority=priority,
            triggers=triggers,
            reasons=reasons,
            requires_dual_review=requires_dual_review,
        )

