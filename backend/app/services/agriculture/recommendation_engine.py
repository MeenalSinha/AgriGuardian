"""Central Recommendation Engine synthesizing verified knowledge, vision, weather, and safety guardrails."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.evidence import EvidenceExcerpt
from backend.app.schemas.knowledge import AtomicFact
from backend.app.schemas.recommendation import ActionItem, PersonalizedRecommendation
from backend.app.services.agriculture.applicability_service import ApplicabilityService
from backend.app.services.agriculture.conflict_resolution_service import ConflictResolutionService
from backend.app.services.agriculture.retrieval_service import RetrievalService
from backend.app.services.agriculture.safety_guardrail_service import SafetyGuardrailService

logger = get_logger("RecommendationEngine")


class RecommendationEngine:
    """Orchestrates evidence-grounded, context-aware agricultural recommendation generation."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        applicability_service: Optional[ApplicabilityService] = None,
        conflict_service: Optional[ConflictResolutionService] = None,
        safety_service: Optional[SafetyGuardrailService] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.applicability_service = applicability_service or ApplicabilityService()
        self.conflict_service = conflict_service or ConflictResolutionService()
        self.safety_service = safety_service or SafetyGuardrailService()

    def generate_recommendation(
        self,
        query: str,
        crop: str = "tomato",
        language: str = "hi-IN",
        state: Optional[str] = None,
        crop_stage: Optional[str] = None,
        days_after_sowing: Optional[int] = None,
        weather_context: Optional[Dict[str, Any]] = None,
        image_diagnosis: Optional[Dict[str, Any]] = None,
    ) -> PersonalizedRecommendation:
        """Execute complete evidence-based recommendation pipeline."""
        rec_id = f"REC-{uuid.uuid4().hex[:10].upper()}"

        # 1. Prompt Injection & Hazardous Chemical Security Inspection
        is_safe, refusal_msg = self.safety_service.inspect_query_security(query)
        if not is_safe:
            return PersonalizedRecommendation(
                recommendation_id=rec_id,
                status="insufficient_evidence",
                query_summary={"text": query, "language": language, "intent": "security_refusal"},
                context_used={"crop": crop},
                assessment={"summary": refusal_msg, "diagnosis_status": "rejected"},
                actions=[],
                precautions=["System safety policy violation detected."],
                missing_information=[],
                clarifying_questions=[],
                supporting_evidence=[],
                confidence_assessment={"level": "Low", "rationale": "Query failed security validation."},
                expert_referral_required=False,
                response_language=language,
            )

        # 2. Extract Context Variables
        farmer_context = {
            "crop": crop,
            "crop_stage": crop_stage,
            "days_after_sowing": days_after_sowing,
            "state": state,
            "weather": weather_context or {},
        }

        # Inject vision diagnosis if provided
        symptom_query = query
        diag_status = "suspected"
        severity_stage = "Mild"
        is_confirmed = False

        if image_diagnosis:
            common_disease = image_diagnosis.get("common_name", "")
            conf = float(image_diagnosis.get("confidence", 0.0))
            is_conf = image_diagnosis.get("is_confident", False)
            sev_data = image_diagnosis.get("severity_assessment", {})
            severity_stage = sev_data.get("severity_stage", "Mild")

            farmer_context["severity_stage"] = severity_stage
            farmer_context["confidence"] = conf

            if is_conf and conf >= 0.75:
                diag_status = "confirmed"
                is_confirmed = True
            elif is_conf:
                diag_status = "suspected"
            else:
                diag_status = "uncertain"

            symptom_query = f"{query} {common_disease}"

        # 3. Retrieve Approved Agricultural Knowledge
        retrieved_tuples = self.retrieval_service.retrieve(
            query=symptom_query,
            crop=crop,
            state=state,
            top_k=6,
        )
        raw_facts = [item[0] for item in retrieved_tuples]

        # 4. Filter by Contextual Applicability
        applicable_facts: List[AtomicFact] = []
        all_missing_prereqs: List[str] = []

        for fact in raw_facts:
            is_app, reasons, missing = self.applicability_service.evaluate_applicability(fact, farmer_context)
            if is_app:
                applicable_facts.append(fact)
            all_missing_prereqs.extend(missing)

        # Fallback to general approved facts if strict crop filter yields empty
        if not applicable_facts and raw_facts:
            applicable_facts = raw_facts[:2]

        # 5. Conflict Resolution
        reconciled_facts, conflicts = self.conflict_service.arbitrate_facts(applicable_facts)

        # 6. Extract Supporting Evidence Provenance
        supporting_evidence = self.retrieval_service.extract_evidence(reconciled_facts)

        # 7. Formulate Actions Grounded in Retrieved Facts
        actions: List[ActionItem] = []
        precautions: List[str] = []

        if reconciled_facts:
            for fact in reconciled_facts:
                ev_ids = [e.evidence_id for e in fact.evidence]

                if fact.predicate == "managed_by_cultural_practice":
                    actions.append(
                        ActionItem(
                            priority="immediate",
                            action=fact.value,
                            reason=f"Cultural sanitation practice grounded in {fact.evidence[0].organization if fact.evidence else 'ICAR standards'}.",
                            evidence_ids=ev_ids,
                            safety_level="low",
                        )
                    )
                elif fact.predicate == "managed_by_biological_agent":
                    actions.append(
                        ActionItem(
                            priority="secondary",
                            action=fact.value,
                            reason="Bio-control agent suppressing foliar inoculum without chemical residue.",
                            evidence_ids=ev_ids,
                            safety_level="low",
                        )
                    )
                elif fact.predicate == "chemical_precaution_recommended":
                    actions.append(
                        ActionItem(
                            priority="secondary",
                            action=fact.value,
                            reason="Statutory chemical protectant registered with CIBRC for severe outbreak containment.",
                            evidence_ids=ev_ids,
                            safety_level="chemical_restricted",
                        )
                    )
                elif fact.predicate == "has_critical_irrigation_stage":
                    actions.append(
                        ActionItem(
                            priority="immediate",
                            action=fact.value,
                            reason="Critical physiological growth window where water stress causes irreversible yield reduction.",
                            evidence_ids=ev_ids,
                            safety_level="low",
                        )
                    )
                elif fact.predicate == "susceptible_to_weather_risk":
                    precautions.append(fact.value)

        # 8. Apply Safety Guardrails to Chemical Interventions
        sanitized_actions = self.safety_service.sanitize_action_items(
            actions=actions,
            is_confirmed_diagnosis=is_confirmed,
        )

        # 9. Evaluate KVK Expert Escalation
        expert_req, expert_details = self.safety_service.evaluate_escalation_triggers(
            severity_stage=severity_stage,
            confidence=farmer_context.get("confidence", 0.8),
            is_uncertain=(diag_status == "uncertain"),
        )

        # 10. Generate Clarifying Questions for Missing Prereqs
        clarifying_qs: List[str] = []
        unique_missing = list(set(all_missing_prereqs))
        if unique_missing:
            for m in unique_missing[:2]:
                clarifying_qs.append(f"Could you confirm your crop's exact growth stage or days after sowing ({m})?")

        # 11. Determine Confidence Assessment
        if is_confirmed and supporting_evidence:
            conf_level = "High"
            conf_rationale = "Diagnosis corroborated by computer vision and grounded in Tier-1 statutory agricultural evidence."
        elif supporting_evidence:
            conf_level = "Moderate"
            conf_rationale = "Grounded in verified agricultural guidelines; field observation recommended to confirm symptoms."
        else:
            conf_level = "Low"
            conf_rationale = "Limited specific evidence in local knowledge base for this exact condition; expert consultation recommended."

        # Summary assessment text
        issue_name = image_diagnosis.get("common_name") if image_diagnosis else crop.title()
        summary_text = (
            f"Assessment for {crop.title()}: {issue_name} ({diag_status.title()} condition, {severity_stage} stage). "
            f"Retrieved {len(reconciled_facts)} applicable facts from {len(supporting_evidence)} authoritative sources."
        )

        return PersonalizedRecommendation(
            recommendation_id=rec_id,
            status="expert_escalation_required" if expert_req else "completed",
            query_summary={"text": query, "language": language, "intent": "agricultural_advisory"},
            context_used={
                "crop": crop,
                "crop_stage": crop_stage,
                "state": state,
                "weather": weather_context or {},
                "image_diagnosis_available": bool(image_diagnosis),
            },
            assessment={
                "summary": summary_text,
                "likely_issue": issue_name,
                "diagnosis_status": diag_status,
                "severity_stage": severity_stage,
            },
            actions=sanitized_actions,
            precautions=precautions,
            missing_information=unique_missing,
            clarifying_questions=clarifying_qs,
            supporting_evidence=supporting_evidence,
            confidence_assessment={"level": conf_level, "rationale": conf_rationale},
            expert_referral_required=expert_req,
            expert_referral_details=expert_details,
            response_language=language,
        )
