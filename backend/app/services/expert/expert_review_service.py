"""Core Human-in-the-Loop Agricultural Expert Review Service."""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.analytics import AdvisoryCase
from backend.app.schemas.expert import (
    CaseDossier,
    CaseReviewStatus,
    DiagnosisDecision,
    ExpertAccountStatus,
    ExpertDecisionSubmission,
    ExpertDiagnosisAssessment,
    ExpertProfile,
    ExpertRecommendationVersion,
    ExpertReviewComment,
    ExpertReviewPolicyResult,
    FarmerBadgeStatus,
    FarmerVerificationBadgeInfo,
    KnowledgeImprovementProposal,
    RecommendationDecision,
    ReviewEligibility,
    ReviewPriority,
    SecondaryReviewSubmission,
)
from backend.app.services.expert.expert_assignment_service import ExpertAssignmentService
from backend.app.services.expert.review_policy_engine import ReviewPolicyEngine

logger = get_logger("ExpertReviewService")


class ExpertReviewService:
    """Manages the full lifecycle of agricultural case reviews, dossiers, and expert governance."""

    def __init__(
        self,
        repo: FarmDatabaseRepository,
        policy_engine: Optional[ReviewPolicyEngine] = None,
        assignment_service: Optional[ExpertAssignmentService] = None,
    ):
        self.repo = repo
        self.policy_engine = policy_engine or ReviewPolicyEngine()
        self.assignment_service = assignment_service or ExpertAssignmentService(repo)

    def route_advisory_case_for_review(
        self,
        case_id: str,
        diagnosis_data: Optional[Dict[str, Any]] = None,
        recommendation_data: Optional[Dict[str, Any]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        field_officer_escalation: bool = False,
        farmer_requested: bool = False,
        missing_facts: bool = False,
        auto_assign: bool = True,
    ) -> Tuple[Optional[str], ExpertReviewPolicyResult]:
        """Evaluate and route an advisory case to the expert review queue."""
        # 1. Run policy engine evaluation
        policy_result = self.policy_engine.evaluate_case(
            case_id=case_id,
            diagnosis_data=diagnosis_data,
            recommendation_data=recommendation_data,
            weather_data=weather_data,
            field_officer_escalation=field_officer_escalation,
            farmer_requested=farmer_requested,
            missing_facts=missing_facts,
        )

        # Ineligible cases don't need queue entry unless farmer or officer explicitly asked
        if policy_result.eligibility == ReviewEligibility.INELIGIBLE and not (farmer_requested or field_officer_escalation):
            return None, policy_result

        # Check if case is already in review queue
        existing = self.repo.get_review_case_by_advisory_id(case_id)
        if existing:
            return existing["review_id"], policy_result

        # Calculate SLA deadline based on priority tier
        now = datetime.now()
        sla_hours = {
            ReviewPriority.CRITICAL: 12,
            ReviewPriority.HIGH: 24,
            ReviewPriority.MEDIUM: 48,
            ReviewPriority.LOW: 72,
        }.get(policy_result.priority, 48)
        sla_deadline = (now + timedelta(hours=sla_hours)).isoformat()

        # Retrieve advisory case to inspect crop and location for matching
        advisory = self.repo.get_advisory_case(case_id)
        crop = advisory.crop if advisory else "General"
        state = advisory.state if advisory else "General"

        assigned_expert_id = None
        if auto_assign:
            best_expert = self.assignment_service.find_best_expert_match(crop=crop, state=state)
            if best_expert:
                assigned_expert_id = best_expert.expert_id

        review_id = self.repo.create_review_case(
            case_id=case_id,
            priority=policy_result.priority.value,
            eligibility=policy_result.eligibility.value,
            trigger_reasons=policy_result.reasons,
            requires_dual_review=policy_result.requires_dual_review,
            sla_deadline=sla_deadline,
            assigned_expert_id=assigned_expert_id,
        )

        logger.info(
            f"Advisory case {case_id} routed to review queue (review_id={review_id}, priority={policy_result.priority.value}, assigned={assigned_expert_id})"
        )
        return review_id, policy_result

    def build_case_dossier(self, case_id: str) -> CaseDossier:
        """Assemble complete multi-source dossier for agronomist review."""
        case = self.repo.get_advisory_case(case_id)
        if not case:
            raise ValueError(f"Advisory case '{case_id}' not found.")

        # Review record
        rev = self.repo.get_review_case_by_advisory_id(case_id)
        review_id = rev["review_id"] if rev else None
        review_priority = ReviewPriority(rev["priority"]) if rev else ReviewPriority.MEDIUM
        review_status = CaseReviewStatus(rev["status"]) if rev else CaseReviewStatus.QUEUED
        triggers = json.loads(rev["trigger_reasons_json"]) if rev and rev.get("trigger_reasons_json") else []
        assigned_expert_id = rev.get("assigned_expert_id") if rev else None
        secondary_expert_id = rev.get("secondary_expert_id") if rev else None
        created_at = rev.get("created_at") if rev else case.created_at
        sla_deadline = rev.get("sla_deadline") if rev else None

        assigned_expert_name = None
        if assigned_expert_id:
            exp = self.repo.get_expert_profile(assigned_expert_id)
            if exp:
                assigned_expert_name = exp.full_name

        # Farmer profile & masked phone
        farmer = None
        farmer_masked_phone = None
        pref_language = "hi-IN"
        if case.farmer_id:
            farmer = self.repo.get_farmer_profile(case.farmer_id)
            if farmer:
                pref_language = farmer.preferred_language
                if farmer.phone_number:
                    raw = farmer.phone_number.strip()
                    if len(raw) >= 10:
                        farmer_masked_phone = f"{raw[:3]}****{raw[-3:]}"
                    else:
                        farmer_masked_phone = "***"

        # Farm & Plot details
        farm_name = None
        plot_size_acres = None
        soil_type = None
        irrigation_type = None
        if case.farm_id:
            farm = self.repo.get_farm(case.farm_id)
            if farm:
                farm_name = farm.farm_name
                plot_size_acres = farm.total_area_acres
            plots = self.repo.list_plots_by_farm(case.farm_id)
            if plots:
                plot = plots[0]
                soil_p = self.repo.get_soil_profile(plot.plot_id)
                if soil_p:
                    soil_type = soil_p.soil_type.value if hasattr(soil_p.soil_type, "value") else str(soil_p.soil_type)
                irrig_p = self.repo.get_irrigation_profile(plot.plot_id)
                if irrig_p:
                    irrigation_type = (
                        irrig_p.primary_method.value
                        if hasattr(irrig_p.primary_method, "value")
                        else str(irrig_p.primary_method)
                    )

        # Vision diagnosis telemetry
        img_analysis = case.image_analysis or {}
        img_url = img_analysis.get("image_path") or img_analysis.get("image_url")
        gradcam = img_analysis.get("gradcam_heatmap_path")
        ai_disease = (img_analysis.get("disease_name") or img_analysis.get("primary_disease")) if img_analysis else None
        ai_conf = float(img_analysis["confidence"]) if (img_analysis and img_analysis.get("confidence") is not None) else None
        ai_sec = img_analysis.get("secondary_diseases", []) if img_analysis else []
        ai_sev = case.risk or "Medium"

        # Microclimate weather telemetry
        weather_summary = case.weather_summary if case.weather_summary else "Weather data not recorded for this case."
        hazard_level = "LOW"
        if "warning" in weather_summary.lower() or "alert" in weather_summary.lower():
            hazard_level = "HIGH"

        # Field Officer observations
        follow_ups = self.repo.list_case_follow_ups(case_id)
        latest_officer_note = None
        officer_name = None
        officer_visited_at = None
        if follow_ups:
            latest_fu = follow_ups[0]
            latest_officer_note = f"{latest_fu.notes} (Action Taken: {latest_fu.action_taken or 'None'})"
            officer_name = latest_fu.officer_name
            officer_visited_at = latest_fu.created_at

        # Immutable original AI recommendation
        original_ai_recommendation = {
            "advisory_en": case.advisory_en,
            "advisory_hi": case.advisory_hi,
            "steps": case.steps,
            "disease_detected": ai_disease,
            "confidence": ai_conf,
            "severity": ai_sev,
        }

        # ICAR & CIBRC Grounding
        icar_facts = []
        try:
            from backend.app.repositories.fact_repository import FactRepository
            fact_repo = FactRepository.get_instance()
            real_facts = fact_repo.filter_facts(subject=case.crop) if case.crop else []
            for rf in real_facts[:5]:
                src = rf.evidence[0].organization if rf.evidence else "ICAR Repository"
                icar_facts.append({
                    "topic": f"{rf.category.title()}: {rf.predicate}",
                    "source": src,
                    "recommendation": rf.value,
                })
        except Exception as e:
            logger.warning(f"Could not load ICAR facts for crop {case.crop}: {e}")
            icar_facts = []

        return CaseDossier(
            case_id=case.case_id,
            farmer_id=case.farmer_id,
            farmer_name=case.farmer_name,
            farmer_masked_phone=farmer_masked_phone,
            farm_id=case.farm_id,
            farm_name=farm_name,
            state=case.state,
            district=case.district,
            village=case.location,
            plot_size_acres=plot_size_acres,
            soil_type=soil_type,
            irrigation_type=irrigation_type,
            crop_name=case.crop,
            symptoms=case.symptoms,
            preferred_language=pref_language,
            image_url_or_path=img_url,
            gradcam_heatmap_path=gradcam,
            ai_primary_disease=ai_disease,
            ai_confidence=ai_conf,
            ai_secondary_diseases=ai_sec,
            ai_severity=ai_sev,
            weather_summary=weather_summary,
            weather_hazard_level=hazard_level,
            icar_retrieved_facts=icar_facts,
            original_ai_recommendation=original_ai_recommendation,
            field_officer_notes=latest_officer_note,
            officer_name=officer_name,
            officer_visited_at=officer_visited_at,
            review_id=review_id,
            review_priority=review_priority,
            review_status=review_status,
            eligibility_reasons=triggers,
            assigned_expert_id=assigned_expert_id,
            assigned_expert_name=assigned_expert_name,
            secondary_expert_id=secondary_expert_id,
            created_at=created_at,
            sla_deadline=sla_deadline,
        )

    def claim_case(self, review_id: str, expert_id: str) -> bool:
        """Allow a verified agronomist to claim an unassigned queued case."""
        expert = self.repo.get_expert_profile(expert_id)
        if not expert or expert.verification_status != ExpertAccountStatus.ACTIVE_VERIFIED:
            raise PermissionError(f"Expert '{expert_id}' is not active or verified to claim cases.")
        return self.repo.claim_review_case(review_id, expert_id)

    def submit_expert_decision(
        self,
        submission: ExpertDecisionSubmission,
        current_expert_id: str,
    ) -> Dict[str, Any]:
        """Submit comprehensive diagnosis and recommendation decision by agronomist."""
        rev = self.repo.get_review_case(submission.review_id)
        if not rev:
            raise ValueError(f"Review case '{submission.review_id}' not found.")

        # Ensure the submitting expert is assigned or claiming
        if rev.get("assigned_expert_id") and rev["assigned_expert_id"] != current_expert_id:
            raise PermissionError("Case is assigned to another agronomist.")

        expert = self.repo.get_expert_profile(current_expert_id)
        if not expert or expert.verification_status != ExpertAccountStatus.ACTIVE_VERIFIED:
            raise PermissionError("Agronomist is not active and verified.")

        case_id = rev["case_id"]
        now_str = datetime.now().isoformat()

        # 1. Save Diagnosis Assessment
        assessment = ExpertDiagnosisAssessment(
            assessment_id=f"diag_{uuid.uuid4().hex[:12]}",
            review_id=submission.review_id,
            case_id=case_id,
            expert_id=current_expert_id,
            decision=submission.diagnosis_decision,
            revised_disease=submission.revised_disease,
            pathology_notes=submission.pathology_notes,
            assessed_at=now_str,
        )
        self.repo.save_diagnosis_assessment(assessment)

        # 2. Check previous recommendation versions
        existing_versions = self.repo.list_recommendation_versions(submission.review_id)
        version_num = len(existing_versions) + 1

        # Check if chemical intervention triggers secondary / dual review
        requires_secondary = False
        if submission.chemical_control:
            for chem in submission.chemical_control:
                c_name = chem.chemical_name.lower()
                from backend.app.services.expert.review_policy_engine import HIGH_RISK_CHEMICALS
                if any(hr in c_name for hr in HIGH_RISK_CHEMICALS) or chem.pre_harvest_interval_days > 14:
                    requires_secondary = True

        rec_version = ExpertRecommendationVersion(
            version_id=f"recver_{uuid.uuid4().hex[:12]}",
            review_id=submission.review_id,
            case_id=case_id,
            expert_id=current_expert_id,
            version_number=version_num,
            decision=submission.recommendation_decision,
            cultural_management=submission.cultural_management,
            biological_control=submission.biological_control,
            chemical_control=submission.chemical_control,
            pre_harvest_interval_days=submission.pre_harvest_interval_days,
            safety_disclaimers=[
                "Wear protective gloves and face mask during chemical mixing and spray.",
                f"Strictly observe Pre-Harvest Interval (PHI) of {submission.pre_harvest_interval_days or 7} days.",
                "Do not spray during high winds or immediate impending rain.",
            ],
            rationale=submission.rationale,
            requires_secondary_review=requires_secondary,
            created_at=now_str,
        )
        self.repo.save_recommendation_version(rec_version)

        # 3. If diagnosis changed, submit Knowledge Improvement Proposal automatically
        if submission.diagnosis_decision in (DiagnosisDecision.MODIFIED, DiagnosisDecision.REJECTED_NON_PATHOLOGICAL):
            dossier = self.build_case_dossier(case_id)
            proposal = KnowledgeImprovementProposal(
                proposal_id=f"kip_{uuid.uuid4().hex[:12]}",
                case_id=case_id,
                expert_id=current_expert_id,
                original_ai_disease=dossier.ai_primary_disease,
                expert_confirmed_disease=submission.revised_disease or "Non-Pathological / Physiological",
                crop=dossier.crop_name,
                correction_category="differential_diagnosis_override",
                reason_for_override=submission.pathology_notes,
                suggested_kb_fact_or_rule=submission.knowledge_improvement_note
                or f"When symptoms match {submission.pathology_notes[:60]}, consider {submission.revised_disease}.",
                created_at=now_str,
            )
            self.repo.create_knowledge_proposal(proposal)

        # 4. Handle Dual Review or Final Status Transition
        if requires_secondary:
            # Assign secondary reviewer
            dossier = self.build_case_dossier(case_id)
            secondary = self.assignment_service.find_secondary_reviewer(
                primary_expert_id=current_expert_id,
                crop=dossier.crop_name,
                state=dossier.state,
            )
            sec_id = secondary.expert_id if secondary else None
            self.repo.update_review_case(
                submission.review_id,
                {
                    "status": CaseReviewStatus.SECONDARY_REVIEW_REQUIRED.value,
                    "secondary_expert_id": sec_id,
                    "reviewed_at": now_str,
                },
            )
            return {
                "review_id": submission.review_id,
                "status": CaseReviewStatus.SECONDARY_REVIEW_REQUIRED.value,
                "requires_secondary_review": True,
                "secondary_expert_id": sec_id,
                "message": "High-risk chemical intervention requires mandatory dual review by a second agronomist.",
            }
        else:
            final_status = (
                CaseReviewStatus.MODIFIED.value
                if submission.recommendation_decision != RecommendationDecision.APPROVED_AS_IS
                else CaseReviewStatus.APPROVED.value
            )
            self.repo.update_review_case(
                submission.review_id,
                {
                    "status": final_status,
                    "reviewed_at": now_str,
                },
            )
            # Update advisory case status
            self.repo.update_advisory_case_status(
                case_id=case_id,
                status="expert_verified" if final_status == CaseReviewStatus.APPROVED.value else "expert_modified",
            )
            # Update expert stats
            self.repo.update_expert_profile(
                current_expert_id,
                {
                    "active_caseload": max(0, expert.active_caseload - 1),
                    "total_reviewed_cases": expert.total_reviewed_cases + 1,
                },
            )
            return {
                "review_id": submission.review_id,
                "status": final_status,
                "requires_secondary_review": False,
                "message": "Case review successfully finalized and guidance released to farmer.",
            }

    def submit_secondary_review(
        self,
        submission: SecondaryReviewSubmission,
        secondary_expert_id: str,
    ) -> Dict[str, Any]:
        """Process dual-review sign-off from secondary independent agronomist."""
        rev = self.repo.get_review_case(submission.review_id)
        if not rev:
            raise ValueError(f"Review case '{submission.review_id}' not found.")

        # Enforce that status requires secondary review
        if rev["status"] != CaseReviewStatus.SECONDARY_REVIEW_REQUIRED.value:
            raise ValueError(f"Review case '{submission.review_id}' does not require secondary review (current status: {rev['status']}).")

        # Must have an explicitly assigned secondary expert
        if not rev.get("secondary_expert_id"):
            raise ValueError("Secondary expert has not been assigned to this case yet.")

        # User submitting must match assigned secondary expert
        if rev["secondary_expert_id"] != secondary_expert_id:
            raise PermissionError("User is not the designated secondary reviewer for this case.")

        # Secondary reviewer cannot be the primary assigned expert
        if rev.get("assigned_expert_id") and secondary_expert_id == rev["assigned_expert_id"]:
            raise PermissionError("Primary reviewer cannot perform their own secondary review.")

        # Verify that secondary expert is active and verified
        sec_expert = self.repo.get_expert_profile(secondary_expert_id)
        if not sec_expert or sec_expert.verification_status != ExpertAccountStatus.ACTIVE_VERIFIED:
            raise PermissionError("Secondary expert must be an active, verified expert.")

        now_str = datetime.now().isoformat()
        case_id = rev["case_id"]

        if submission.approved:
            final_status = CaseReviewStatus.APPROVED.value
            self.repo.update_review_case(
                submission.review_id,
                {
                    "status": final_status,
                    "secondary_reviewed_at": now_str,
                },
            )
            self.repo.update_advisory_case_status(case_id, "expert_verified")
            return {
                "review_id": submission.review_id,
                "status": final_status,
                "message": "Secondary review approved; prescription verified for safe field application.",
            }
        else:
            final_status = CaseReviewStatus.MODIFIED.value
            self.repo.update_review_case(
                submission.review_id,
                {
                    "status": final_status,
                    "secondary_reviewed_at": now_str,
                },
            )
            self.repo.update_advisory_case_status(case_id, "expert_modified")
            return {
                "review_id": submission.review_id,
                "status": final_status,
                "message": f"Secondary review required modifications: {submission.secondary_notes}",
            }

    def get_farmer_verification_badge(self, case_id: str) -> FarmerVerificationBadgeInfo:
        """Provide verifiable badge, accredited expert provenance, and multilingual voice summary."""
        case = self.repo.get_advisory_case(case_id)
        if not case:
            raise ValueError(f"Case '{case_id}' not found.")

        rev = self.repo.get_review_case_by_advisory_id(case_id)
        if not rev:
            return FarmerVerificationBadgeInfo(
                case_id=case_id,
                badge=FarmerBadgeStatus.AI_GENERATED,
                badge_label_en="AI-Generated Guidance (Autonomous)",
                badge_label_hi="एआई-जनित परामर्श (स्वचालित)",
                confirmed_diagnosis=case.crop + " Assessment",
                verified_recommendations={"advisory_en": case.advisory_en, "advisory_hi": case.advisory_hi},
                audio_tts_text_en="This recommendation was generated by AgriGuardian AI based on automated visual and agronomic analysis.",
                audio_tts_text_hi="यह परामर्श एग्रीगार्डियन एआई द्वारा स्वचालित विश्लेषण के आधार पर तैयार किया गया है।",
            )

        status_str = rev["status"]
        expert = self.repo.get_expert_profile(rev["assigned_expert_id"]) if rev.get("assigned_expert_id") else None
        expert_name = expert.full_name if expert else None
        institution = expert.institution if expert else None

        latest_diag = self.repo.get_diagnosis_assessment(rev["review_id"])
        latest_rec = self.repo.get_latest_recommendation_version(rev["review_id"])

        diag_name = (
            latest_diag.revised_disease
            if latest_diag and latest_diag.revised_disease
            else (case.image_analysis.get("disease_name") if (case.image_analysis and case.image_analysis.get("disease_name")) else "Observed Condition")
        )

        if status_str in ("queued", "claimed", "in_review", "secondary_review_required"):
            badge = FarmerBadgeStatus.AWAITING_EXPERT_REVIEW
            label_en = "Awaiting Expert Review"
            label_hi = "कृषि विशेषज्ञ समीक्षा प्रतीक्षित"
            if institution and expert_name:
                tts_en = f"Your case has been routed to accredited agricultural expert {expert_name} from {institution}. Please await human confirmation before spraying chemical treatments."
                tts_hi = f"आपका मामला {institution} के कृषि विशेषज्ञ {expert_name} को भेजा गया है। रासायनिक उपचार छिड़कने से पहले विशेषज्ञ की पुष्टि की प्रतीक्षा करें।"
            else:
                tts_en = "Your case has been routed to an accredited agricultural expert. Please await human confirmation before spraying chemical treatments."
                tts_hi = "आपका मामला कृषि विशेषज्ञ को भेजा गया है। रासायनिक उपचार छिड़कने से पहले विशेषज्ञ की पुष्टि की प्रतीक्षा करें।"
        elif status_str == "approved":
            badge = FarmerBadgeStatus.EXPERT_VERIFIED
            name_disp = expert_name or "Accredited Agronomist"
            inst_disp = f" ({institution})" if institution else ""
            label_en = f"Verified by {name_disp}{inst_disp}"
            label_hi = f"विशेषज्ञ सत्यापित: {name_disp}{inst_disp}"
            tts_en = f"Diagnosis of {diag_name} and recommended interventions have been thoroughly verified by agronomist {name_disp}."
            tts_hi = f"कृषि विशेषज्ञ {name_disp} ने {diag_name} के निदान और अनुशंसित उपचार की पूर्ण पुष्टि की है।"
        elif status_str == "modified":
            badge = FarmerBadgeStatus.EXPERT_MODIFIED
            name_disp = expert_name or "Accredited Agronomist"
            inst_disp = f" ({institution})" if institution else ""
            label_en = f"Expert-Reviewed with Modifications ({name_disp})"
            label_hi = f"विशेषज्ञ द्वारा संशोधित परामर्श ({name_disp})"
            tts_en = f"Agronomist {name_disp} reviewed your case and updated the treatment protocol to ensure safe crop recovery."
            tts_hi = f"कृषि विशेषज्ञ {name_disp} ने आपके मामले की समीक्षा की है और फसल सुरक्षा हेतु उपचार में आवश्यक सुधार किए हैं।"
        else:
            badge = FarmerBadgeStatus.AWAITING_INFORMATION
            label_en = "Awaiting Additional Details"
            label_hi = "अतिरिक्त जानकारी प्रतीक्षित"
            tts_en = "The agronomist has requested clearer leaf photos or soil details to finalize your prescription."
            tts_hi = "सटीक परामर्श देने के लिए कृषि विशेषज्ञ ने पत्ती की साफ तस्वीर या मिट्टी के विवरण का अनुरोध किया है।"

        rec_dict = {
            "cultural_management": latest_rec.cultural_management if latest_rec else case.steps,
            "biological_control": [b.model_dump() for b in latest_rec.biological_control] if latest_rec else [],
            "chemical_control": [c.model_dump() for c in latest_rec.chemical_control] if latest_rec else [],
            "safety_disclaimers": latest_rec.safety_disclaimers if latest_rec else [],
        }

        return FarmerVerificationBadgeInfo(
            case_id=case_id,
            badge=badge,
            badge_label_en=label_en,
            badge_label_hi=label_hi,
            expert_name=expert_name,
            expert_institution=institution,
            verified_at=rev.get("reviewed_at") or rev.get("updated_at"),
            confirmed_diagnosis=str(diag_name),
            verified_recommendations=rec_dict,
            audio_tts_text_en=tts_en,
            audio_tts_text_hi=tts_hi,
        )

