"""Pydantic schemas and domain models for Phase 8 Expert Verification & Human-in-the-Loop AI."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExpertSpecializationDomain(str, Enum):
    """Agricultural domain areas for expert specialization."""
    PLANT_PATHOLOGY = "plant_pathology"
    ENTOMOLOGY = "entomology"
    SOIL_SCIENCE = "soil_science"
    AGRONOMY = "agronomy"
    HORTICULTURE = "horticulture"
    CROP_PHYSIOLOGY = "crop_physiology"
    GENERAL_AGRONOMY = "general_agronomy"


class ExpertAccountStatus(str, Enum):
    """Lifecycle status of an agricultural expert account."""
    PENDING_VERIFICATION = "pending_verification"
    ACTIVE_VERIFIED = "active_verified"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


class ReviewTriggerType(str, Enum):
    """Triggers determining human-in-the-loop expert review eligibility."""
    LOW_MODEL_CONFIDENCE = "low_model_confidence"
    CONFLICTING_PREDICTIONS = "conflicting_predictions"
    HIGH_SEVERITY = "high_severity"
    CHEMICAL_RECOMMENDATION = "chemical_recommendation"
    MISSING_ICAR_FACTS = "missing_icar_facts"
    FIELD_OFFICER_ESCALATION = "field_officer_escalation"
    FARMER_REQUEST = "farmer_request"
    RANDOM_AUDIT = "random_audit"


class ReviewEligibility(str, Enum):
    """Eligibility classification for expert review."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    INELIGIBLE = "ineligible"


class ReviewPriority(str, Enum):
    """SLA priority tiers for expert case reviews."""
    CRITICAL = "critical"  # SLA < 12h
    HIGH = "high"          # SLA < 24h
    MEDIUM = "medium"      # SLA < 48h
    LOW = "low"            # SLA < 72h


class CaseReviewStatus(str, Enum):
    """Workflow state of an expert review case."""
    QUEUED = "queued"
    CLAIMED = "claimed"
    IN_REVIEW = "in_review"
    SECONDARY_REVIEW_REQUIRED = "secondary_review_required"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    INFO_REQUESTED = "info_requested"


class DiagnosisDecision(str, Enum):
    """Expert assessment of the AI computer vision diagnosis."""
    CONFIRMED = "confirmed"
    MODIFIED = "modified"
    REJECTED_NON_PATHOLOGICAL = "rejected_non_pathological"
    INSUFFICIENT_IMAGE_QUALITY = "insufficient_image_quality"


class RecommendationDecision(str, Enum):
    """Expert assessment and action on AI treatment recommendation."""
    APPROVED_AS_IS = "approved_as_is"
    MODIFIED_DOSAGE_TIMING = "modified_dosage_timing"
    SUBSTITUTED_ORGANIC_BIO = "substituted_organic_bio"
    REPLACED_FORMULATION = "replaced_formulation"
    CANCELLED_INTERVENTION = "cancelled_intervention"


class FarmerBadgeStatus(str, Enum):
    """Farmer-facing verification badge indicating trust and human sign-off status."""
    AI_GENERATED = "ai_generated"
    AWAITING_EXPERT_REVIEW = "awaiting_expert_review"
    EXPERT_VERIFIED = "expert_verified"
    EXPERT_MODIFIED = "expert_modified"
    AWAITING_INFORMATION = "awaiting_information"


# --- Expert Profile & Specializations ---

class ExpertSpecializationItem(BaseModel):
    """Agricultural domain areas, crops, locations, and languages handled by an expert."""
    crop_types: List[str] = Field(default_factory=list, description="Crops e.g. Wheat, Rice, Tomato, Cotton")
    domains: List[ExpertSpecializationDomain] = Field(default_factory=list)
    states: List[str] = Field(default_factory=list, description="States e.g. Haryana, Punjab, Maharashtra")
    districts: List[str] = Field(default_factory=list, description="Specific districts if applicable")
    languages: List[str] = Field(default_factory=list, description="Languages e.g. hi-IN, en-IN, pa-IN")


class ExpertCredentialRecord(BaseModel):
    """Verified academic, institutional, or government credential for an agronomist."""
    credential_id: str
    expert_id: str
    institution_name: str
    degree_or_certification: str
    registration_number: str
    document_url: Optional[str] = None
    issued_date: Optional[str] = None
    expiry_date: Optional[str] = None
    is_verified: bool = False
    verified_by_admin: Optional[str] = None
    verified_at: Optional[str] = None


class ExpertCredentialCreate(BaseModel):
    institution_name: str = Field(..., min_length=2)
    degree_or_certification: str = Field(..., min_length=2)
    registration_number: str = Field(..., min_length=2)
    document_url: Optional[str] = None
    issued_date: Optional[str] = None
    expiry_date: Optional[str] = None


class ExpertProfile(BaseModel):
    """Complete profile for an Agricultural Expert / Agronomist."""
    expert_id: str
    user_id: str
    full_name: str
    email: Optional[str] = None
    phone_number: Optional[str] = None
    institution: str = Field(..., description="e.g. ICAR, PAU, CCS HAU, KVK")
    designation: str = Field(default="Agronomist")
    experience_years: int = Field(default=0, ge=0)
    specializations: ExpertSpecializationItem = Field(default_factory=ExpertSpecializationItem)
    credentials: List[ExpertCredentialRecord] = Field(default_factory=list)
    verification_status: ExpertAccountStatus = Field(default=ExpertAccountStatus.PENDING_VERIFICATION)
    verified_by_admin_id: Optional[str] = None
    verified_at: Optional[str] = None
    active_caseload: int = Field(default=0, ge=0)
    total_reviewed_cases: int = Field(default=0, ge=0)
    accuracy_rating: float = Field(default=5.0, ge=0.0, le=5.0)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExpertProfileCreate(BaseModel):
    full_name: str = Field(..., min_length=2)
    email: Optional[str] = None
    phone_number: Optional[str] = None
    institution: str = Field(..., min_length=2)
    designation: str = Field(default="Agronomist")
    experience_years: int = Field(default=0, ge=0)
    specializations: ExpertSpecializationItem = Field(default_factory=ExpertSpecializationItem)
    initial_credential: Optional[ExpertCredentialCreate] = None


class ExpertProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    institution: Optional[str] = None
    designation: Optional[str] = None
    experience_years: Optional[int] = None
    specializations: Optional[ExpertSpecializationItem] = None


class ExpertProvisionRequest(BaseModel):
    """Admin request payload to provision an accredited agronomist."""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    email: Optional[str] = None
    phone_number: Optional[str] = None
    institution: str = Field(..., min_length=2)
    designation: str = "Agronomist"
    experience_years: int = 0
    crop_types: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    states: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=lambda: ["hi-IN", "en-IN"])
    initial_credential_institution: Optional[str] = None
    initial_credential_degree: Optional[str] = None
    initial_credential_reg_num: Optional[str] = None


class ExpertVerificationAction(BaseModel):
    """Admin action to approve, reject, or suspend an expert."""
    action: ExpertAccountStatus
    rejection_reason: Optional[str] = None
    admin_notes: Optional[str] = None


# --- Policy Engine & Eligibility ---

class ExpertReviewPolicyResult(BaseModel):
    """Evaluation result from the Expert Review Policy Engine."""
    case_id: str
    eligibility: ReviewEligibility
    priority: ReviewPriority
    triggers: List[ReviewTriggerType]
    reasons: List[str]
    policy_version: str = "v1.0-icar-cibrc"
    requires_dual_review: bool = False
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# --- Case Dossier & Review Records ---

class CaseDossier(BaseModel):
    """Comprehensive agricultural case dossier presented to the verifying agronomist."""
    case_id: str
    farmer_id: Optional[str] = None
    farmer_name: str
    farmer_masked_phone: Optional[str] = None
    farm_id: Optional[str] = None
    farm_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    plot_size_acres: Optional[float] = None
    soil_type: Optional[str] = None
    irrigation_type: Optional[str] = None
    crop_name: str
    variety: Optional[str] = None
    sowing_date: Optional[str] = None
    growth_stage: Optional[str] = None
    symptoms: Optional[str] = None
    farmer_notes: Optional[str] = None
    preferred_language: str = "hi-IN"
    
    # Vision & ML diagnosis
    image_url_or_path: Optional[str] = None
    gradcam_heatmap_path: Optional[str] = None
    ai_primary_disease: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_secondary_diseases: List[Dict[str, Any]] = Field(default_factory=list)
    ai_severity: str = "Low"

    # Microclimate & weather context
    weather_summary: Optional[str] = None
    weather_hazard_level: str = "LOW"
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall_mm: Optional[float] = None

    # Agronomic Evidence & Guardrails
    icar_retrieved_facts: List[Dict[str, Any]] = Field(default_factory=list)
    cibrc_guidelines: List[Dict[str, Any]] = Field(default_factory=list)
    original_ai_recommendation: Dict[str, Any] = Field(default_factory=dict, description="Immutable original AI recommendation")

    # Field Officer ground observation
    field_officer_notes: Optional[str] = None
    officer_name: Optional[str] = None
    officer_visited_at: Optional[str] = None

    # Review workflow state
    review_id: Optional[str] = None
    review_priority: ReviewPriority = ReviewPriority.MEDIUM
    review_status: CaseReviewStatus = CaseReviewStatus.QUEUED
    eligibility_reasons: List[str] = Field(default_factory=list)
    assigned_expert_id: Optional[str] = None
    assigned_expert_name: Optional[str] = None
    secondary_expert_id: Optional[str] = None
    created_at: str
    sla_deadline: Optional[str] = None


class ExpertDiagnosisAssessment(BaseModel):
    """Detailed expert assessment of plant pathology and visual symptoms."""
    assessment_id: str
    review_id: str
    case_id: str
    expert_id: str
    decision: DiagnosisDecision
    revised_crop: Optional[str] = None
    revised_disease: Optional[str] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    differential_diagnosis: List[str] = Field(default_factory=list)
    pathology_notes: str = Field(..., min_length=5)
    assessed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChemicalInterventionItem(BaseModel):
    """CIBRC compliant chemical prescription."""
    chemical_name: str
    trade_name: Optional[str] = None
    cibrc_registration_number: Optional[str] = None
    dosage_per_acre: str
    dilution_water_liters: Optional[str] = None
    application_method: str = "Foliar Spray"
    pre_harvest_interval_days: int = Field(default=7, ge=0)
    waiting_period_hours: int = Field(default=24, ge=0)
    toxicity_class: str = "Class III - Slightly Hazardous (Blue/Green)"
    safety_gear_required: List[str] = Field(default_factory=lambda: ["Gloves", "Mask", "Protective Clothing"])


class BiologicalControlItem(BaseModel):
    """ICAR approved bio-control and organic formulation."""
    agent_name: str
    target_pest_or_disease: str
    dosage: str
    timing: str


class ExpertRecommendationVersion(BaseModel):
    """Immutable, versioned treatment recommendation produced by an accredited expert."""
    version_id: str
    review_id: str
    case_id: str
    expert_id: str
    version_number: int = 1
    decision: RecommendationDecision
    cultural_management: List[str] = Field(default_factory=list)
    biological_control: List[BiologicalControlItem] = Field(default_factory=list)
    chemical_control: List[ChemicalInterventionItem] = Field(default_factory=list)
    pre_harvest_interval_days: Optional[int] = None
    safety_disclaimers: List[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=5)
    requires_secondary_review: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExpertReviewComment(BaseModel):
    """Communication between experts, field officers, and administrators on a case."""
    comment_id: str
    review_id: str
    author_id: str
    author_name: str
    author_role: str
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExpertDecisionSubmission(BaseModel):
    """Payload submitted by agronomist to finalize review of a case."""
    review_id: str
    diagnosis_decision: DiagnosisDecision
    revised_disease: Optional[str] = None
    pathology_notes: str
    recommendation_decision: RecommendationDecision
    cultural_management: List[str] = Field(default_factory=list)
    biological_control: List[BiologicalControlItem] = Field(default_factory=list)
    chemical_control: List[ChemicalInterventionItem] = Field(default_factory=list)
    pre_harvest_interval_days: Optional[int] = None
    rationale: str
    farmer_summary_notes: str = Field(..., min_length=5, description="Clear summary translated or communicated to farmer")
    field_officer_instructions: Optional[str] = None
    knowledge_improvement_note: Optional[str] = None


class SecondaryReviewSubmission(BaseModel):
    """Payload submitted by secondary reviewing agronomist for dual-review cases."""
    review_id: str
    approved: bool
    secondary_notes: str = Field(..., min_length=5)
    adjusted_dosage_or_notes: Optional[str] = None


class KnowledgeImprovementProposal(BaseModel):
    """Continuous improvement feedback record proposing updates to the agronomy knowledge base."""
    proposal_id: str
    case_id: str
    expert_id: str
    original_ai_disease: str
    expert_confirmed_disease: str
    crop: str
    correction_category: str = Field(..., description="e.g. false_positive, mistaken_symptom, missing_variant")
    reason_for_override: str
    suggested_kb_fact_or_rule: str
    status: str = "pending"  # pending, reviewed, integrated, rejected
    reviewed_by_admin: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class FarmerVerificationBadgeInfo(BaseModel):
    """Status returned to farmer mobile / voice assistant with verified guidance."""
    case_id: str
    badge: FarmerBadgeStatus
    badge_label_en: str
    badge_label_hi: str
    expert_name: Optional[str] = None
    expert_institution: Optional[str] = None
    verified_at: Optional[str] = None
    confirmed_diagnosis: str
    verified_recommendations: Dict[str, Any]
    audio_tts_text_en: str
    audio_tts_text_hi: str
