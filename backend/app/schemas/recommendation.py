"""Schemas for evidence-grounded, personalized agricultural recommendations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.evidence import EvidenceExcerpt


class ActionItem(BaseModel):
    """Schema for individual actionable agricultural advice step."""
    priority: str = Field(..., description="'immediate', 'secondary', 'preventive', 'monitoring'")
    action: str = Field(..., description="Actionable guideline in clear, farmer-accessible phrasing")
    reason: str = Field(..., description="Underlying agronomic justification explaining why this action is recommended")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting EvidenceExcerpt items")
    applicability: str = Field("General applicability for verified conditions", description="Applicability boundary")
    safety_level: str = Field("low", description="'low', 'medium', 'restricted_chemical', 'expert_referral'")
    regulatory_warning: Optional[str] = Field(None, description="CIBRC safety or waiting period precaution if chemical")


class PersonalizedRecommendation(BaseModel):
    """Comprehensive schema for personalized, evidence-grounded agricultural advisory."""
    recommendation_id: str = Field(..., description="Unique recommendation session trace ID")
    status: str = Field(
        "completed",
        description="'completed', 'clarification_needed', 'expert_escalation_required', 'insufficient_evidence'"
    )
    query_summary: Dict[str, Any] = Field(
        ...,
        description="Farmer's original query, detected language, and classified intent"
    )
    context_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Foliar vision diagnosis, weather context, market price, growth stage, location"
    )
    assessment: Dict[str, Any] = Field(
        ...,
        description="Factual assessment: likely issue, clinical staging, diagnosis status ('confirmed', 'suspected', 'uncertain')"
    )
    actions: List[ActionItem] = Field(
        default_factory=list,
        description="Ordered list of evidence-backed practical actions"
    )
    precautions: List[str] = Field(
        default_factory=list,
        description="Safety, ecological, and chemical application precautions"
    )
    missing_information: List[str] = Field(
        default_factory=list,
        description="Unspecified farm variables (e.g. soil pH, exact sowing date, recent sprays)"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Farmer-directed clarifying questions to refine recommendation if ambiguous"
    )
    supporting_evidence: List[EvidenceExcerpt] = Field(
        default_factory=list,
        description="Verified source citations with titles, organizations, and excerpts"
    )
    confidence_assessment: Dict[str, Any] = Field(
        ...,
        description="Epistemic confidence level ('High', 'Moderate', 'Low') with transparent rationale"
    )
    expert_referral_required: bool = Field(
        False,
        description="True if symptoms require physical inspection by Krishi Vigyan Kendra (KVK)"
    )
    expert_referral_details: Optional[str] = Field(
        None,
        description="Actionable contact guidance for regional agricultural extension officers"
    )
    knowledge_manifest_version: str = Field("3.0.0", description="Version of knowledge base utilized")
    response_language: str = Field("hi-IN", description="Language of generated guidance")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

