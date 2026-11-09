"""Schemas for atomic, verified agricultural facts and knowledge catalog records."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.evidence import EvidenceExcerpt


class AtomicFact(BaseModel):
    """Schema for an atomic, verifiable agricultural knowledge fact."""
    fact_id: str = Field(..., description="Unique fact identifier (e.g., 'FCT-TOM-BLIGHT-CULT-01')")
    category: str = Field(
        ...,
        description="Domain category: 'disease_management', 'pest_management', 'irrigation', 'soil_health', 'nutrient_management', 'weather_risk', 'agronomic_practice'"
    )
    subject: str = Field(..., description="Core agricultural entity (e.g., 'tomato', 'wheat', 'rice')")
    predicate: str = Field(
        ...,
        description="Semantic relation: 'has_critical_irrigation_stage', 'managed_by_cultural_practice', 'managed_by_biological_agent', 'chemical_precaution_recommended', 'susceptible_to_weather_risk'"
    )
    value: str = Field(..., description="Actionable fact statement or instruction")
    scientific_name: Optional[str] = Field(None, description="Botanical or pathogen scientific nomenclature")
    conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Prerequisite conditions: crop_stage, soil_type, weather_thresholds, etc."
    )
    geographic_scope: Dict[str, Any] = Field(
        default_factory=lambda: {"country": "India", "states": []},
        description="Geographic and agroclimatic applicability boundaries"
    )
    evidence: List[EvidenceExcerpt] = Field(
        default_factory=list,
        description="Supporting verified evidence excerpts and source citations"
    )
    verification_status: str = Field(
        "approved",
        description="Lifecycle status: 'approved', 'draft', 'pending_review', 'expired', 'rejected'"
    )
    safety_classification: str = Field(
        "low_risk_cultural",
        description="Safety tier: 'low_risk_cultural', 'biological_control', 'chemical_restricted', 'advisory_only'"
    )
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid_from: Optional[str] = Field(None, description="Start date of validity (YYYY-MM-DD)")
    valid_until: Optional[str] = Field(None, description="Expiration or sunset date (YYYY-MM-DD)")
    review_due_at: Optional[str] = Field(None, description="Scheduled specialist review due date")

    @property
    def crop(self) -> str:
        """Alias for subject entity."""
        return self.subject

    @property
    def authority_weight(self) -> float:
        """Derive authority weight from supporting evidence tiers."""
        for ev in self.evidence:
            if "Tier-1" in getattr(ev, "authority_tier", "") or "ICAR" in ev.organization or "CIBRC" in ev.organization or "IMD" in ev.organization:
                return 1.4
            elif "Tier-2" in getattr(ev, "authority_tier", "") or "University" in ev.organization or "TNAU" in ev.organization:
                return 1.2
        return 1.0

    @property
    def confidence_score(self) -> float:
        """Primary confidence score from verified evidence."""
        if self.evidence:
            return self.evidence[0].confidence_score
        return 0.95
