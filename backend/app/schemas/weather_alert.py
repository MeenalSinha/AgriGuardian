"""Weather alert schemas and risk evaluation data models."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HazardType(str, Enum):
    """Categorized agricultural weather hazards."""
    HEAT_STRESS = "heat_stress"
    COLD_FROST = "cold_frost"
    HEAVY_RAINFALL = "heavy_rainfall"
    DRY_SPELL = "dry_spell"
    DISEASE_CONDUCIVE = "disease_conducive"
    HIGH_WIND = "high_wind"
    EVAPOTRANSPIRATION = "evapotranspiration"
    GENERAL = "general"


class RiskLevel(str, Enum):
    """Severity classification for agricultural climate risks."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class AlertStatus(str, Enum):
    """Lifecycle status of an in-app agricultural weather alert."""
    NEW = "new"
    ACTIVE = "active"
    UPDATED = "updated"
    EXPIRED = "expired"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class WeatherAlert(BaseModel):
    """Actionable, evidence-grounded agricultural weather risk alert."""
    alert_id: str = Field(..., description="Unique deterministic or generated identifier (e.g. ALT-HEAT-20261109-001)")
    hazard_type: HazardType = Field(..., description="Agricultural weather hazard category")
    risk_level: RiskLevel = Field(..., description="Risk severity level")
    title: str = Field(..., description="Short headline in plain farmer-friendly language")
    summary: str = Field(..., description="Concise synopsis of the expected risk")
    crop: Optional[str] = Field(None, description="Target crop (e.g. 'tomato', 'wheat', 'rice') or None if general")
    crop_stage: Optional[str] = Field(None, description="Susceptible phenological stage or None if all stages")
    
    triggered_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physical metric triggers that caused this alert (e.g. max_temp_c, consecutive_dry_days)"
    )
    forecast_window: Dict[str, str] = Field(
        default_factory=dict,
        description="Time period during which risk is elevated, e.g. {'start': '2026-11-10', 'end': '2026-11-13'}"
    )
    
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Linked Fact IDs from Phase 3 Knowledge Base (e.g. ['FCT-TOM-HEAT-STRESS-01'])"
    )
    source_ids: List[str] = Field(
        default_factory=list,
        description="Authoritative Source IDs backing the rule (e.g. ['SRC-ICAR-TOMATO-2024'])"
    )
    
    explanation: str = Field(..., description="Detailed biological/meteorological rationale explaining why the risk occurs")
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Prioritized, actionable agricultural mitigation steps"
    )
    prohibited_actions: List[str] = Field(
        default_factory=list,
        description="Actions farmers MUST avoid during this event (e.g. spraying chemical pesticides during rain)"
    )
    
    status: AlertStatus = Field(AlertStatus.NEW, description="Lifecycle status")
    is_official_warning: bool = Field(
        False,
        description="True ONLY if corroborated by official IMD / government bulletin; False for automated agro-met models"
    )
    official_bulletin_source: Optional[str] = Field(
        None,
        description="Reference to official IMD bulletin or advisory agency if corroborated"
    )
    
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: Optional[str] = Field(None)
    expires_at: Optional[str] = Field(None)
    acknowledged: bool = Field(False)
    location_district: Optional[str] = Field(None)
    location_state: Optional[str] = Field(None)


class WeatherRiskAssessment(BaseModel):
    """Aggregate risk profile across all evaluated agricultural hazards."""
    location_district: str
    location_state: str
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_crop: Optional[str] = None
    target_stage: Optional[str] = None
    overall_risk_level: RiskLevel = RiskLevel.LOW
    active_alerts: List[WeatherAlert] = Field(default_factory=list)
    risk_summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    advisory_recommendations: List[str] = Field(default_factory=list)

