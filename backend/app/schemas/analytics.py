"""Pydantic schemas for Farmer, Field Officer, and Administrator Dashboards and Analytics."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.farm import (
    CropCycle,
    CropGrowthStage,
    Farm,
    FarmActivity,
    FarmerProfile,
    Plot,
    UserRole,
)


class PredefinedDateRange(str, Enum):
    TODAY = "today"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    LAST_90D = "last_90d"
    ALL_TIME = "all_time"


class MetricStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OPTIMAL = "optimal"
    UNAVAILABLE = "unavailable"


class MetricItem(BaseModel):
    """Structured dashboard metric with formatting, status, and data provenance."""
    metric_id: str
    name: str
    value: Any
    unit: str = ""
    display_value: str
    change_pct: Optional[float] = None
    period: str = "current"
    status: MetricStatus = MetricStatus.NORMAL
    data_source: str = "AgriGuardian Relational Core"
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_available: bool = True
    missing_reason: Optional[str] = None


class ChartPoint(BaseModel):
    """Individual data point for time-series, categorical distribution, or geospatial display."""
    label: str
    value: float
    category: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class InsightSeverity(str, Enum):
    INFO = "info"
    RECOMMENDATION = "recommendation"
    WARNING = "warning"
    CRITICAL = "critical"


class InsightItem(BaseModel):
    """Explainable agricultural, agronomic, or platform health insight."""
    insight_id: str
    category: str  # e.g., 'crop_stage', 'weather_risk', 'market_trend', 'integration_health'
    title: str
    description: str
    severity: InsightSeverity = InsightSeverity.INFO
    source: str
    suggested_action: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    valid_until: Optional[str] = None


# --- Advisory Case & Follow-Up Schemas ---

class AdvisoryCase(BaseModel):
    """Durable relational advisory and crop-health record."""
    case_id: str
    farmer_id: Optional[str] = None
    farmer_name: str
    farm_id: Optional[str] = None
    crop: str
    location: str
    state: Optional[str] = None
    district: Optional[str] = None
    symptoms: str = ""
    risk: str = "Low"
    field_visit_required: bool = False
    status: str = "pending"  # pending, in_progress, reviewed, resolved, closed
    advisory_hi: str = ""
    advisory_en: str = ""
    steps: List[str] = Field(default_factory=list)
    weather_summary: str = ""
    market_summary: str = ""
    image_analysis: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class CaseFollowUpRecord(BaseModel):
    """Audit log of officer interaction or field visit outcome."""
    follow_up_id: str
    case_id: str
    officer_id: str
    officer_name: str
    notes: str
    action_taken: Optional[str] = None
    status_transition: Optional[str] = None
    created_at: str


class CaseFollowUpCreate(BaseModel):
    notes: str = Field(..., min_length=2, description="Agronomist field notes or interaction details")
    action_taken: Optional[str] = Field(default=None, description="Actions executed (e.g. spray schedule, soil sample)")
    new_status: Optional[str] = Field(default=None, description="New case status: pending, in_progress, reviewed, resolved")


# --- Officer Territory Schemas ---

class OfficerTerritoryScope(BaseModel):
    assignment_id: str
    officer_id: str
    state: str
    district: str
    block: Optional[str] = None
    assigned_at: str


class OfficerTerritoryAssignRequest(BaseModel):
    officer_id: str
    state: str
    district: str
    block: Optional[str] = None


# --- Farmer Dashboard Overview Schema ---

class FarmerDashboardOverview(BaseModel):
    """Personalized agricultural dashboard for authenticated farmer."""
    farmer: FarmerProfile
    active_farm: Optional[Farm] = None
    available_farms: List[Farm] = Field(default_factory=list)
    metrics: Dict[str, MetricItem]
    crop_distribution: List[ChartPoint] = Field(default_factory=list)
    active_crop_stages: List[Dict[str, Any]] = Field(default_factory=list)
    recent_health_cases: List[AdvisoryCase] = Field(default_factory=list)
    weather_summary: Optional[Dict[str, Any]] = None
    market_rate_summary: Optional[Dict[str, Any]] = None
    due_activities: List[FarmActivity] = Field(default_factory=list)
    insights: List[InsightItem] = Field(default_factory=list)
    data_freshness: str = "Live Relational Cache"


# --- Field Officer Dashboard Overview Schema ---

class FieldOfficerDashboardOverview(BaseModel):
    """Regional agricultural intelligence dashboard for authorized field officers."""
    officer_id: str
    officer_name: str
    assigned_territory: List[OfficerTerritoryScope] = Field(default_factory=list)
    metrics: Dict[str, MetricItem]
    crop_distribution: List[ChartPoint] = Field(default_factory=list)
    risk_distribution: Dict[str, int] = Field(default_factory=dict)
    case_register: List[AdvisoryCase] = Field(default_factory=list)
    follow_up_queue: List[AdvisoryCase] = Field(default_factory=list)
    village_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[InsightItem] = Field(default_factory=list)
    data_freshness: str = "Live Relational Cache"


# --- Administrator Dashboard Overview Schema ---

class IntegrationHealthRecord(BaseModel):
    service_name: str
    status: str  # 'active', 'degraded', 'offline', 'mock_mode'
    latency_ms: float
    error_rate_pct: float = 0.0
    last_checked: str
    details: str


class ModelPerformanceMetrics(BaseModel):
    model_name: str
    version: str
    device: str
    avg_latency_ms: float
    total_inferences: int
    low_confidence_pct: float


class DataQualitySummary(BaseModel):
    missing_market_records: int
    stale_weather_records: int
    unverified_sources: int
    incomplete_profiles: int
    overall_quality_pct: float


class AdminDashboardOverview(BaseModel):
    """System-level operational and product analytics for platform administrators."""
    admin_id: str
    admin_name: str
    metrics: Dict[str, MetricItem]
    user_growth_trend: List[ChartPoint] = Field(default_factory=list)
    usage_by_feature: Dict[str, int] = Field(default_factory=dict)
    integration_health: List[IntegrationHealthRecord] = Field(default_factory=list)
    model_performance: ModelPerformanceMetrics
    data_quality: DataQualitySummary
    security_events_count: int = 0
    system_alerts: List[InsightItem] = Field(default_factory=list)
    data_freshness: str = "Live Operational Telemetry"

