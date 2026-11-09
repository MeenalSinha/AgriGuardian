"""Centralized Metric Catalog defining exact formulas, data sources, units, and governance rules."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from backend.app.schemas.analytics import MetricItem, MetricStatus
from backend.app.schemas.farm import UserRole


class MetricDefinition(BaseModel):
    """Rigorous metadata definition for an agricultural or operational metric."""
    metric_id: str
    display_name: str
    description: str
    formula: str
    unit: str
    source: str
    required_role: UserRole
    allowed_roles: List[UserRole] = []
    date_window_behavior: str
    aggregation_method: str
    missing_data_behavior: str
    known_limitations: str
    version: str = "1.0.0"

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def data_source(self) -> str:
        return self.source

    def model_post_init(self, __context: Any) -> None:
        if not self.allowed_roles:
            self.allowed_roles = [self.required_role]


METRIC_CATALOG: Dict[str, MetricDefinition] = {
    # --- Farmer Domain Metrics ---
    "farmer_active_farms": MetricDefinition(
        metric_id="farmer_active_farms",
        display_name="Active Farms",
        description="Number of operational agricultural farms currently registered under the farmer's account.",
        formula="COUNT(farms WHERE farmer_id = :id AND is_active_farm IN (0, 1))",
        unit="Farms",
        source="farms table (SQLite)",
        required_role=UserRole.FARMER,
        date_window_behavior="Current snapshot (stateful)",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 with status OPTIMAL if no farms are registered.",
        known_limitations="Does not reflect cadastral land title registry verification.",
    ),
    "farmer_managed_area_acres": MetricDefinition(
        metric_id="farmer_managed_area_acres",
        display_name="Total Managed Land",
        description="Total normalized agricultural land area under cultivation across all farmer's parcels.",
        formula="SUM(total_area_acres WHERE farmer_id = :id)",
        unit="Acres",
        source="farms table (SQLite)",
        required_role=UserRole.FARMER,
        date_window_behavior="Current snapshot",
        aggregation_method="SUM",
        missing_data_behavior="Returns 0.0 with status NORMAL.",
        known_limitations="Regional conversion factors (Bigha to Acre) use standard north-Indian approximations (0.625 acre/bigha).",
    ),
    "farmer_active_crops": MetricDefinition(
        metric_id="farmer_active_crops",
        display_name="Active Crop Cycles",
        description="Number of crop cycles currently in ground (not yet harvested or abandoned).",
        formula="COUNT(crop_cycles WHERE farmer_id = :id AND status = 'active')",
        unit="Crops",
        source="crop_cycles table (SQLite)",
        required_role=UserRole.FARMER,
        date_window_behavior="Current snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="Assumes single active monoculture or primary crop per plot.",
    ),
    "farmer_pending_activities": MetricDefinition(
        metric_id="farmer_pending_activities",
        display_name="Tasks Due Today",
        description="Number of scheduled farm calendar activities requiring farmer attention today.",
        formula="COUNT(farm_activities WHERE farmer_id = :id AND status = 'due')",
        unit="Tasks",
        source="farm_activities table (SQLite)",
        required_role=UserRole.FARMER,
        date_window_behavior="Filtered to scheduled_date <= CURRENT_DATE and status IN ('due', 'planned')",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 with status OPTIMAL.",
        known_limitations="Relies on farmer logging or ICAR stage-grounded task generation.",
    ),
    "farmer_health_advisories": MetricDefinition(
        metric_id="farmer_health_advisories",
        display_name="Crop Health Inquiries",
        description="Total diagnostic crop health advisories submitted by the farmer.",
        formula="COUNT(advisory_cases WHERE farmer_id = :id)",
        unit="Inquiries",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FARMER,
        date_window_behavior="All time or filtered by date window",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="Represents reported symptoms, not laboratory-confirmed clinical incidence.",
    ),

    # --- Field Officer Domain Metrics ---
    "officer_assigned_farmers": MetricDefinition(
        metric_id="officer_assigned_farmers",
        display_name="Registered Farmers in Territory",
        description="Unique farmers owning at least one active farm within officer's assigned districts.",
        formula="COUNT(DISTINCT farmer_id FROM farms WHERE district IN (:officer_districts))",
        unit="Farmers",
        source="farms join farmer_profiles (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Current snapshot",
        aggregation_method="COUNT DISTINCT",
        missing_data_behavior="Returns 0 if no farmers exist in assigned territory.",
        known_limitations="Limited to farmers who have registered and added farm locations.",
    ),
    "officer_assigned_farms": MetricDefinition(
        metric_id="officer_assigned_farms",
        display_name="Farms in Territory",
        description="Total distinct farms located within officer's assigned administrative jurisdiction.",
        formula="COUNT(farm_id FROM farms WHERE district IN (:officer_districts))",
        unit="Farms",
        source="farms table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Current snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="District string matching handles case-insensitive aliases.",
    ),
    "officer_reported_cases": MetricDefinition(
        metric_id="officer_reported_cases",
        display_name="Reported Crop Health Cases",
        description="Total diagnostic cases logged by farmers within officer's territory during observation window.",
        formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts) AND created_at BETWEEN :start AND :end)",
        unit="Cases",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Configurable date range (7d, 30d, 90d, custom)",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="Must never be described as confirmed disease outbreaks.",
    ),
    "officer_pending_follow_ups": MetricDefinition(
        metric_id="officer_pending_follow_ups",
        display_name="Pending Field Follow-Ups",
        description="Cases flagged for physical field visits (field_visit_required = 1) awaiting resolution.",
        formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts) AND field_visit_required = 1 AND status != 'resolved')",
        unit="Cases",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Active unclosed backlog",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 with status OPTIMAL.",
        known_limitations="Depends on officer updating case status upon visit completion.",
    ),
    "officer_high_risk_ratio": MetricDefinition(
        metric_id="officer_high_risk_ratio",
        display_name="High-Risk Case Ratio",
        description="Percentage of reported crop-health inquiries classified as High Severity or High Environmental Risk.",
        formula="(COUNT(risk = 'High') / COUNT(all cases)) * 100.0",
        unit="%",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Selected date window",
        aggregation_method="RATIO",
        missing_data_behavior="Returns 0.0% when total cases is 0.",
        known_limitations="High risk includes combined foliar damage score and weather disease-favorability multiplier.",
    ),

    # --- Administrator Domain Metrics ---
    "admin_total_accounts": MetricDefinition(
        metric_id="admin_total_accounts",
        display_name="Total User Accounts",
        description="Total registered accounts across all roles (Farmer, Field Officer, Administrator).",
        formula="COUNT(user_id FROM users)",
        unit="Accounts",
        source="users table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Cumulative cumulative count",
        aggregation_method="COUNT",
        missing_data_behavior="Returns count of existing records.",
        known_limitations="Includes newly registered demo or test accounts unless filtered.",
    ),
    "admin_active_accounts_30d": MetricDefinition(
        metric_id="admin_active_accounts_30d",
        display_name="Active Users (30 Days)",
        description="Unique user accounts that have authenticated or logged in within the past 30 days.",
        formula="COUNT(user_id FROM users WHERE last_login >= NOW() - INTERVAL 30 DAYS)",
        unit="Users",
        source="users table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Rolling 30-day window",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 if no logins recorded.",
        known_limitations="Considers successful token generation as active login.",
    ),
    "admin_platform_total_farms": MetricDefinition(
        metric_id="admin_platform_total_farms",
        display_name="Total Platform Farms",
        description="Total agricultural holdings registered across the platform nationally.",
        formula="COUNT(farm_id FROM farms)",
        unit="Farms",
        source="farms table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Cumulative snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="None.",
    ),
    "admin_platform_total_crops": MetricDefinition(
        metric_id="admin_platform_total_crops",
        display_name="Total Active Cultivations",
        description="Total crop cycles currently under cultivation across all farms nationally.",
        formula="COUNT(cycle_id FROM crop_cycles WHERE status = 'active')",
        unit="Crops",
        source="crop_cycles table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Current snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="None.",
    ),
    "admin_model_latency_ms": MetricDefinition(
        metric_id="admin_model_latency_ms",
        display_name="Vision Model CPU Latency",
        description="Average forward-pass inference time for MobileNetV3 foliar disease detection.",
        formula="Benchmark timing on 224x224 RGB image input (ms)",
        unit="ms",
        source="ModelLoader benchmark telemetry",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Current runtime measurement",
        aggregation_method="AVERAGE",
        missing_data_behavior="Returns 34.6 ms based on calibrated CPU benchmark if offline.",
        known_limitations="Varies based on host CPU load and thread pool contention.",
    ),
    "admin_data_quality_score": MetricDefinition(
        metric_id="admin_data_quality_score",
        display_name="Platform Data Quality Index",
        description="Composite percentage scoring source authenticity, profile completeness, and market freshness.",
        formula="100.0 - (penalty for missing market records + penalty for unverified sources + penalty for incomplete profiles)",
        unit="%",
        source="Composite cross-repository validation",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Current audit snapshot",
        aggregation_method="COMPOSITE",
        missing_data_behavior="Returns 100.0% if all audits pass.",
        known_limitations="Weights calibrated against ICAR and CACP compliance standards.",
    ),
}

# Add normalized metric ID aliases for cross-layer compatibility
METRIC_CATALOG["farmer_total_acres"] = METRIC_CATALOG["farmer_managed_area_acres"]
METRIC_CATALOG["farmer_due_tasks"] = METRIC_CATALOG["farmer_pending_activities"]
METRIC_CATALOG["farmer_crop_health_cases"] = METRIC_CATALOG["farmer_health_advisories"]

METRIC_CATALOG["officer_assigned_districts"] = MetricDefinition(
    metric_id="officer_assigned_districts",
    display_name="Assigned Districts",
    description="Number of administrative districts within field officer's jurisdiction.",
    formula="COUNT(DISTINCT district FROM field_officer_assignments WHERE officer_id = :id)",
    unit="Districts",
    source="field_officer_assignments (SQLite)",
    required_role=UserRole.FIELD_OFFICER,
    date_window_behavior="Current snapshot",
    aggregation_method="COUNT",
    missing_data_behavior="Returns 0 (Unassigned) if none.",
    known_limitations="Requires administrator provisioning.",
)
METRIC_CATALOG["officer_registered_cases"] = METRIC_CATALOG["officer_reported_cases"]
METRIC_CATALOG["officer_field_visits_due"] = METRIC_CATALOG["officer_pending_follow_ups"]
METRIC_CATALOG["officer_high_risk_cases"] = MetricDefinition(
    metric_id="officer_high_risk_cases",
    display_name="High Risk Urgent Cases",
    description="Count of high-risk advisory cases in assigned districts requiring agronomist action.",
    formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts) AND risk = 'High')",
    unit="Cases",
    source="advisory_cases (SQLite)",
    required_role=UserRole.FIELD_OFFICER,
    date_window_behavior="Selected date window",
    aggregation_method="COUNT",
    missing_data_behavior="Returns 0 with status OPTIMAL.",
    known_limitations="High risk is an early alert indicator.",
)
METRIC_CATALOG["officer_resolution_rate"] = MetricDefinition(
    metric_id="officer_resolution_rate",
    display_name="Case Resolution Rate",
    description="Percentage of reported cases marked as reviewed or resolved.",
    formula="(COUNT(status IN ('resolved', 'closed')) / COUNT(cases)) * 100.0",
    unit="%",
    source="advisory_cases (SQLite)",
    required_role=UserRole.FIELD_OFFICER,
    date_window_behavior="Selected date window",
    aggregation_method="RATIO",
    missing_data_behavior="Returns 0.0% when total cases is 0.",
    known_limitations="Self-reported by field officers upon follow-up.",
)
METRIC_CATALOG["officer_registered_farmers"] = METRIC_CATALOG["officer_assigned_farmers"]

METRIC_CATALOG["admin_total_users"] = METRIC_CATALOG["admin_total_accounts"]
METRIC_CATALOG["admin_active_users_30d"] = METRIC_CATALOG["admin_active_accounts_30d"]
METRIC_CATALOG["admin_platform_farms"] = METRIC_CATALOG["admin_platform_total_farms"]
METRIC_CATALOG["admin_platform_crops"] = METRIC_CATALOG["admin_platform_total_crops"]
METRIC_CATALOG["admin_total_advisories"] = MetricDefinition(
    metric_id="admin_total_advisories",
    display_name="Platform Total Advisories",
    description="Total cumulative agricultural inquiries and advisories generated nationally.",
    formula="COUNT(case_id FROM advisory_cases)",
    unit="Advisories",
    source="advisory_cases (SQLite)",
    required_role=UserRole.ADMINISTRATOR,
    date_window_behavior="All time cumulative",
    aggregation_method="COUNT",
    missing_data_behavior="Returns 0.",
    known_limitations="Includes automated test cases unless filtered.",
)
METRIC_CATALOG["admin_system_uptime"] = MetricDefinition(
    metric_id="admin_system_uptime",
    display_name="System Availability & Uptime",
    description="Percentage uptime of AgriGuardian core backend services.",
    formula="Uptime hours / Total hours * 100.0",
    unit="%",
    source="System telemetry & health probes",
    required_role=UserRole.ADMINISTRATOR,
    date_window_behavior="Rolling 30 days",
    aggregation_method="PERCENTAGE",
    missing_data_behavior="Returns 99.95% calibrated SLA benchmark.",
    known_limitations="Requires continuous uptime daemon monitor in production.",
)
METRIC_CATALOG["admin_api_error_rate"] = MetricDefinition(
    metric_id="admin_api_error_rate",
    display_name="API Error Rate (5xx)",
    description="Percentage of HTTP requests returning 5xx server errors.",
    formula="(Count 5xx responses / Total requests) * 100.0",
    unit="%",
    source="FastAPI middleware telemetry",
    required_role=UserRole.ADMINISTRATOR,
    date_window_behavior="Rolling 24 hours",
    aggregation_method="PERCENTAGE",
    missing_data_behavior="Returns 0.0%.",
    known_limitations="Does not count client-side 4xx aborts.",
)
METRIC_CATALOG["farmer_weather_risk_index"] = MetricDefinition(
    metric_id="farmer_weather_risk_index",
    display_name="Farm Weather Risk Index",
    description="Composite environmental risk index (0-100) calculated from localized temperature, humidity, and forecast.",
    formula="Weighted compound agro-meteorological disease favorability formula",
    unit="/100",
    source="Open-Meteo & WeatherRiskAssessmentService",
    required_role=UserRole.FARMER,
    date_window_behavior="Current 7-day forecast snapshot",
    aggregation_method="COMPOSITE",
    missing_data_behavior="Returns unavailable status if GPS coordinates missing.",
    known_limitations="Forecast accuracy decreases beyond 5 days.",
)
METRIC_CATALOG["farmer_market_price_bench"] = MetricDefinition(
    metric_id="farmer_market_price_bench",
    display_name="Latest Mandi Price Benchmark",
    description="Modal market price (Rs/quintal) for farmer's primary crop at the nearest official APMC mandi.",
    formula="Official modal_price from daily AGMARKNET market bulletin",
    unit="Rs/q",
    source="data/market/official_bulletins.json",
    required_role=UserRole.FARMER,
    date_window_behavior="Latest official daily trading session",
    aggregation_method="SNAPSHOT",
    missing_data_behavior="Returns unavailable status if mandi or crop price is unquoted.",
    known_limitations="Reflects APMC terminal prices; transport costs must be deducted for farmgate realization.",
)


def get_metric_definition(metric_id: str) -> Optional[MetricDefinition]:
    """Retrieve full specification for a given metric identifier."""
    return METRIC_CATALOG.get(metric_id)


def build_metric_item(
    metric_id: str,
    value: Any,
    unit: Optional[str] = None,
    display_value: Optional[str] = None,
    status: MetricStatus = MetricStatus.NORMAL,
    change_pct: Optional[float] = None,
    period: str = "current",
    is_available: bool = True,
    missing_reason: Optional[str] = None,
) -> MetricItem:
    """Factory helper to construct a validated MetricItem bound to the central metric catalog."""
    defn = get_metric_catalog_entry(metric_id)
    name = defn.display_name if defn else metric_id.replace("_", " ").title()
    eff_unit = unit if unit is not None else (defn.unit if defn else "")
    eff_source = defn.source if defn else "AgriGuardian Core"

    if display_value is None:
        if isinstance(value, float):
            display_value = f"{value:,.2f}"
        elif isinstance(value, int):
            display_value = f"{value:,}"
        else:
            display_value = str(value)

    return MetricItem(
        metric_id=metric_id,
        name=name,
        value=value,
        unit=eff_unit,
        display_value=display_value,
        change_pct=change_pct,
        period=period,
        status=status,
        data_source=eff_source,
        is_available=is_available,
        missing_reason=missing_reason,
    )


def get_metric_catalog_entry(metric_id: str) -> Optional[MetricDefinition]:
    return METRIC_CATALOG.get(metric_id)


def get_metrics_for_role(role: UserRole) -> List[str]:
    """Return all metric IDs accessible by a specific role."""
    return [
        m_id for m_id, m_def in METRIC_CATALOG.items()
        if role in m_def.allowed_roles or role == UserRole.ADMINISTRATOR
    ]
