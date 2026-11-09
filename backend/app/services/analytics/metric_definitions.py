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

    @property
    def allowed_roles(self) -> List[UserRole]:
        if self.required_role == UserRole.ADMINISTRATOR:
            return [UserRole.ADMINISTRATOR]
        elif self.required_role == UserRole.FIELD_OFFICER:
            return [UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR]
        elif self.required_role == UserRole.AGRONOMIST:
            return [UserRole.AGRONOMIST, UserRole.ADMINISTRATOR]
        return [UserRole.FARMER, UserRole.FIELD_OFFICER, UserRole.AGRONOMIST, UserRole.ADMINISTRATOR]


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
    "farmer_total_acres": MetricDefinition(
        metric_id="farmer_total_acres",
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
    "farmer_due_tasks": MetricDefinition(
        metric_id="farmer_due_tasks",
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
    "farmer_weather_risk_index": MetricDefinition(
        metric_id="farmer_weather_risk_index",
        display_name="Farm Microclimate Risk",
        description="Composite agricultural weather risk index on a 0-100 scale evaluating active hazards.",
        formula="Weighted sum of active weather hazard alerts (Heat, Frost, Excess Rain, Dry-spell)",
        unit="/100",
        source="WeatherIntelligenceService / Open-Meteo",
        required_role=UserRole.FARMER,
        date_window_behavior="Real-time 24h hazard window",
        aggregation_method="MAX",
        missing_data_behavior="Sets is_available=False if GPS coordinates are missing or API unreachable.",
        known_limitations="Spatial resolution is 5.5 km grid cell.",
    ),
    "farmer_market_price_bench": MetricDefinition(
        metric_id="farmer_market_price_bench",
        display_name="Local Mandi Rate Benchmark",
        description="Official modal price (Rs/qtl) for the primary crop at the closest APMC market yard.",
        formula="AGMARKNET verified bulletin modal_price for (crop, location)",
        unit="₹/q",
        source="AGMARKNET / e-NAM Daily Market Bulletins",
        required_role=UserRole.FARMER,
        date_window_behavior="Latest official bulletin date",
        aggregation_method="LATEST",
        missing_data_behavior="Sets is_available=False with missing_reason if crop has no reported trade today.",
        known_limitations="Reflects wholesale market auctions, excluding private local farmgate deals.",
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
    "officer_assigned_districts": MetricDefinition(
        metric_id="officer_assigned_districts",
        display_name="Assigned Districts",
        description="Count of distinct administrative districts allocated to the field officer.",
        formula="COUNT(DISTINCT district FROM field_officer_assignments WHERE officer_id = :id)",
        unit="Districts",
        source="field_officer_assignments table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Current administrative allocation",
        aggregation_method="COUNT DISTINCT",
        missing_data_behavior="Sets is_available=False with explanation if no territory assigned.",
        known_limitations="Does not subdivide below sub-district/block boundaries unless specified.",
    ),
    "officer_registered_cases": MetricDefinition(
        metric_id="officer_registered_cases",
        display_name="Reported Cases in Jurisdiction",
        description="Total crop health incidents reported within the officer's assigned districts.",
        formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts))",
        unit="Cases",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Selected date window or all time",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 if no incidents reported in territory.",
        known_limitations="Depends on farmer proactive app usage and diagnostic submissions.",
    ),
    "officer_high_risk_cases": MetricDefinition(
        metric_id="officer_high_risk_cases",
        display_name="High Priority / Outbreak Cases",
        description="Cases with high foliar damage or severe weather risk requiring urgent agronomist triage.",
        formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts) AND risk = 'High')",
        unit="Alerts",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Selected date window",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 if no high risk cases found.",
        known_limitations="AI foliar prediction severity, requiring ground-truth verification.",
    ),
    "officer_field_visits_due": MetricDefinition(
        metric_id="officer_field_visits_due",
        display_name="Pending Field Visits",
        description="Cases explicitly flagged as requiring in-person field visits that remain pending.",
        formula="COUNT(case_id FROM advisory_cases WHERE district IN (:officer_districts) AND field_visit_required = 1 AND status = 'pending')",
        unit="Visits",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Current queue snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 with status OPTIMAL if all visits completed.",
        known_limitations="Field visit flags set by AI severity thresholds or farmer manual request.",
    ),
    "officer_resolution_rate": MetricDefinition(
        metric_id="officer_resolution_rate",
        display_name="Case Resolution Rate",
        description="Percentage of assigned district advisory cases reviewed or resolved.",
        formula="(COUNT(status IN ('reviewed', 'resolved')) / COUNT(all territory cases)) * 100.0",
        unit="%",
        source="advisory_cases and case_follow_ups (SQLite)",
        required_role=UserRole.FIELD_OFFICER,
        date_window_behavior="Selected date window",
        aggregation_method="PERCENTAGE",
        missing_data_behavior="Returns 100.0% if total cases is 0.",
        known_limitations="Relies on field officers marking case status transitions in app.",
    ),
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

    # --- Administrator Domain Metrics ---
    "admin_total_users": MetricDefinition(
        metric_id="admin_total_users",
        display_name="Total User Accounts",
        description="Total registered accounts across all roles (Farmer, Field Officer, Agronomist, Administrator).",
        formula="COUNT(user_id FROM users)",
        unit="Accounts",
        source="users table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Cumulative snapshot",
        aggregation_method="COUNT",
        missing_data_behavior="Returns count of existing records.",
        known_limitations="Includes all registered accounts.",
    ),
    "admin_active_users_30d": MetricDefinition(
        metric_id="admin_active_users_30d",
        display_name="Active Users (30 Days)",
        description="Unique user accounts that have authenticated or logged in within the past 30 days.",
        formula="COUNT(user_id FROM users WHERE last_login >= NOW() - INTERVAL 30 DAYS)",
        unit="Users",
        source="users table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Rolling 30-day window",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0 if no logins recorded.",
        known_limitations="Considers token authentication as active login.",
    ),
    "admin_platform_farms": MetricDefinition(
        metric_id="admin_platform_farms",
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
    "admin_platform_crops": MetricDefinition(
        metric_id="admin_platform_crops",
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
    "admin_total_advisories": MetricDefinition(
        metric_id="admin_total_advisories",
        display_name="Advisories Generated",
        description="Total agricultural advisory cases generated across voice, text, and foliar diagnosis.",
        formula="COUNT(case_id FROM advisory_cases)",
        unit="Cases",
        source="advisory_cases table (SQLite)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Cumulative count",
        aggregation_method="COUNT",
        missing_data_behavior="Returns 0.",
        known_limitations="None.",
    ),
    "admin_system_uptime": MetricDefinition(
        metric_id="admin_system_uptime",
        display_name="System Core Health",
        description="Real-time availability percentage across core service dependency health checks.",
        formula="(active_services / total_services) * 100.0",
        unit="%",
        source="Live Service Probes (5 Subsystems)",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Current probe execution",
        aggregation_method="PERCENTAGE",
        missing_data_behavior="Evaluated live on every overview request.",
        known_limitations="Evaluates live responsiveness of Gnani, PyTorch, Open-Meteo, AGMARKNET, and ICAR KB.",
    ),
    "admin_api_error_rate": MetricDefinition(
        metric_id="admin_api_error_rate",
        display_name="API Error Rate",
        description="Percentage of HTTP API requests failing with 5xx status codes over a rolling window.",
        formula="(COUNT(5xx errors) / COUNT(all requests)) * 100.0",
        unit="%",
        source="OpenTelemetry / API Gateway logs",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Rolling 24-hour window",
        aggregation_method="RATIO",
        missing_data_behavior="Marked as is_available=False when continuous telemetry exporter is unconfigured.",
        known_limitations="Requires access log parser or tracing exporter in production.",
    ),
    "admin_data_quality_score": MetricDefinition(
        metric_id="admin_data_quality_score",
        display_name="Platform Data Quality Index",
        description="Composite percentage scoring source authenticity, profile completeness, and market freshness.",
        formula="100.0 * (1.0 - incomplete_profiles / total_profiles)",
        unit="%",
        source="Profile & Source Integrity Audit",
        required_role=UserRole.ADMINISTRATOR,
        date_window_behavior="Current audit snapshot",
        aggregation_method="COMPOSITE",
        missing_data_behavior="Defaults to 100% if no incomplete records are found.",
        known_limitations="Evaluates completeness of mandatory phone and profile fields.",
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
}

# Aliases for backwards compatibility
METRIC_CATALOG["farmer_managed_area_acres"] = METRIC_CATALOG["farmer_total_acres"]
METRIC_CATALOG["farmer_pending_activities"] = METRIC_CATALOG["farmer_due_tasks"]
METRIC_CATALOG["officer_reported_cases"] = METRIC_CATALOG["officer_registered_cases"]
METRIC_CATALOG["admin_total_accounts"] = METRIC_CATALOG["admin_total_users"]
METRIC_CATALOG["admin_active_accounts_30d"] = METRIC_CATALOG["admin_active_users_30d"]
METRIC_CATALOG["admin_platform_total_farms"] = METRIC_CATALOG["admin_platform_farms"]
METRIC_CATALOG["admin_platform_total_crops"] = METRIC_CATALOG["admin_platform_crops"]


def get_metric_catalog_entry(metric_id: str) -> Optional[MetricDefinition]:
    """Retrieve metric specification from catalog."""
    return METRIC_CATALOG.get(metric_id)


def get_metric_definition(metric_id: str) -> Optional[MetricDefinition]:
    """Alias for get_metric_catalog_entry."""
    return METRIC_CATALOG.get(metric_id)


def get_metrics_for_role(role: Any) -> List[str]:
    """Retrieve all metric identifiers applicable to a specific user role."""
    target_role = role if isinstance(role, UserRole) else UserRole(str(role).lower().replace("userrole.", ""))
    result = []
    for m_id, m in METRIC_CATALOG.items():
        if target_role in m.allowed_roles:
            result.append(m_id)
    return result


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
    data_source: Optional[str] = None,
) -> MetricItem:
    """Factory helper to construct a validated MetricItem bound to the central metric catalog."""
    defn = get_metric_catalog_entry(metric_id)
    name = defn.name if defn else metric_id.replace("_", " ").title()
    eff_unit = unit if unit is not None else (defn.unit if defn else "")
    eff_source = data_source or (defn.data_source if defn else "AgriGuardian Core")

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
