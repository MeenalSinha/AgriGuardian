"""Farmer analytics service providing personalized farm and crop intelligence."""

from datetime import datetime, date
import logging
from typing import Any, Dict, List, Optional

from backend.app.repositories.farm_db import FarmDatabase
from backend.app.schemas.analytics import (
    ChartPoint,
    FarmerDashboardOverview,
    InsightItem,
    InsightSeverity,
    MetricItem,
    MetricStatus,
)
from backend.app.schemas.farm import CropCycle, Farm, FarmActivity, FarmerProfile
from backend.app.services.analytics.analytics_cache import AnalyticsCache
from backend.app.services.analytics.metric_definitions import build_metric_item
from backend.app.services.farm.farm_context_service import FarmContextService

logger = logging.getLogger(__name__)


class FarmerAnalyticsService:
    """Computes personalized analytics, KPIs, and recommendations for individual farmers."""

    def __init__(
        self,
        farm_db: Optional[FarmDatabase] = None,
        farm_context_service: Optional[FarmContextService] = None,
        cache: Optional[AnalyticsCache] = None,
        weather_service: Optional[Any] = None,
        market_service: Optional[Any] = None,
    ):
        self.db = farm_db or FarmDatabase()
        self.context_service = farm_context_service or FarmContextService(self.db)
        self.cache = cache
        self.weather_service = weather_service
        self.market_service = market_service

    def get_farmer_overview(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> FarmerDashboardOverview:
        """Generate comprehensive, personalized dashboard overview for authenticated farmer."""
        cache_key = f"farmer:{farmer_id}:{farm_id or 'default'}"
        if not force_refresh and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # 1. Fetch farmer profile
        farmer = self.db.get_farmer_profile(farmer_id)
        if not farmer:
            raise ValueError(f"Farmer profile not found for user ID: {farmer_id}")

        # 2. Fetch farms owned by farmer
        farms = self.db.get_farms_by_farmer(farmer_id)
        active_farm: Optional[Farm] = None

        if farm_id:
            # Verify ownership
            for f in farms:
                if f.id == farm_id:
                    active_farm = f
                    break
            if not active_farm:
                raise PermissionError(f"Access denied: Farm {farm_id} is not owned by farmer {farmer_id}")
        elif farms:
            active_farm = farms[0]

        # 3. Retrieve plots and active crop cycles for active farm (or all farms)
        active_plots = []
        active_cycles: List[CropCycle] = []
        if active_farm:
            active_plots = self.db.get_plots_by_farm(active_farm.id, farmer_id)
            active_cycles = self.db.get_active_crop_cycles_by_farm(active_farm.id, farmer_id)
        else:
            # Check all farms owned
            for f in farms:
                active_plots.extend(self.db.get_plots_by_farm(f.id, farmer_id))
                active_cycles.extend(self.db.get_active_crop_cycles_by_farm(f.id, farmer_id))

        # 4. Fetch recent advisory cases recorded for this farmer
        advisory_cases = self.db.list_advisory_cases_for_farmer(farmer_id, limit=5)

        # 5. Fetch due farm activities
        due_activities: List[FarmActivity] = []
        if active_farm:
            due_activities = self.db.get_activities_by_farm(active_farm.id, farmer_id, pending_only=True)
        else:
            for f in farms:
                due_activities.extend(self.db.get_activities_by_farm(f.id, farmer_id, pending_only=True))

        # 6. Compute core KPIs using METRIC_CATALOG
        total_farms_count = len(farms)
        total_acres = round(sum(f.total_acres for f in farms), 2)
        active_crops_count = len(active_cycles)
        due_tasks_count = len(due_activities)
        health_cases_count = len(advisory_cases)

        # Build metrics dictionary
        metrics: Dict[str, MetricItem] = {}
        metrics["farmer_active_farms"] = build_metric_item(
            "farmer_active_farms",
            total_farms_count,
            display_value=f"{total_farms_count} Registered",
            status=MetricStatus.NORMAL if total_farms_count > 0 else MetricStatus.WARNING,
        )
        metrics["farmer_total_acres"] = build_metric_item(
            "farmer_total_acres",
            total_acres,
            display_value=f"{total_acres:.1f} Acres",
            status=MetricStatus.OPTIMAL if total_acres > 0 else MetricStatus.NORMAL,
        )
        metrics["farmer_active_crops"] = build_metric_item(
            "farmer_active_crops",
            active_crops_count,
            display_value=f"{active_crops_count} Active",
            status=MetricStatus.NORMAL if active_crops_count > 0 else MetricStatus.WARNING,
        )
        metrics["farmer_due_tasks"] = build_metric_item(
            "farmer_due_tasks",
            due_tasks_count,
            display_value=f"{due_tasks_count} Pending",
            status=MetricStatus.WARNING if due_tasks_count > 3 else (MetricStatus.NORMAL if due_tasks_count > 0 else MetricStatus.OPTIMAL),
        )
        metrics["farmer_crop_health_cases"] = build_metric_item(
            "farmer_crop_health_cases",
            health_cases_count,
            display_value=f"{health_cases_count} Reported",
            status=MetricStatus.WARNING if health_cases_count > 2 else MetricStatus.NORMAL,
        )

        # Weather risk metric
        weather_summary = None
        weather_risk_score = 0.0
        weather_available = False
        weather_reason = "No active farm coordinates available"

        if active_farm and active_farm.latitude and active_farm.longitude:
            try:
                if self.weather_service:
                    loc = {"latitude": active_farm.latitude, "longitude": active_farm.longitude}
                    w_crop = active_cycles[0].crop_name if active_cycles else "General"
                    w_data = self.weather_service.get_weather_intelligence(location=loc, crop=w_crop)
                    weather_summary = {
                        "temperature_c": w_data["report"].current_temp_c,
                        "humidity_pct": w_data["report"].current_humidity_pct,
                        "condition": w_data["report"].current_condition,
                        "risk_score": w_data["risk_assessment"].risk_score,
                        "risk_level": w_data["risk_assessment"].risk_level.value,
                    }
                    weather_risk_score = w_data["risk_assessment"].risk_score
                    weather_available = True
                else:
                    weather_summary = {
                        "location": f"{active_farm.district}, {active_farm.state}",
                        "status": "Weather service offline",
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch weather for farmer dashboard: {e}")
                weather_reason = f"Weather lookup failed: {str(e)}"

        metrics["farmer_weather_risk_index"] = build_metric_item(
            "farmer_weather_risk_index",
            weather_risk_score if weather_available else 0.0,
            unit="/100",
            display_value=f"{weather_risk_score:.0f}/100" if weather_available else "N/A",
            status=MetricStatus.CRITICAL if weather_risk_score >= 70 else (
                MetricStatus.WARNING if weather_risk_score >= 40 else (
                    MetricStatus.OPTIMAL if weather_available else MetricStatus.UNAVAILABLE
                )
            ),
            is_available=weather_available,
            missing_reason=None if weather_available else weather_reason,
        )

        # Market benchmark metric
        market_rate_summary = None
        primary_crop = active_cycles[0].crop_name if active_cycles else "Wheat"
        primary_loc = f"{active_farm.district}, {active_farm.state}" if active_farm else "Meerut, Uttar Pradesh"
        market_price = 0.0
        market_available = False
        market_reason = "Market data service not connected"

        if self.market_service:
            try:
                m_data = self.market_service.get_market_data(crop=primary_crop, location=primary_loc)
                market_price = getattr(m_data, "current_price", None) or getattr(m_data, "modal_price", None) or 0.0
                market_available = market_price > 0
                market_rate_summary = {
                    "crop": primary_crop,
                    "mandi": getattr(m_data, "mandi", None) or getattr(m_data, "source", "APMC Mandi"),
                    "modal_price": market_price,
                    "price_range": f"{getattr(m_data, 'min_price', 0)} - {getattr(m_data, 'max_price', 0)}",
                    "trend": getattr(m_data, "trend", "stable"),
                    "is_official": getattr(m_data, "is_official_bulletin", True),
                }
            except Exception as e:
                logger.warning(f"Failed to fetch market data for farmer dashboard: {e}")
                market_reason = f"Market lookup failed: {str(e)}"

        metrics["farmer_market_price_bench"] = build_metric_item(
            "farmer_market_price_bench",
            market_price if market_available else 0.0,
            unit="₹/q",
            display_value=f"₹{market_price:.0f}/q" if market_available else "N/A",
            status=MetricStatus.OPTIMAL if market_available else MetricStatus.UNAVAILABLE,
            is_available=market_available,
            missing_reason=None if market_available else market_reason,
        )

        # 7. Crop Distribution Chart
        crop_acres_map: Dict[str, float] = {}
        for c in active_cycles:
            crop_acres_map[c.crop_name] = crop_acres_map.get(c.crop_name, 0.0) + (getattr(c, "target_yield_quintals", None) or 1.0)
        
        crop_distribution = [
            ChartPoint(label=crop_name, value=round(val, 2), category="Crop Area")
            for crop_name, val in crop_acres_map.items()
        ]

        # 8. Active Crop Stages Detail
        today = date.today()
        active_crop_stages = []
        for c in active_cycles:
            s_dt = None
            if c.sowing_date:
                if isinstance(c.sowing_date, date):
                    s_dt = c.sowing_date
                elif isinstance(c.sowing_date, str):
                    try:
                        s_dt = date.fromisoformat(c.sowing_date[:10])
                    except Exception:
                        s_dt = None
            days_since = (today - s_dt).days if s_dt else getattr(c, "days_since_sowing", 0)
            active_crop_stages.append({
                "cycle_id": getattr(c, "id", c.cycle_id),
                "crop_name": c.crop_name,
                "variety": getattr(c, "variety", c.variety_name),
                "sowing_date": s_dt.isoformat() if s_dt else str(c.sowing_date),
                "days_since_sowing": days_since,
                "current_stage": c.current_stage.value,
                "season": c.season.value,
                "target_yield": getattr(c, "target_yield_q_per_acre", c.target_yield_quintals),
            })

        # 9. Explainable Insights Generation
        insights: List[InsightItem] = []

        # Insight for pending activities
        if due_activities:
            overdue_count = 0
            for a in due_activities:
                s_date_str = getattr(a, "scheduled_date", None)
                if s_date_str:
                    try:
                        a_date = date.fromisoformat(s_date_str[:10])
                        if a_date < today:
                            overdue_count += 1
                    except Exception:
                        pass
            if overdue_count > 0:
                insights.append(InsightItem(
                    insight_id=f"ins_act_overdue_{farmer_id}",
                    category="crop_calendar",
                    title=f"{overdue_count} Overdue Farm Tasks",
                    description=f"You have {overdue_count} overdue farm activities requiring immediate field action.",
                    severity=InsightSeverity.WARNING,
                    source="AgriGuardian Agronomic Calendar Engine",
                    suggested_action="Review and complete pending tasks in the My Farms & Crops section.",
                ))
            else:
                insights.append(InsightItem(
                    insight_id=f"ins_act_due_{farmer_id}",
                    category="crop_calendar",
                    title="Upcoming Farm Activities Scheduled",
                    description=f"{len(due_activities)} agronomic activities scheduled for the current crop stage.",
                    severity=InsightSeverity.INFO,
                    source="AgriGuardian Agronomic Calendar Engine",
                    suggested_action="Ensure inputs (fertilizers/irrigation) are prepared.",
                ))

        # Insight for active crop stages
        for c in active_cycles:
            s_dt = None
            if c.sowing_date:
                if isinstance(c.sowing_date, date):
                    s_dt = c.sowing_date
                elif isinstance(c.sowing_date, str):
                    try:
                        s_dt = date.fromisoformat(c.sowing_date[:10])
                    except Exception:
                        s_dt = None
            days = (today - s_dt).days if s_dt else getattr(c, "days_since_sowing", 0)
            if c.crop_name.lower() == "wheat" and 18 <= days <= 25:
                insights.append(InsightItem(
                    insight_id=f"ins_pheno_{getattr(c, 'cycle_id', 'w')}",
                    category="phenology",
                    title="Wheat: Crown Root Initiation (CRI) Stage",
                    description="Wheat is at 21±3 days after sowing. This is the single most critical irrigation window for root development.",
                    severity=InsightSeverity.RECOMMENDATION,
                    source="ICAR Wheat Package of Practices",
                    suggested_action="Apply the first irrigation immediately to prevent up to 25% yield loss.",
                ))
            elif c.crop_name.lower() == "rice" and 30 <= days <= 45:
                insights.append(InsightItem(
                    insight_id=f"ins_pheno_{c.id}",
                    category="phenology",
                    title="Rice: Active Tillering Phase",
                    description="Rice crop is in active tillering stage. Monitor for stem borer and maintain 2-3 cm shallow water level.",
                    severity=InsightSeverity.INFO,
                    source="ICAR Rice Crop Management Protocols",
                    suggested_action="Apply top-dressing nitrogen if recommended by soil test.",
                ))

        # Insight for recent high risk health cases
        high_risk_cases = [c for c in advisory_cases if c.risk == "High"]
        if high_risk_cases:
            insights.append(InsightItem(
                insight_id=f"ins_health_{high_risk_cases[0].case_id}",
                category="crop_health",
                title=f"High Risk Flag: {high_risk_cases[0].crop}",
                description=f"Advisory case {high_risk_cases[0].case_id} identified high-risk symptoms ({high_risk_cases[0].symptoms[:60]}...).",
                severity=InsightSeverity.CRITICAL,
                source="AgriGuardian Computer Vision & Foliar Diagnosis",
                suggested_action="Schedule an agronomist field inspection or follow recommended 3-step advisory.",
            ))

        overview = FarmerDashboardOverview(
            farmer=farmer,
            active_farm=active_farm,
            available_farms=farms,
            metrics=metrics,
            crop_distribution=crop_distribution,
            active_crop_stages=active_crop_stages,
            recent_health_cases=advisory_cases,
            weather_summary=weather_summary,
            market_rate_summary=market_rate_summary,
            due_activities=due_activities[:8],
            insights=insights,
            data_freshness="Live Relational Core",
        )

        if self.cache:
            self.cache.set(cache_key, overview)

        return overview
