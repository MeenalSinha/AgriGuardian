"""Field Officer analytics service providing regional agricultural intelligence and case management."""

from collections import Counter
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional

from backend.app.repositories.farm_db import FarmDatabase
from backend.app.schemas.analytics import (
    AdvisoryCase,
    CaseFollowUpRecord,
    ChartPoint,
    FieldOfficerDashboardOverview,
    InsightItem,
    InsightSeverity,
    MetricItem,
    MetricStatus,
    OfficerTerritoryScope,
)
from backend.app.services.analytics.analytics_cache import AnalyticsCache
from backend.app.services.analytics.metric_definitions import build_metric_item

logger = logging.getLogger(__name__)


class FieldOfficerAnalyticsService:
    """Manages regional territory intelligence, case tracking, and agronomist follow-ups."""

    def __init__(
        self,
        farm_db: Optional[FarmDatabase] = None,
        cache: Optional[AnalyticsCache] = None,
    ):
        self.db = farm_db or FarmDatabase()
        self.cache = cache

    def get_officer_overview(
        self,
        officer_id: str,
        officer_name: str = "Field Officer",
        filter_crop: Optional[str] = None,
        filter_risk: Optional[str] = None,
        filter_status: Optional[str] = None,
        date_range_days: int = 30,
        force_refresh: bool = False,
    ) -> FieldOfficerDashboardOverview:
        """Generate regional intelligence dashboard for authorized field officer."""
        cache_key = f"officer:{officer_id}:{filter_crop}:{filter_risk}:{filter_status}:{date_range_days}"
        if not force_refresh and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # 1. Fetch assigned territories
        territories: List[OfficerTerritoryScope] = self.db.get_field_officer_territory(officer_id)
        if not territories:
            # Officer has no assigned territory
            return self._build_empty_territory_overview(officer_id, officer_name)

        assigned_districts = list({t.district for t in territories})

        # Calculate date threshold
        date_from = None
        if date_range_days > 0:
            date_from = (datetime.now() - timedelta(days=date_range_days)).isoformat()

        # 2. Fetch cases within assigned territory
        cases = self.db.list_advisory_cases_for_territory(
            districts=assigned_districts,
            crop=filter_crop,
            risk=filter_risk,
            status=filter_status,
            date_from=date_from,
            limit=100,
        )

        # 3. Compute metrics
        total_cases = len(cases)
        high_risk_cases = sum(1 for c in cases if c.risk == "High")
        visits_due = sum(1 for c in cases if c.field_visit_required and c.status not in ("resolved", "closed"))
        resolved_cases = sum(1 for c in cases if c.status in ("resolved", "closed"))
        res_rate = round((resolved_cases / total_cases * 100), 1) if total_cases > 0 else 0.0

        unique_farmers = len({c.farmer_id for c in cases if c.farmer_id})

        metrics: Dict[str, MetricItem] = {}
        metrics["officer_assigned_districts"] = build_metric_item(
            "officer_assigned_districts",
            len(assigned_districts),
            display_value=f"{len(assigned_districts)} Districts",
            status=MetricStatus.NORMAL,
        )
        metrics["officer_registered_cases"] = build_metric_item(
            "officer_registered_cases",
            total_cases,
            display_value=f"{total_cases} Reported",
            status=MetricStatus.WARNING if total_cases > 20 else MetricStatus.NORMAL,
        )
        metrics["officer_high_risk_cases"] = build_metric_item(
            "officer_high_risk_cases",
            high_risk_cases,
            display_value=f"{high_risk_cases} Urgent",
            status=MetricStatus.CRITICAL if high_risk_cases > 5 else (
                MetricStatus.WARNING if high_risk_cases > 0 else MetricStatus.OPTIMAL
            ),
        )
        metrics["officer_field_visits_due"] = build_metric_item(
            "officer_field_visits_due",
            visits_due,
            display_value=f"{visits_due} Pending",
            status=MetricStatus.WARNING if visits_due > 5 else MetricStatus.NORMAL,
        )
        metrics["officer_resolution_rate"] = build_metric_item(
            "officer_resolution_rate",
            res_rate,
            unit="%",
            display_value=f"{res_rate:.1f}%",
            status=MetricStatus.OPTIMAL if res_rate >= 80 else (
                MetricStatus.NORMAL if res_rate >= 50 else MetricStatus.WARNING
            ),
        )
        metrics["officer_registered_farmers"] = build_metric_item(
            "officer_registered_farmers",
            unique_farmers,
            display_value=f"{unique_farmers} Farmers",
            status=MetricStatus.NORMAL,
        )

        # 4. Risk distribution
        risk_counts = Counter(c.risk for c in cases)
        risk_distribution = {
            "Low": risk_counts.get("Low", 0),
            "Medium": risk_counts.get("Medium", 0),
            "High": risk_counts.get("High", 0),
        }

        # 5. Crop distribution
        crop_counts = Counter(c.crop for c in cases)
        crop_distribution = [
            ChartPoint(label=crop, value=float(count), category="Reported Cases")
            for crop, count in crop_counts.most_common(8)
        ]

        # 6. Follow-up queue
        follow_up_queue = [
            c for c in cases if c.field_visit_required and c.status not in ("resolved", "closed")
        ]

        # 7. Village/Location Clusters with Privacy Suppression
        # Rule: If cluster has <3 cases, redact individual farmer identity in display
        loc_groups: Dict[str, List[AdvisoryCase]] = {}
        for c in cases:
            loc_key = c.district or c.location or "Regional Unspecified"
            loc_groups.setdefault(loc_key, []).append(c)

        village_clusters = []
        for loc_name, cluster_cases in loc_groups.items():
            cluster_size = len(cluster_cases)
            privacy_masked = cluster_size < 3
            sample_farmers = []
            for c in cluster_cases:
                if privacy_masked:
                    sample_farmers.append("Confidential Farmer")
                else:
                    sample_farmers.append(c.farmer_name)

            village_clusters.append({
                "location": loc_name,
                "case_count": cluster_size,
                "high_risk_count": sum(1 for c in cluster_cases if c.risk == "High"),
                "primary_crops": [item[0] for item in Counter(c.crop for c in cluster_cases).most_common(2)],
                "privacy_masked": privacy_masked,
                "sample_farmers": list(set(sample_farmers))[:3],
            })

        # 8. Regional Insights
        insights: List[InsightItem] = []
        if high_risk_cases > 0:
            insights.append(InsightItem(
                insight_id=f"ins_off_high_{officer_id}",
                category="regional_risk",
                title=f"{high_risk_cases} High-Risk Cases in Assigned Jurisdiction",
                description=f"There are {high_risk_cases} high-risk agricultural cases across {', '.join(assigned_districts)} requiring field agronomist follow-up.",
                severity=InsightSeverity.CRITICAL,
                source="AgriGuardian Regional Telemetry",
                suggested_action="Review follow-up queue and schedule on-site verification visits.",
            ))

        if crop_counts:
            top_crop, top_vol = crop_counts.most_common(1)[0]
            insights.append(InsightItem(
                insight_id=f"ins_off_crop_{top_crop}",
                category="crop_incidence",
                title=f"Dominant Query Crop: {top_crop}",
                description=f"{top_crop} represents {top_vol} reported case(s) ({round(top_vol / total_cases * 100, 1)}% of territory volume).",
                severity=InsightSeverity.INFO,
                source="AgriGuardian Case Registry",
                suggested_action=f"Ensure local agro-dealers maintain adequate stocks of approved {top_crop} crop protection formulations.",
            ))

        overview = FieldOfficerDashboardOverview(
            officer_id=officer_id,
            officer_name=officer_name,
            assigned_territory=territories,
            metrics=metrics,
            crop_distribution=crop_distribution,
            risk_distribution=risk_distribution,
            case_register=cases,
            follow_up_queue=follow_up_queue,
            village_clusters=village_clusters,
            insights=insights,
            data_freshness="Live Relational Case Register",
        )

        if self.cache:
            self.cache.set(cache_key, overview)

        return overview

    def _build_empty_territory_overview(
        self,
        officer_id: str,
        officer_name: str,
    ) -> FieldOfficerDashboardOverview:
        """Handle case where an officer has not yet been assigned any districts."""
        metrics: Dict[str, MetricItem] = {}
        for m_id in [
            "officer_assigned_districts",
            "officer_registered_cases",
            "officer_high_risk_cases",
            "officer_field_visits_due",
            "officer_resolution_rate",
            "officer_registered_farmers",
        ]:
            metrics[m_id] = build_metric_item(
                m_id,
                0,
                display_value="0 (Unassigned)",
                status=MetricStatus.UNAVAILABLE,
                is_available=False,
                missing_reason="No geographic territory currently assigned to this field officer account.",
            )

        insight = InsightItem(
            insight_id="ins_no_territory",
            category="account_provisioning",
            title="No Territory Assigned",
            description="Your field officer account has not been mapped to any agricultural districts. Please contact a platform administrator.",
            severity=InsightSeverity.WARNING,
            source="AgriGuardian Governance System",
            suggested_action="Request territory assignment for your state and district from the administrator.",
        )

        return FieldOfficerDashboardOverview(
            officer_id=officer_id,
            officer_name=officer_name,
            assigned_territory=[],
            metrics=metrics,
            crop_distribution=[],
            risk_distribution={"Low": 0, "Medium": 0, "High": 0},
            case_register=[],
            follow_up_queue=[],
            village_clusters=[],
            insights=[insight],
            data_freshness="Territory Unassigned",
        )

    def record_case_follow_up(
        self,
        case_id: str,
        officer_id: str,
        officer_name: str,
        notes: str,
        action_taken: Optional[str] = None,
        new_status: Optional[str] = None,
        is_admin: bool = False,
    ) -> CaseFollowUpRecord:
        """Record agronomist follow-up audit log with strict territory permission check."""
        case = self.db.get_advisory_case(case_id)
        if not case:
            raise ValueError(f"Advisory case '{case_id}' not found.")

        # Unless admin, verify territory assignment
        if not is_admin:
            territories = self.db.get_field_officer_territory(officer_id)
            assigned_districts = {t.district.lower() for t in territories}
            case_district = (case.district or "").lower()
            if case_district not in assigned_districts:
                raise PermissionError(
                    f"Access Denied: Case '{case_id}' belongs to district '{case.district or 'Unknown'}', "
                    f"which is outside officer '{officer_id}' assigned territory."
                )

        # Create follow-up record
        record = self.db.create_case_follow_up(
            case_id=case_id,
            officer_id=officer_id,
            officer_name=officer_name,
            notes=notes,
            action_taken=action_taken,
            status_transition=new_status,
        )

        # Update case status if changed
        if new_status and new_status != case.status:
            self.db.update_advisory_case_status(case_id=case_id, new_status=new_status)

        # Invalidate officer cache
        if self.cache:
            self.cache.invalidate_prefix("officer:")

        return record

    def list_case_follow_ups(
        self,
        case_id: str,
        officer_id: str,
        is_admin: bool = False,
    ) -> List[CaseFollowUpRecord]:
        """List historical follow-ups for a case, enforcing territory access."""
        case = self.db.get_advisory_case(case_id)
        if not case:
            raise ValueError(f"Advisory case '{case_id}' not found.")

        if not is_admin:
            territories = self.db.get_field_officer_territory(officer_id)
            assigned_districts = {t.district.lower() for t in territories}
            case_district = (case.district or "").lower()
            if case_district not in assigned_districts:
                raise PermissionError(
                    f"Access Denied: Case '{case_id}' is outside your assigned territory."
                )

        return self.db.list_case_follow_ups(case_id)
