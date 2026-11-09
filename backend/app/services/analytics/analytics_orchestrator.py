"""Unified Analytics Orchestrator dispatching role-specific dashboards and managing shared cache."""

import logging
from typing import Any, Dict, List, Optional

from backend.app.repositories.farm_db import FarmDatabase
from backend.app.schemas.analytics import (
    AdminDashboardOverview,
    CaseFollowUpRecord,
    FarmerDashboardOverview,
    FieldOfficerDashboardOverview,
    OfficerTerritoryScope,
)
from backend.app.schemas.farm import UserRole
from backend.app.services.analytics.analytics_cache import AnalyticsCache
from backend.app.services.analytics.farmer_analytics import FarmerAnalyticsService
from backend.app.services.analytics.field_officer_analytics import FieldOfficerAnalyticsService
from backend.app.services.analytics.platform_analytics import PlatformAnalyticsService

logger = logging.getLogger(__name__)


class AnalyticsOrchestrator:
    """Central gateway for role-based analytics, reporting, and operational intelligence."""

    def __init__(
        self,
        farm_db: Optional[FarmDatabase] = None,
        cache_ttl_seconds: int = 180,
        weather_service: Optional[Any] = None,
        market_service: Optional[Any] = None,
    ):
        self.db = farm_db or FarmDatabase()
        self.cache = AnalyticsCache(default_ttl_seconds=cache_ttl_seconds)

        self.farmer_analytics = FarmerAnalyticsService(
            farm_db=self.db,
            cache=self.cache,
            weather_service=weather_service,
            market_service=market_service,
        )
        self.field_officer_analytics = FieldOfficerAnalyticsService(
            farm_db=self.db,
            cache=self.cache,
        )
        self.platform_analytics = PlatformAnalyticsService(
            farm_db=self.db,
            cache=self.cache,
        )

    # --- Role-Based Dashboard Access ---

    def get_farmer_dashboard(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> FarmerDashboardOverview:
        """Retrieve personalized dashboard for a farmer."""
        return self.farmer_analytics.get_farmer_overview(
            farmer_id=farmer_id,
            farm_id=farm_id,
            force_refresh=force_refresh,
        )

    def get_field_officer_dashboard(
        self,
        officer_id: str,
        officer_name: str = "Field Officer",
        filter_crop: Optional[str] = None,
        filter_risk: Optional[str] = None,
        filter_status: Optional[str] = None,
        date_range_days: int = 30,
        force_refresh: bool = False,
    ) -> FieldOfficerDashboardOverview:
        """Retrieve regional intelligence dashboard for a field officer."""
        return self.field_officer_analytics.get_officer_overview(
            officer_id=officer_id,
            officer_name=officer_name,
            filter_crop=filter_crop,
            filter_risk=filter_risk,
            filter_status=filter_status,
            date_range_days=date_range_days,
            force_refresh=force_refresh,
        )

    def get_platform_admin_dashboard(
        self,
        admin_id: str = "adm_system",
        admin_name: str = "Platform Administrator",
        force_refresh: bool = False,
    ) -> AdminDashboardOverview:
        """Retrieve system operations and telemetry dashboard for platform administrators."""
        return self.platform_analytics.get_admin_overview(
            admin_id=admin_id,
            admin_name=admin_name,
            force_refresh=force_refresh,
        )

    # --- Case & Territory Management ---

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
        """Record agronomist follow-up audit log."""
        return self.field_officer_analytics.record_case_follow_up(
            case_id=case_id,
            officer_id=officer_id,
            officer_name=officer_name,
            notes=notes,
            action_taken=action_taken,
            new_status=new_status,
            is_admin=is_admin,
        )

    def list_case_follow_ups(
        self,
        case_id: str,
        officer_id: str,
        is_admin: bool = False,
    ) -> List[CaseFollowUpRecord]:
        """List historical follow-ups for a case."""
        return self.field_officer_analytics.list_case_follow_ups(
            case_id=case_id,
            officer_id=officer_id,
            is_admin=is_admin,
        )

    def assign_officer_territory(
        self,
        officer_id: str,
        state: str,
        district: str,
        block: Optional[str] = None,
    ) -> OfficerTerritoryScope:
        """Assign geographic district to a field officer and invalidate cache."""
        scope = self.db.assign_field_officer_territory(
            officer_id=officer_id,
            state=state,
            district=district,
            block=block,
        )
        self.cache.invalidate_prefix(f"officer:{officer_id}")
        return scope

    def get_officer_territory(self, officer_id: str) -> List[OfficerTerritoryScope]:
        """Get assigned districts for an officer."""
        return self.db.get_field_officer_territory(officer_id)

    def invalidate_all(self) -> None:
        """Flush the cache."""
        self.cache.clear()
