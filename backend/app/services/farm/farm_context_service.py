"""Centralized Farm Context Resolution Service bridging farmer profile, active farm, crop cycle, and soil to advisory engines."""

from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import (
    CropCycle,
    Farm,
    FarmActivity,
    FarmContext,
    FarmerProfile,
    IrrigationProfile,
    Plot,
    SoilProfile,
)
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService
from backend.app.services.farm.soil_irrigation_service import SoilIrrigationService

logger = get_logger("FarmContextService")


class FarmContextService:
    """Resolves authenticated farmer context, active farm, plot, and crop stage for personalized agricultural decision-support."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        profile_service: Optional[FarmerProfileService] = None,
        farm_service: Optional[FarmService] = None,
        plot_service: Optional[PlotService] = None,
        cycle_service: Optional[CropCycleService] = None,
        soil_service: Optional[SoilIrrigationService] = None,
        calendar_service: Optional[CropCalendarService] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.farm_db = self.repo
        self.profile_service = profile_service or FarmerProfileService(self.repo)
        self.farm_service = farm_service or FarmService(self.repo)
        self.plot_service = plot_service or PlotService(self.repo)
        self.cycle_service = cycle_service or CropCycleService(self.repo)
        self.soil_service = soil_service or SoilIrrigationService(self.repo)
        self.calendar_service = calendar_service or CropCalendarService(self.repo)

    def resolve_context(
        self,
        farmer_id: Optional[str] = None,
        farm_id_override: Optional[str] = None,
        plot_id_override: Optional[str] = None,
    ) -> FarmContext:
        """Resolve full hierarchical context for farmer."""
        if not farmer_id:
            return FarmContext(
                is_authenticated=False,
                is_ambiguous=False,
                ambiguity_reason="No farmer identity provided (Guest/Anonymous mode).",
            )

        # 1. Fetch Profile
        profile = self.profile_service.get_profile(farmer_id)
        if not profile:
            return FarmContext(
                is_authenticated=False,
                is_ambiguous=False,
                ambiguity_reason=f"Farmer profile '{farmer_id}' does not exist.",
            )

        all_farms = self.farm_service.list_farms(farmer_id)

        # 2. Resolve Active Farm
        active_farm: Optional[Farm] = None
        is_ambiguous = False
        ambiguity_reason = None

        if farm_id_override:
            active_farm = self.farm_service.get_farm(farm_id=farm_id_override, farmer_id=farmer_id)
            if not active_farm:
                return FarmContext(
                    is_authenticated=True,
                    farmer=profile,
                    all_farms_count=len(all_farms),
                    is_ambiguous=True,
                    ambiguity_reason=f"Specified farm '{farm_id_override}' not found or not owned by farmer.",
                )
        else:
            active_farm = self.farm_service.get_active_farm(farmer_id)
            if not active_farm and len(all_farms) > 1:
                is_ambiguous = True
                ambiguity_reason = (
                    f"Farmer has {len(all_farms)} farms, but no active farm has been designated. "
                    "Please specify which farm context to use."
                )

        if not active_farm:
            return FarmContext(
                is_authenticated=True,
                farmer=profile,
                all_farms_count=len(all_farms),
                active_farm=None,
                is_ambiguous=is_ambiguous,
                ambiguity_reason=ambiguity_reason,
            )

        # 3. Resolve Plots
        active_plots = self.plot_service.list_plots_by_farm(farm_id=active_farm.farm_id, farmer_id=farmer_id)

        # 4. Resolve Active Crop Cycles
        active_cycles = self.cycle_service.get_active_cycles_by_farm(farm_id=active_farm.farm_id, farmer_id=farmer_id)

        current_cycle: Optional[CropCycle] = None
        if plot_id_override:
            for c in active_cycles:
                if c.plot_id == plot_id_override:
                    current_cycle = c
                    break
        elif active_cycles:
            current_cycle = active_cycles[0]

        # 5. Resolve Soil & Irrigation Profiles for selected plot
        target_plot_id = current_cycle.plot_id if current_cycle else (active_plots[0].plot_id if active_plots else None)
        soil_profile: Optional[SoilProfile] = None
        irrigation_profile: Optional[IrrigationProfile] = None

        if target_plot_id:
            soil_profile = self.soil_service.get_soil_profile(plot_id=target_plot_id, farmer_id=farmer_id)
            irrigation_profile = self.soil_service.get_irrigation_profile(plot_id=target_plot_id, farmer_id=farmer_id)

        # 6. Resolve Due Activities
        due_activities = self.calendar_service.list_activities(
            farmer_id=farmer_id,
            farm_id=active_farm.farm_id,
            status="due",
        )

        return FarmContext(
            is_authenticated=True,
            farmer=profile,
            active_farm=active_farm,
            all_farms_count=len(all_farms),
            active_plots=active_plots,
            current_cycle=current_cycle,
            all_active_cycles=active_cycles,
            soil_profile=soil_profile,
            irrigation_profile=irrigation_profile,
            due_activities=due_activities,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
        )

    def get_weather_coordinates(self, ctx: FarmContext) -> Optional[Tuple[float, float, str]]:
        """Extract GPS coordinates or district name from farm context for weather intelligence."""
        if not ctx.active_farm:
            return None
        farm = ctx.active_farm
        if farm.latitude is not None and farm.longitude is not None:
            loc_label = f"{farm.farm_name} ({farm.district}, {farm.state})"
            return (farm.latitude, farm.longitude, loc_label)
        return None

    def get_market_location_query(self, ctx: FarmContext) -> Optional[Dict[str, Any]]:
        """Extract geographic and administrative location attributes for APMC mandi discovery."""
        if not ctx.active_farm:
            return None
        farm = ctx.active_farm
        return {
            "state": farm.state,
            "district": farm.district,
            "location_query": f"{farm.district}, {farm.state}",
            "latitude": farm.latitude,
            "longitude": farm.longitude,
        }

    def get_agronomic_context_payload(self, ctx: FarmContext) -> Dict[str, Any]:
        """Formulate complete agronomic payload for recommendation rules and personalized advisory."""
        payload: Dict[str, Any] = {
            "is_personalized": ctx.is_authenticated and ctx.has_active_farm,
            "farmer_name": ctx.farmer.full_name if ctx.farmer else "Farmer",
            "preferred_language": ctx.farmer.preferred_language if ctx.farmer else "hi-IN",
            "has_farm": ctx.has_active_farm,
        }

        if ctx.active_farm:
            payload["farm_name"] = ctx.active_farm.farm_name
            payload["farm_state"] = ctx.active_farm.state
            payload["farm_district"] = ctx.active_farm.district
            payload["farm_area_acres"] = ctx.active_farm.total_area_acres

        if ctx.current_cycle:
            cycle = ctx.current_cycle
            payload["crop"] = cycle.crop_name
            payload["variety"] = cycle.variety_name
            payload["growth_stage"] = cycle.current_stage.value
            payload["estimated_stage"] = cycle.estimated_stage.value
            payload["days_since_sowing"] = cycle.days_since_sowing
            payload["season"] = cycle.season.value
            payload["stage_confirmed"] = cycle.stage_confirmed_by_farmer

        if ctx.soil_profile:
            sp = ctx.soil_profile
            payload["soil_type"] = sp.soil_type.value
            payload["soil_ph"] = sp.ph_level
            payload["soil_nitrogen"] = sp.nitrogen_kg_ha
            payload["soil_phosphorus"] = sp.phosphorus_kg_ha
            payload["soil_potassium"] = sp.potassium_kg_ha
            payload["soil_is_lab_tested"] = sp.is_lab_tested

        if ctx.irrigation_profile:
            ip = ctx.irrigation_profile
            payload["irrigation_source"] = ip.primary_source
            payload["irrigation_method"] = ip.irrigation_method.value
            payload["water_availability"] = ip.water_availability.value

        payload["due_tasks_count"] = len(ctx.due_activities)
        return payload

