"""Farm Activity and In-App Crop Calendar Service with growth-stage grounded agronomic task generation."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    CropCycle,
    CropGrowthStage,
    FarmActivity,
    FarmActivityCreate,
    FarmActivityUpdate,
)

logger = get_logger("CropCalendarService")


# Grounded stage activity recommendations based on ICAR agronomic packages
STAGE_ACTIVITY_BLUEPRINT: Dict[str, Dict[CropGrowthStage, List[Dict[str, Any]]]] = {
    "wheat": {
        CropGrowthStage.SOWING_GERMINATION: [
            {
                "type": ActivityType.FERTILIZER_APPLICATION,
                "title": "Basal Fertilizer Application",
                "desc": "Apply recommended basal dose: 50 kg DAP + 30 kg MOP + 25 kg Urea per acre at sowing.",
                "days_offset": 0,
            },
            {
                "type": ActivityType.WEEDING,
                "title": "Pre-Emergence Weed Control",
                "desc": "Spray Pendimethalin 30% EC within 48 hours of sowing on moist soil.",
                "days_offset": 2,
            },
        ],
        CropGrowthStage.SEEDLING_EMERGENCE: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Crown Root Initiation (CRI) Critical Irrigation",
                "desc": "CRI occurs 20-25 days after sowing. Most critical irrigation; moisture stress now reduces tillering by up to 30%.",
                "days_offset": 21,
            },
            {
                "type": ActivityType.FERTILIZER_APPLICATION,
                "title": "First Top Dressing of Urea",
                "desc": "Broadcast 35-40 kg Urea per acre immediately following the CRI irrigation.",
                "days_offset": 24,
            },
        ],
        CropGrowthStage.VEGETATIVE_TILLERING: [
            {
                "type": ActivityType.WEEDING,
                "title": "Post-Emergence Weed Management",
                "desc": "Inspect for broadleaf and grassy weeds (Phalaris minor). Apply selective herbicide if weed pressure exceeds threshold.",
                "days_offset": 35,
            },
            {
                "type": ActivityType.IRRIGATION,
                "title": "Tillering & Jointing Irrigation",
                "desc": "Provide secondary irrigation 40-45 days after sowing to support active tillers.",
                "days_offset": 45,
            },
        ],
        CropGrowthStage.FLOWERING_REPRODUCTIVE: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Booting / Flowering Stage Irrigation",
                "desc": "Ensure adequate root zone moisture during spikelet emergence. Avoid heavy irrigation during high winds to prevent lodging.",
                "days_offset": 75,
            },
            {
                "type": ActivityType.PESTICIDE_SPRAY,
                "title": "Foliar Rust & Blight Scouting",
                "desc": "Inspect flag leaves for yellow rust pustules or leaf blight. Spray Propiconazole 25% EC if rust is detected.",
                "days_offset": 80,
            },
        ],
        CropGrowthStage.GRAIN_FILLING_MATURITY: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Milking & Dough Stage Light Irrigation",
                "desc": "Provide light irrigation to prevent premature drying and shriveled grains.",
                "days_offset": 100,
            },
        ],
        CropGrowthStage.HARVEST_READY: [
            {
                "type": ActivityType.HARVESTING,
                "title": "Wheat Grain Harvest",
                "desc": "Harvest when straw turns yellow-golden and grain moisture drops to 14-18%.",
                "days_offset": 130,
            },
            {
                "type": ActivityType.POST_HARVEST_STORAGE,
                "title": "Sun Drying and Clean Storage",
                "desc": "Sun-dry harvested grain to below 12% moisture before hermetic bag or metal silo storage.",
                "days_offset": 135,
            },
        ],
    },
    "rice": {
        CropGrowthStage.SOWING_GERMINATION: [
            {
                "type": ActivityType.LAND_PREPARATION,
                "title": "Nursery Bed Preparation and Seed Treatment",
                "desc": "Treat seeds with Carbendazim (2g/kg) and prepare raised nursery beds.",
                "days_offset": 0,
            },
        ],
        CropGrowthStage.SEEDLING_EMERGENCE: [
            {
                "type": ActivityType.TRANSPLANTING,
                "title": "Seedling Transplantation to Main Field",
                "desc": "Transplant 20-25 day old seedlings at 2-3 seedlings per hill with 20x15 cm spacing in puddled field.",
                "days_offset": 25,
            },
            {
                "type": ActivityType.FERTILIZER_APPLICATION,
                "title": "Basal NPK & Zinc Application",
                "desc": "Incorporate full dose of P, K, and Zinc Sulphate along with 1/3rd Nitrogen into puddled soil.",
                "days_offset": 26,
            },
        ],
        CropGrowthStage.VEGETATIVE_TILLERING: [
            {
                "type": ActivityType.FERTILIZER_APPLICATION,
                "title": "Active Tillering Urea Top-Dressing",
                "desc": "Broadcast 1/3rd Nitrogen dose (Urea) at active tillering stage (20-25 days after transplanting).",
                "days_offset": 45,
            },
            {
                "type": ActivityType.PESTICIDE_SPRAY,
                "title": "Stem Borer & Leaf Folder Monitoring",
                "desc": "Scout for dead hearts or folded leaves. Install pheromone traps for yellow stem borer.",
                "days_offset": 50,
            },
        ],
        CropGrowthStage.FLOWERING_REPRODUCTIVE: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Maintain Standing Water at Panicle Initiation",
                "desc": "Maintain 2-5 cm standing water during panicle initiation and flowering. Moisture stress now causes sterile spikelets.",
                "days_offset": 75,
            },
        ],
        CropGrowthStage.GRAIN_FILLING_MATURITY: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Terminal Drainage Before Harvest",
                "desc": "Drain field completely 10-12 days before anticipated harvest to facilitate mechanized cutting and uniform ripening.",
                "days_offset": 115,
            },
        ],
        CropGrowthStage.HARVEST_READY: [
            {
                "type": ActivityType.HARVESTING,
                "title": "Paddy Harvesting & Threshing",
                "desc": "Harvest when 85% of grains in the panicle have turned straw-colored.",
                "days_offset": 130,
            },
        ],
    },
    "tomato": {
        CropGrowthStage.SOWING_GERMINATION: [
            {
                "type": ActivityType.LAND_PREPARATION,
                "title": "Nursery Bed Preparation and Soil Solarization",
                "desc": "Drench nursery beds with Trichoderma viride to prevent damping-off disease.",
                "days_offset": 0,
            },
        ],
        CropGrowthStage.SEEDLING_EMERGENCE: [
            {
                "type": ActivityType.TRANSPLANTING,
                "title": "Tomato Seedling Transplantation",
                "desc": "Transplant 25-30 day sturdy seedlings on raised ridges or plastic mulch with drip irrigation.",
                "days_offset": 30,
            },
        ],
        CropGrowthStage.VEGETATIVE_TILLERING: [
            {
                "type": ActivityType.INTER_CULTIVATION,
                "title": "Staking & Trellising",
                "desc": "Install bamboo or string stakes to lift indeterminate tomato foliage off damp soil.",
                "days_offset": 45,
            },
            {
                "type": ActivityType.PESTICIDE_SPRAY,
                "title": "Early Blight & Whitefly Scouting",
                "desc": "Inspect lower leaves for concentric brown rings (Alternaria) and yellow sticky traps for whiteflies.",
                "days_offset": 55,
            },
        ],
        CropGrowthStage.FLOWERING_REPRODUCTIVE: [
            {
                "type": ActivityType.FERTILIZER_APPLICATION,
                "title": "Fertigation with Calcium Nitrate & Boron",
                "desc": "Apply Calcium Nitrate through drip to prevent Blossom End Rot and improve flower set.",
                "days_offset": 70,
            },
        ],
        CropGrowthStage.GRAIN_FILLING_MATURITY: [
            {
                "type": ActivityType.IRRIGATION,
                "title": "Regulated Deficit Drip Irrigation",
                "desc": "Maintain consistent soil moisture; irregular irrigation causes fruit cracking and skin splitting.",
                "days_offset": 90,
            },
        ],
        CropGrowthStage.HARVEST_READY: [
            {
                "type": ActivityType.HARVESTING,
                "title": "Tomato Picking at Breaker/Pink Stage",
                "desc": "Harvest fruits at breaker or pink stage for distant APMC mandi transport.",
                "days_offset": 115,
            },
        ],
    },
}


class CropCalendarService:
    """Manages farmer activity calendar, due tasks, completions, and stage-specific agronomic recommendations."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def create_activity(
        self,
        farm_id: str,
        farmer_id: str,
        activity_data: FarmActivityCreate,
        plot_id: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> FarmActivity:
        """Schedule a new farm activity verifying tenant ownership."""
        farm = self.repo.get_farm(farm_id=farm_id, farmer_id=farmer_id)
        if not farm:
            raise ValueError(f"Farm '{farm_id}' not found or not owned by farmer '{farmer_id}'.")

        now_str = datetime.now().isoformat()
        activity_id = f"act_{uuid.uuid4().hex[:12]}"

        # Determine initial status (if scheduled for today or earlier, mark as DUE)
        today_str = datetime.now().strftime("%Y-%m-%d")
        initial_status = ActivityStatus.DUE if activity_data.scheduled_date <= today_str else ActivityStatus.PLANNED

        activity = FarmActivity(
            activity_id=activity_id,
            farm_id=farm_id,
            plot_id=plot_id,
            cycle_id=cycle_id,
            farmer_id=farmer_id,
            activity_type=activity_data.activity_type,
            activity_title=activity_data.activity_title.strip(),
            description=activity_data.description.strip() if activity_data.description else None,
            scheduled_date=activity_data.scheduled_date,
            completed_date=None,
            status=initial_status,
            stage_tag=activity_data.stage_tag,
            cost_incurred=activity_data.cost_incurred,
            notes=activity_data.notes,
            created_at=now_str,
            updated_at=now_str,
        )

        created = self.repo.create_farm_activity(activity)
        logger.info(f"Created farm activity '{activity.activity_title}' ({activity_id}) on farm {farm_id}")
        return created

    def complete_activity(
        self,
        activity_id: str,
        farmer_id: str,
        completed_date: Optional[str] = None,
        cost_incurred: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Optional[FarmActivity]:
        """Mark farm task as completed verifying tenant ownership."""
        activity = self.repo.get_farm_activity(activity_id=activity_id, farmer_id=farmer_id)
        if not activity:
            return None

        now_str = datetime.now().isoformat()
        act_dict = activity.model_dump()

        act_dict["status"] = ActivityStatus.COMPLETED
        act_dict["completed_date"] = completed_date or datetime.now().strftime("%Y-%m-%d")
        if cost_incurred is not None:
            act_dict["cost_incurred"] = cost_incurred
        if notes:
            act_dict["notes"] = f"{act_dict.get('notes') or ''}\n[Completion Note]: {notes}".strip()
        act_dict["updated_at"] = now_str

        updated_act = FarmActivity(**act_dict)
        return self.repo.update_farm_activity(updated_act)

    def list_activities(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[FarmActivity]:
        """List activities with automatic due-date synchronization."""
        if farm_id:
            raw_activities = self.repo.get_activities_by_farm(farm_id=farm_id, farmer_id=farmer_id, status=status)
        else:
            raw_activities = self.repo.get_activities_by_farmer(farmer_id=farmer_id, status=status)

        today_str = datetime.now().strftime("%Y-%m-%d")
        synced: List[FarmActivity] = []

        for act in raw_activities:
            # Check if planned activity is now due
            if act.status == ActivityStatus.PLANNED and act.scheduled_date <= today_str:
                act_dict = act.model_dump()
                act_dict["status"] = ActivityStatus.DUE
                act_dict["updated_at"] = datetime.now().isoformat()
                act = self.repo.update_farm_activity(FarmActivity(**act_dict))
            synced.append(act)

        return synced

    def generate_grounded_stage_activities(
        self,
        cycle: CropCycle,
        farmer_id: str,
    ) -> List[FarmActivity]:
        """Automatically schedule recommended agronomic activities matching current crop growth stage."""
        crop_clean = cycle.crop_name.lower().strip()
        blueprint = STAGE_ACTIVITY_BLUEPRINT.get(crop_clean)

        if not blueprint:
            for key, val in STAGE_ACTIVITY_BLUEPRINT.items():
                if key in crop_clean or crop_clean in key:
                    blueprint = val
                    break

        if not blueprint:
            return []

        stage_tasks = blueprint.get(cycle.current_stage, [])
        created_tasks: List[FarmActivity] = []

        # Check existing activities to prevent duplicates
        existing = self.repo.get_activities_by_farm(farm_id=cycle.farm_id, farmer_id=farmer_id)
        existing_titles = {e.activity_title.lower() for e in existing}

        from datetime import date, timedelta
        base_date = datetime.now().date()

        for item in stage_tasks:
            if item["title"].lower() in existing_titles:
                continue

            scheduled_d = (base_date + timedelta(days=item.get("days_offset", 0))).strftime("%Y-%m-%d")

            task_create = FarmActivityCreate(
                activity_type=item["type"],
                activity_title=item["title"],
                description=item["desc"],
                scheduled_date=scheduled_d,
                stage_tag=cycle.current_stage.value,
                cost_incurred=None,
                notes="Recommended by AgriGuardian Agronomic Engine for current crop stage.",
            )

            created = self.create_activity(
                farm_id=cycle.farm_id,
                farmer_id=farmer_id,
                activity_data=task_create,
                plot_id=cycle.plot_id,
                cycle_id=cycle.cycle_id,
            )
            created_tasks.append(created)

        logger.info(f"Generated {len(created_tasks)} stage tasks for cycle {cycle.cycle_id} ({cycle.current_stage.value})")
        return created_tasks

