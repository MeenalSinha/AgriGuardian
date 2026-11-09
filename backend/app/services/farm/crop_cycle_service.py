"""Crop Lifecycle Tracking Service with agronomic stage estimation and historical preservation."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.farm import (
    CropCycle,
    CropCycleCreate,
    CropCycleStatus,
    CropCycleUpdate,
    CropGrowthStage,
    CropStageTransitionRequest,
)

logger = get_logger("CropCycleService")


# Empirical growth stage thresholds (in days since sowing) for major Indian crops
CROP_PHENOLOGY_MAP: Dict[str, List[Tuple[int, CropGrowthStage]]] = {
    "wheat": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),       # CRI stage (~21d)
        (65, CropGrowthStage.VEGETATIVE_TILLERING),     # Tillering & Jointing
        (90, CropGrowthStage.FLOWERING_REPRODUCTIVE),   # Booting & Flowering
        (120, CropGrowthStage.GRAIN_FILLING_MATURITY),  # Milking & Dough
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "rice": [
        (15, CropGrowthStage.SOWING_GERMINATION),       # Nursery
        (30, CropGrowthStage.SEEDLING_EMERGENCE),       # Transplanting
        (70, CropGrowthStage.VEGETATIVE_TILLERING),     # Active Tillering & Panicle
        (95, CropGrowthStage.FLOWERING_REPRODUCTIVE),   # Heading & Flowering
        (125, CropGrowthStage.GRAIN_FILLING_MATURITY),  # Grain filling & Ripening
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "paddy": [
        (15, CropGrowthStage.SOWING_GERMINATION),
        (30, CropGrowthStage.SEEDLING_EMERGENCE),
        (70, CropGrowthStage.VEGETATIVE_TILLERING),
        (95, CropGrowthStage.FLOWERING_REPRODUCTIVE),
        (125, CropGrowthStage.GRAIN_FILLING_MATURITY),
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "cotton": [
        (15, CropGrowthStage.SOWING_GERMINATION),
        (40, CropGrowthStage.SEEDLING_EMERGENCE),
        (75, CropGrowthStage.VEGETATIVE_TILLERING),     # Squaring
        (115, CropGrowthStage.FLOWERING_REPRODUCTIVE),  # Flowering & Boll Formation
        (150, CropGrowthStage.GRAIN_FILLING_MATURITY),  # Boll Development
        (999, CropGrowthStage.HARVEST_READY),          # Boll Bursting
    ],
    "maize": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),
        (55, CropGrowthStage.VEGETATIVE_TILLERING),     # Knee-high stage
        (75, CropGrowthStage.FLOWERING_REPRODUCTIVE),   # Tasseling & Silking
        (95, CropGrowthStage.GRAIN_FILLING_MATURITY),   # Grain filling
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "mustard": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),
        (50, CropGrowthStage.VEGETATIVE_TILLERING),     # Rosette stage
        (75, CropGrowthStage.FLOWERING_REPRODUCTIVE),   # Flowering
        (105, CropGrowthStage.GRAIN_FILLING_MATURITY),  # Siliqua development
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "tomato": [
        (15, CropGrowthStage.SOWING_GERMINATION),
        (35, CropGrowthStage.SEEDLING_EMERGENCE),
        (65, CropGrowthStage.VEGETATIVE_TILLERING),
        (90, CropGrowthStage.FLOWERING_REPRODUCTIVE),
        (115, CropGrowthStage.GRAIN_FILLING_MATURITY),
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "potato": [
        (15, CropGrowthStage.SOWING_GERMINATION),
        (35, CropGrowthStage.SEEDLING_EMERGENCE),
        (60, CropGrowthStage.VEGETATIVE_TILLERING),     # Tuber initiation
        (85, CropGrowthStage.GRAIN_FILLING_MATURITY),   # Tuber bulking
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "soybean": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),
        (45, CropGrowthStage.VEGETATIVE_TILLERING),
        (70, CropGrowthStage.FLOWERING_REPRODUCTIVE),
        (90, CropGrowthStage.GRAIN_FILLING_MATURITY),
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "chickpea": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),
        (50, CropGrowthStage.VEGETATIVE_TILLERING),
        (75, CropGrowthStage.FLOWERING_REPRODUCTIVE),   # Pod formation
        (100, CropGrowthStage.GRAIN_FILLING_MATURITY),
        (999, CropGrowthStage.HARVEST_READY),
    ],
    "gram": [
        (10, CropGrowthStage.SOWING_GERMINATION),
        (25, CropGrowthStage.SEEDLING_EMERGENCE),
        (50, CropGrowthStage.VEGETATIVE_TILLERING),
        (75, CropGrowthStage.FLOWERING_REPRODUCTIVE),
        (100, CropGrowthStage.GRAIN_FILLING_MATURITY),
        (999, CropGrowthStage.HARVEST_READY),
    ],
}


class CropCycleService:
    """Manages crop lifecycle, growth stage estimation, stage transitions, and rotation history."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def create_crop_cycle(self, plot_id: str, farmer_id: str, cycle_data: CropCycleCreate) -> CropCycle:
        """Create a new crop cycle on a plot verifying tenant ownership."""
        plot = self.repo.get_plot(plot_id=plot_id, farmer_id=farmer_id)
        if not plot:
            raise ValueError(f"Plot '{plot_id}' not found or not owned by farmer '{farmer_id}'.")

        # Check for existing active cycle on this plot
        existing_active = self.repo.get_active_crop_cycle_by_plot(plot_id=plot_id, farmer_id=farmer_id)
        if existing_active:
            raise ValueError(
                f"Plot '{plot.plot_name}' already has an active crop cycle ({existing_active.crop_name}, "
                f"Cycle ID: {existing_active.cycle_id}). Complete or harvest earlier cycle before starting a new one."
            )

        now_str = datetime.now().isoformat()
        cycle_id = f"cyc_{uuid.uuid4().hex[:12]}"

        # Calculate estimated stage based on elapsed days since sowing
        estimated_stage, days_elapsed, target_harvest = self.estimate_growth_stage(
            cycle_data.crop_name,
            cycle_data.sowing_date,
        )

        effective_harvest_date = cycle_data.estimated_harvest_date or target_harvest

        cycle = CropCycle(
            cycle_id=cycle_id,
            plot_id=plot_id,
            farm_id=plot.farm_id,
            farmer_id=farmer_id,
            crop_name=cycle_data.crop_name.strip(),
            variety_name=cycle_data.variety_name.strip() if cycle_data.variety_name else None,
            season=cycle_data.season,
            sowing_date=cycle_data.sowing_date,
            transplanting_date=cycle_data.transplanting_date,
            estimated_harvest_date=effective_harvest_date,
            actual_harvest_date=None,
            current_stage=cycle_data.current_stage or estimated_stage,
            estimated_stage=estimated_stage,
            days_since_sowing=days_elapsed,
            stage_confirmed_by_farmer=False,
            target_yield_quintals=cycle_data.target_yield_quintals,
            actual_yield_quintals=None,
            status=CropCycleStatus.ACTIVE,
            notes=cycle_data.notes,
            created_at=now_str,
            updated_at=now_str,
        )

        created = self.repo.create_crop_cycle(cycle)
        logger.info(f"Created crop cycle '{cycle.crop_name}' ({cycle_id}) on plot {plot_id}")
        return created

    def get_crop_cycle(self, cycle_id: str, farmer_id: str) -> Optional[CropCycle]:
        """Fetch crop cycle and dynamically enrich current days elapsed and estimated stage."""
        cycle = self.repo.get_crop_cycle(cycle_id=cycle_id, farmer_id=farmer_id)
        if not cycle:
            return None
        return self._enrich_cycle(cycle)

    def get_active_cycles_by_farm(self, farm_id: str, farmer_id: str) -> List[CropCycle]:
        """Fetch all active crop cycles for a farm."""
        cycles = self.repo.get_active_crop_cycles_by_farm(farm_id=farm_id, farmer_id=farmer_id)
        return [self._enrich_cycle(c) for c in cycles]

    def get_active_cycle_by_plot(self, plot_id: str, farmer_id: str) -> Optional[CropCycle]:
        """Fetch active crop cycle for a plot."""
        cycle = self.repo.get_active_crop_cycle_by_plot(plot_id=plot_id, farmer_id=farmer_id)
        if not cycle:
            return None
        return self._enrich_cycle(cycle)

    def list_crop_cycles_by_farmer(self, farmer_id: str, status: Optional[str] = None) -> List[CropCycle]:
        """List crop cycles (active, completed, or all) for farmer, preserving rotation history."""
        cycles = self.repo.get_all_crop_cycles_by_farmer(farmer_id=farmer_id, status=status)
        return [self._enrich_cycle(c) for c in cycles]

    def transition_stage(
        self,
        cycle_id: str,
        farmer_id: str,
        req: CropStageTransitionRequest,
    ) -> Optional[CropCycle]:
        """Update crop stage, handling harvest completion without overwriting historical records."""
        cycle = self.get_crop_cycle(cycle_id=cycle_id, farmer_id=farmer_id)
        if not cycle:
            return None

        now_str = datetime.now().isoformat()
        cycle_dict = cycle.model_dump()

        cycle_dict["current_stage"] = req.new_stage
        cycle_dict["stage_confirmed_by_farmer"] = True
        cycle_dict["updated_at"] = now_str

        if req.notes:
            prev_notes = cycle_dict.get("notes") or ""
            cycle_dict["notes"] = f"{prev_notes}\n[Stage update to {req.new_stage.value}]: {req.notes}".strip()

        # Handle harvest completion
        if req.new_stage in (CropGrowthStage.HARVESTED,):
            cycle_dict["status"] = CropCycleStatus.COMPLETED
            cycle_dict["actual_harvest_date"] = req.actual_date or datetime.now().strftime("%Y-%m-%d")
            if req.yield_quintals is not None:
                cycle_dict["actual_yield_quintals"] = req.yield_quintals

        updated_cycle = CropCycle(**cycle_dict)
        return self.repo.update_crop_cycle(updated_cycle)

    def complete_harvest(
        self,
        cycle_id: str,
        farmer_id: str,
        harvest_date: str,
        actual_yield_quintals: float,
        notes: Optional[str] = None,
    ) -> Optional[CropCycle]:
        """Mark cycle as completed and record yield metrics while preserving historical record."""
        return self.transition_stage(
            cycle_id=cycle_id,
            farmer_id=farmer_id,
            req=CropStageTransitionRequest(
                new_stage=CropGrowthStage.HARVESTED,
                actual_date=harvest_date,
                yield_quintals=actual_yield_quintals,
                notes=notes or "Harvest completed successfully.",
            ),
        )

    def estimate_growth_stage(
        self,
        crop_name: str,
        sowing_date_str: str,
    ) -> Tuple[CropGrowthStage, int, Optional[str]]:
        """Compute empirical growth stage and estimated harvest date based on elapsed days."""
        try:
            d0 = datetime.strptime(sowing_date_str, "%Y-%m-%d").date()
            days_elapsed = max(0, (datetime.now().date() - d0).days)
        except Exception:
            days_elapsed = 0

        crop_clean = crop_name.lower().strip()
        phenology = CROP_PHENOLOGY_MAP.get(crop_clean)

        if not phenology:
            for key, val in CROP_PHENOLOGY_MAP.items():
                if key in crop_clean or crop_clean in key:
                    phenology = val
                    break

        # Default 120-day phenology if crop not specifically mapped
        if not phenology:
            phenology = [
                (12, CropGrowthStage.SOWING_GERMINATION),
                (30, CropGrowthStage.SEEDLING_EMERGENCE),
                (65, CropGrowthStage.VEGETATIVE_TILLERING),
                (90, CropGrowthStage.FLOWERING_REPRODUCTIVE),
                (115, CropGrowthStage.GRAIN_FILLING_MATURITY),
                (999, CropGrowthStage.HARVEST_READY),
            ]

        estimated_stage = CropGrowthStage.HARVEST_READY
        for max_days, stage in phenology:
            if days_elapsed <= max_days:
                estimated_stage = stage
                break

        # Estimate harvest date as sowing date + typical duration
        typical_duration = phenology[-2][0] + 15 if len(phenology) >= 2 else 120
        try:
            from datetime import timedelta
            est_harvest = (d0 + timedelta(days=typical_duration)).strftime("%Y-%m-%d")
        except Exception:
            est_harvest = None

        return estimated_stage, days_elapsed, est_harvest

    def _enrich_cycle(self, cycle: CropCycle) -> CropCycle:
        """Enrich cycle with real-time days elapsed and estimated stage."""
        estimated_stage, days_elapsed, _ = self.estimate_growth_stage(
            cycle.crop_name,
            cycle.sowing_date,
        )
        cycle.days_since_sowing = days_elapsed
        cycle.estimated_stage = estimated_stage
        return cycle
