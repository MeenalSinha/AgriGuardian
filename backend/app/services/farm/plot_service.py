"""Plot and field management service with area normalization and farm binding."""

import uuid
from datetime import datetime
from typing import List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import (
    FarmDatabaseRepository,
    get_farm_repo,
    normalize_area_to_acres,
)
from backend.app.schemas.farm import Plot, PlotCreate, PlotUpdate

logger = get_logger("PlotService")


class PlotService:
    """Manages plots/fields within farms with strict tenant isolation and area normalization."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def create_plot(self, farm_id: str, farmer_id: str, plot_data: PlotCreate) -> Plot:
        """Create a new plot inside a farm verifying farm ownership."""
        farm = self.repo.get_farm(farm_id=farm_id, farmer_id=farmer_id)
        if not farm:
            raise ValueError(f"Farm '{farm_id}' not found or does not belong to farmer '{farmer_id}'.")

        now_str = datetime.now().isoformat()
        plot_id = f"plt_{uuid.uuid4().hex[:12]}"
        acres = normalize_area_to_acres(plot_data.area, plot_data.area_unit)

        plot = Plot(
            plot_id=plot_id,
            farm_id=farm_id,
            farmer_id=farmer_id,
            plot_name=plot_data.plot_name.strip(),
            area=plot_data.area,
            area_unit=plot_data.area_unit,
            area_acres=acres,
            soil_type=plot_data.soil_type,
            irrigation_source=plot_data.irrigation_source.strip(),
            drainage_type=plot_data.drainage_type.strip(),
            slope=plot_data.slope.strip(),
            notes=plot_data.notes,
            created_at=now_str,
            updated_at=now_str,
        )

        created = self.repo.create_plot(plot)
        logger.info(f"Created plot '{plot.plot_name}' ({plot_id}) on farm {farm_id}")
        return created

    def get_plot(self, plot_id: str, farmer_id: str) -> Optional[Plot]:
        """Fetch plot verifying tenant ownership."""
        return self.repo.get_plot(plot_id=plot_id, farmer_id=farmer_id)

    def list_plots_by_farm(self, farm_id: str, farmer_id: str) -> List[Plot]:
        """List plots within a farm verifying farm ownership."""
        farm = self.repo.get_farm(farm_id=farm_id, farmer_id=farmer_id)
        if not farm:
            return []
        return self.repo.get_plots_by_farm(farm_id=farm_id, farmer_id=farmer_id)

    def list_plots_by_farmer(self, farmer_id: str) -> List[Plot]:
        """List all plots across all farms belonging to this farmer."""
        return self.repo.get_plots_by_farmer(farmer_id=farmer_id)

    def update_plot(self, plot_id: str, farmer_id: str, update_data: PlotUpdate) -> Optional[Plot]:
        """Update plot verifying tenant ownership."""
        existing = self.repo.get_plot(plot_id=plot_id, farmer_id=farmer_id)
        if not existing:
            return None

        updated_dict = existing.model_dump()
        for field, val in update_data.model_dump(exclude_unset=True).items():
            if val is not None:
                updated_dict[field] = val

        updated_dict["area_acres"] = normalize_area_to_acres(
            updated_dict["area"],
            updated_dict["area_unit"],
        )
        updated_dict["updated_at"] = datetime.now().isoformat()

        updated_plot = Plot(**updated_dict)
        return self.repo.update_plot(updated_plot)

    def delete_plot(self, plot_id: str, farmer_id: str) -> bool:
        """Delete plot verifying tenant ownership."""
        return self.repo.delete_plot(plot_id=plot_id, farmer_id=farmer_id)
