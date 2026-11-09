"""Farm Management Service providing multi-farm operations, location binding, and area normalization."""

import uuid
from datetime import datetime
from typing import List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import (
    FarmDatabaseRepository,
    get_farm_repo,
    normalize_area_to_acres,
)
from backend.app.schemas.farm import Farm, FarmCreate, FarmUpdate
from backend.app.services.weather.location_manager import LocationManager

logger = get_logger("FarmService")


class FarmService:
    """Provides multi-farm lifecycle management with geographic resolution and tenant isolation."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        location_manager: Optional[LocationManager] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.location_manager = location_manager or LocationManager()

    def create_farm(self, farmer_id: str, farm_data: FarmCreate) -> Farm:
        """Create a new farm under the authenticated farmer's account."""
        now_str = datetime.now().isoformat()
        farm_id = f"frm_{uuid.uuid4().hex[:12]}"

        # Standardize area in acres
        acres = normalize_area_to_acres(farm_data.total_area, farm_data.area_unit)

        # Resolve GPS coordinates if missing using state/district presets
        lat = farm_data.latitude
        lon = farm_data.longitude
        if lat is None or lon is None:
            resolved_loc = self._resolve_location_coordinates(farm_data.state, farm_data.district)
            if resolved_loc:
                lat, lon = resolved_loc[0], resolved_loc[1]

        # Check existing farms for this farmer; if this is the first farm, auto-activate
        existing_farms = self.repo.get_farms_by_farmer(farmer_id)
        is_active = farm_data.is_active_farm or (len(existing_farms) == 0)

        farm = Farm(
            farm_id=farm_id,
            farmer_id=farmer_id,
            farm_name=farm_data.farm_name.strip(),
            state=farm_data.state.strip(),
            district=farm_data.district.strip(),
            tehsil_taluka=farm_data.tehsil_taluka.strip() if farm_data.tehsil_taluka else None,
            village=farm_data.village.strip() if farm_data.village else None,
            latitude=lat,
            longitude=lon,
            total_area=farm_data.total_area,
            area_unit=farm_data.area_unit,
            total_area_acres=acres,
            ownership_type=farm_data.ownership_type,
            is_active_farm=is_active,
            notes=farm_data.notes,
            created_at=now_str,
            updated_at=now_str,
        )

        created = self.repo.create_farm(farm)
        logger.info(f"Created farm '{farm.farm_name}' ({farm_id}) for farmer {farmer_id}")
        return created

    def get_farm(self, farm_id: str, farmer_id: str) -> Optional[Farm]:
        """Fetch farm verifying strict tenant ownership."""
        return self.repo.get_farm(farm_id=farm_id, farmer_id=farmer_id)

    def list_farms(self, farmer_id: str) -> List[Farm]:
        """List all farms belonging to farmer."""
        return self.repo.get_farms_by_farmer(farmer_id=farmer_id)

    def get_active_farm(self, farmer_id: str) -> Optional[Farm]:
        """Get the active farm for context resolution."""
        return self.repo.get_active_farm(farmer_id=farmer_id)

    def set_active_farm(self, farm_id: str, farmer_id: str) -> Optional[Farm]:
        """Activate a specific farm for this farmer."""
        return self.repo.set_active_farm(farm_id=farm_id, farmer_id=farmer_id)

    def update_farm(self, farm_id: str, farmer_id: str, update_data: FarmUpdate) -> Optional[Farm]:
        """Update farm details verifying tenant ownership."""
        existing = self.repo.get_farm(farm_id=farm_id, farmer_id=farmer_id)
        if not existing:
            return None

        updated_dict = existing.model_dump()
        for field, val in update_data.model_dump(exclude_unset=True).items():
            if val is not None:
                updated_dict[field] = val

        # Recalculate acres if area or unit changed
        updated_dict["total_area_acres"] = normalize_area_to_acres(
            updated_dict["total_area"],
            updated_dict["area_unit"],
        )
        updated_dict["updated_at"] = datetime.now().isoformat()

        updated_farm = Farm(**updated_dict)
        return self.repo.update_farm(updated_farm)

    def delete_farm(self, farm_id: str, farmer_id: str) -> bool:
        """Delete farm verifying tenant ownership."""
        return self.repo.delete_farm(farm_id=farm_id, farmer_id=farmer_id)

    def _resolve_location_coordinates(self, state: str, district: str) -> Optional[tuple]:
        """Look up district coordinates from LocationManager presets."""
        for state_name, dists in self.location_manager.presets.items():
            if state.lower() in state_name.lower() or state_name.lower() in state.lower():
                for dist_name, meta in dists.items():
                    if district.lower() in dist_name.lower() or dist_name.lower() in district.lower():
                        return (meta["lat"], meta["lon"])
        return None

