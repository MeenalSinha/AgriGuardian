"""Agricultural harvest recording, yield calculations, and production reconciliation."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import FarmHarvestCreate, FarmHarvestRecord, FarmHarvestUpdate
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("harvest_service")


class HarvestService:
    """Manages multi-event crop harvesting, yield calculations, and storage/sale reconciliation."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def record_harvest(
        self, farmer_id: str, payload: FarmHarvestCreate, user_id: str
    ) -> FarmHarvestRecord:
        """Create and persist a crop harvest record after validating consistency."""
        # 1. Verify Farm & Crop Cycle
        farm = self.repo.get_farm(payload.farm_id, farmer_id=farmer_id)
        if not farm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm '{payload.farm_id}' not found.",
            )

        cycle = self.repo.get_crop_cycle(payload.crop_cycle_id, farmer_id=farmer_id)
        if not cycle:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Crop cycle '{payload.crop_cycle_id}' not found.",
            )

        # 2. Production reconciliation check
        allocated = (
            payload.damaged_quantity
            + payload.rejected_quantity
            + payload.stored_quantity
            + payload.sold_quantity
            + payload.retained_quantity
        )
        if allocated > (payload.harvested_quantity * 1.01):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Allocated produce ({allocated} {payload.quantity_unit}) exceeds total harvested "
                    f"quantity ({payload.harvested_quantity} {payload.quantity_unit})."
                ),
            )

        now_str = datetime.now(timezone.utc).isoformat()
        harvest_id = f"hrv_{uuid.uuid4().hex[:12]}"

        record = FarmHarvestRecord(
            harvest_id=harvest_id,
            farmer_id=farmer_id,
            farm_id=payload.farm_id,
            plot_id=payload.plot_id,
            crop_cycle_id=payload.crop_cycle_id,
            harvest_date=payload.harvest_date,
            crop_name=payload.crop_name,
            variety_name=payload.variety_name or cycle.variety_name,
            harvested_quantity=float(UnitConversionService.to_decimal(payload.harvested_quantity)),
            quantity_unit=payload.quantity_unit.lower(),
            quality_grade=payload.quality_grade,
            damaged_quantity=float(UnitConversionService.to_decimal(payload.damaged_quantity)),
            rejected_quantity=float(UnitConversionService.to_decimal(payload.rejected_quantity)),
            stored_quantity=float(UnitConversionService.to_decimal(payload.stored_quantity)),
            sold_quantity=float(UnitConversionService.to_decimal(payload.sold_quantity)),
            retained_quantity=float(UnitConversionService.to_decimal(payload.retained_quantity)),
            moisture_percentage=payload.moisture_percentage,
            notes=payload.notes,
            attachment_id=payload.attachment_id,
            version_number=1,
            created_at=now_str,
            updated_at=now_str,
        )

        persisted = self.repo.create_harvest_record(record)
        self.repo.record_financial_audit(
            entity_type="farm_harvest",
            entity_id=harvest_id,
            farmer_id=farmer_id,
            action="CREATE",
            changed_fields=record.model_dump(),
            performed_by=user_id,
        )
        logger.info(f"Recorded harvest {harvest_id} for cycle {payload.crop_cycle_id}: {record.harvested_quantity} {record.quantity_unit}")
        return persisted

    def get_harvest(self, harvest_id: str, farmer_id: str) -> FarmHarvestRecord:
        """Fetch harvest record verifying ownership."""
        record = self.repo.get_harvest_record(harvest_id, farmer_id=farmer_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Harvest record '{harvest_id}' not found.",
            )
        return record

    def list_harvests(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        crop_cycle_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FarmHarvestRecord]:
        """List harvest events for farmer."""
        return self.repo.list_harvest_records(
            farmer_id=farmer_id,
            farm_id=farm_id,
            crop_cycle_id=crop_cycle_id,
            from_date=from_date,
            to_date=to_date,
        )

    def update_harvest(
        self,
        harvest_id: str,
        farmer_id: str,
        payload: FarmHarvestUpdate,
        user_id: str,
    ) -> FarmHarvestRecord:
        """Update mutable fields of a harvest record."""
        existing = self.get_harvest(harvest_id, farmer_id)
        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            return existing

        success = self.repo.update_harvest_record(harvest_id, farmer_id, updates)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update harvest record.",
            )

        updated = self.get_harvest(harvest_id, farmer_id)
        self.repo.record_financial_audit(
            entity_type="farm_harvest",
            entity_id=harvest_id,
            farmer_id=farmer_id,
            action="UPDATE",
            changed_fields=updates,
            performed_by=user_id,
        )
        return updated

    def delete_harvest(self, harvest_id: str, farmer_id: str, user_id: str) -> bool:
        """Delete harvest record."""
        existing = self.get_harvest(harvest_id, farmer_id)
        success = self.repo.delete_harvest_record(harvest_id, farmer_id)
        if success:
            self.repo.record_financial_audit(
                entity_type="farm_harvest",
                entity_id=harvest_id,
                farmer_id=farmer_id,
                action="DELETE",
                changed_fields=existing.model_dump(),
                performed_by=user_id,
            )
        return success


_harvest_service_instance: Optional[HarvestService] = None


def get_harvest_service(repo: Optional[FarmDatabaseRepository] = None) -> HarvestService:
    """Singleton getter for HarvestService."""
    global _harvest_service_instance
    if _harvest_service_instance is None or repo is not None:
        _harvest_service_instance = HarvestService(repo=repo)
    return _harvest_service_instance

