"""Farm sales, revenue realization, and mandi transaction service."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import FarmSaleCreate, FarmSaleRecord, FarmSaleUpdate, PaymentStatus
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("sales_service")


class SalesService:
    """Manages actual agricultural produce sales, realized revenue calculations, and market reconciliation."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def record_sale(
        self, farmer_id: str, payload: FarmSaleCreate, user_id: str
    ) -> FarmSaleRecord:
        """Create and persist a produce sale record with strict revenue arithmetic."""
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

        # 2. Strict Arithmetic: Gross = Quantity * Realized Price
        qty_dec = UnitConversionService.to_decimal(payload.quantity_sold)
        price_dec = UnitConversionService.to_decimal(payload.realized_price_per_unit)
        calc_gross = UnitConversionService.round_currency(qty_dec * price_dec)

        if payload.gross_amount is not None:
            supplied_gross = UnitConversionService.round_currency(payload.gross_amount)
            if abs(calc_gross - supplied_gross) > Decimal("0.05"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Gross amount mismatch: quantity ({payload.quantity_sold}) * price ({payload.realized_price_per_unit}) "
                        f"= {calc_gross}, but supplied gross_amount is {supplied_gross}."
                    ),
                )
        gross_final = float(calc_gross)

        # 3. Net = Gross - Deductions - Transport - Commission
        total_deductions = (
            UnitConversionService.to_decimal(payload.deductions_charges)
            + UnitConversionService.to_decimal(payload.transport_charges)
            + UnitConversionService.to_decimal(payload.commission_charges)
        )
        calc_net = UnitConversionService.round_currency(Decimal(str(gross_final)) - total_deductions)
        if calc_net < Decimal("0.00"):
            calc_net = Decimal("0.00")

        if payload.net_amount_received is not None:
            supplied_net = UnitConversionService.round_currency(payload.net_amount_received)
            if abs(calc_net - supplied_net) > Decimal("0.05"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Net amount mismatch: gross ({gross_final}) - deductions ({total_deductions}) "
                        f"= {calc_net}, but supplied net_amount_received is {supplied_net}."
                    ),
                )
        net_final = float(calc_net)

        now_str = datetime.now(timezone.utc).isoformat()
        sale_id = f"sal_{uuid.uuid4().hex[:12]}"

        record = FarmSaleRecord(
            sale_id=sale_id,
            farmer_id=farmer_id,
            farm_id=payload.farm_id,
            plot_id=payload.plot_id,
            crop_cycle_id=payload.crop_cycle_id,
            harvest_id=payload.harvest_id,
            crop_name=payload.crop_name,
            quantity_sold=float(qty_dec),
            quantity_unit=payload.quantity_unit.lower(),
            realized_price_per_unit=float(price_dec),
            gross_amount=gross_final,
            deductions_charges=float(payload.deductions_charges),
            net_amount_received=net_final,
            currency=payload.currency,
            sale_date=payload.sale_date,
            buyer_name=payload.buyer_name,
            mandi_id=payload.mandi_id,
            quality_grade=payload.quality_grade,
            payment_status=payload.payment_status,
            payment_received_date=payload.payment_received_date,
            transport_charges=float(payload.transport_charges),
            commission_charges=float(payload.commission_charges),
            attachment_id=payload.attachment_id,
            notes=payload.notes,
            version_number=1,
            created_at=now_str,
            updated_at=now_str,
        )

        persisted = self.repo.create_farm_sale(record)
        self.repo.record_financial_audit(
            entity_type="farm_sale",
            entity_id=sale_id,
            farmer_id=farmer_id,
            action="CREATE",
            changed_fields=record.model_dump(),
            performed_by=user_id,
        )
        logger.info(
            f"Recorded sale {sale_id} for crop {payload.crop_name}: "
            f"{record.quantity_sold} {record.quantity_unit} @ ₹{record.realized_price_per_unit} = Net ₹{record.net_amount_received}"
        )
        return persisted

    def get_sale(self, sale_id: str, farmer_id: str) -> FarmSaleRecord:
        """Fetch sale record verifying ownership."""
        record = self.repo.get_farm_sale(sale_id, farmer_id=farmer_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sale record '{sale_id}' not found.",
            )
        return record

    def list_sales(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        crop_cycle_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FarmSaleRecord]:
        """List sales records matching filters."""
        return self.repo.list_farm_sales(
            farmer_id=farmer_id,
            farm_id=farm_id,
            crop_cycle_id=crop_cycle_id,
            from_date=from_date,
            to_date=to_date,
        )

    def update_sale(
        self,
        sale_id: str,
        farmer_id: str,
        payload: FarmSaleUpdate,
        user_id: str,
    ) -> FarmSaleRecord:
        """Update mutable fields of a sale record."""
        existing = self.get_sale(sale_id, farmer_id)
        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            return existing

        # Recalculate gross and net if price/quantity/deductions changed
        qty = updates.get("quantity_sold", existing.quantity_sold)
        price = updates.get("realized_price_per_unit", existing.realized_price_per_unit)
        ded = updates.get("deductions_charges", existing.deductions_charges)
        trans = updates.get("transport_charges", existing.transport_charges)
        comm = updates.get("commission_charges", existing.commission_charges)

        calc_gross = UnitConversionService.round_currency(Decimal(str(qty)) * Decimal(str(price)))
        tot_ded = UnitConversionService.to_decimal(ded) + UnitConversionService.to_decimal(trans) + UnitConversionService.to_decimal(comm)
        calc_net = UnitConversionService.round_currency(calc_gross - tot_ded)
        if calc_net < Decimal("0.00"):
            calc_net = Decimal("0.00")

        updates["gross_amount"] = float(calc_gross)
        updates["net_amount_received"] = float(calc_net)

        success = self.repo.update_farm_sale(sale_id, farmer_id, updates)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update sale record.",
            )

        updated = self.get_sale(sale_id, farmer_id)
        self.repo.record_financial_audit(
            entity_type="farm_sale",
            entity_id=sale_id,
            farmer_id=farmer_id,
            action="UPDATE",
            changed_fields=updates,
            performed_by=user_id,
        )
        return updated

    def delete_sale(self, sale_id: str, farmer_id: str, user_id: str) -> bool:
        """Delete sale record."""
        existing = self.get_sale(sale_id, farmer_id)
        success = self.repo.delete_farm_sale(sale_id, farmer_id)
        if success:
            self.repo.record_financial_audit(
                entity_type="farm_sale",
                entity_id=sale_id,
                farmer_id=farmer_id,
                action="DELETE",
                changed_fields=existing.model_dump(),
                performed_by=user_id,
            )
        return success


_sales_service_instance: Optional[SalesService] = None


def get_sales_service(repo: Optional[FarmDatabaseRepository] = None) -> SalesService:
    """Singleton getter for SalesService."""
    global _sales_service_instance
    if _sales_service_instance is None or repo is not None:
        _sales_service_instance = SalesService(repo=repo)
    return _sales_service_instance

