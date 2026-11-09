"""Agricultural expense management service with Decimal precision and auditable corrections."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import (
    ExpenseCategory,
    ExpenseCorrectionCreate,
    ExpenseCorrectionRecord,
    FarmExpenseCreate,
    FarmExpenseRecord,
    FarmExpenseUpdate,
    PaymentStatus,
)
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("expense_service")


class ExpenseService:
    """Manages agricultural expenditure, labor tracking, receipt linking, and corrections."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def record_expense(
        self, farmer_id: str, payload: FarmExpenseCreate, user_id: str
    ) -> FarmExpenseRecord:
        """Create and persist an expense entry after validating tenant ownership and arithmetic."""
        # 1. Verify Farm Ownership
        farm = self.repo.get_farm(payload.farm_id, farmer_id=farmer_id)
        if not farm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farm '{payload.farm_id}' does not exist or does not belong to farmer.",
            )

        # 2. Arithmetic validation if both quantity and unit price provided
        if payload.quantity is not None and payload.unit_price is not None:
            expected = UnitConversionService.round_currency(
                Decimal(str(payload.quantity)) * Decimal(str(payload.unit_price))
            )
            supplied = UnitConversionService.round_currency(payload.total_amount)
            # Allow at most 0.05 rounding tolerance
            if abs(expected - supplied) > Decimal("0.05"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Total amount mismatch: quantity ({payload.quantity}) * unit_price ({payload.unit_price}) "
                        f"= {expected}, but supplied total_amount is {supplied}."
                    ),
                )

        now_str = datetime.now(timezone.utc).isoformat()
        expense_id = f"exp_{uuid.uuid4().hex[:12]}"

        record = FarmExpenseRecord(
            expense_id=expense_id,
            farmer_id=farmer_id,
            farm_id=payload.farm_id,
            plot_id=payload.plot_id,
            crop_cycle_id=payload.crop_cycle_id,
            expense_category=payload.expense_category,
            subcategory=payload.subcategory,
            expense_description=payload.expense_description,
            input_name=payload.input_name,
            quantity=payload.quantity,
            quantity_unit=payload.quantity_unit,
            unit_price=payload.unit_price,
            total_amount=float(UnitConversionService.round_currency(payload.total_amount)),
            currency=payload.currency,
            purchase_date=payload.purchase_date,
            payment_date=payload.payment_date,
            payment_status=payload.payment_status,
            is_planned=payload.is_planned,
            supplier_vendor=payload.supplier_vendor,
            invoice_attachment_id=payload.invoice_attachment_id,
            notes=payload.notes,
            entry_source=payload.entry_source,
            version_number=1,
            created_at=now_str,
            updated_at=now_str,
        )

        persisted = self.repo.create_farm_expense(record)
        self.repo.record_financial_audit(
            entity_type="farm_expense",
            entity_id=expense_id,
            farmer_id=farmer_id,
            action="CREATE",
            changed_fields=record.model_dump(),
            performed_by=user_id,
        )
        logger.info(f"Recorded expense {expense_id} for farmer {farmer_id}: ₹{record.total_amount}")
        return persisted

    def get_expense(self, expense_id: str, farmer_id: str) -> FarmExpenseRecord:
        """Retrieve single expense verifying tenant ownership."""
        record = self.repo.get_farm_expense(expense_id, farmer_id=farmer_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Expense '{expense_id}' not found.",
            )
        return record

    def list_expenses(
        self,
        farmer_id: str,
        farm_id: Optional[str] = None,
        plot_id: Optional[str] = None,
        crop_cycle_id: Optional[str] = None,
        category: Optional[str] = None,
        payment_status: Optional[str] = None,
        is_planned: Optional[bool] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FarmExpenseRecord]:
        """List expenses matching filters."""
        return self.repo.list_farm_expenses(
            farmer_id=farmer_id,
            farm_id=farm_id,
            plot_id=plot_id,
            crop_cycle_id=crop_cycle_id,
            category=category,
            payment_status=payment_status,
            is_planned=is_planned,
            from_date=from_date,
            to_date=to_date,
        )

    def update_expense(
        self,
        expense_id: str,
        farmer_id: str,
        payload: FarmExpenseUpdate,
        user_id: str,
    ) -> FarmExpenseRecord:
        """Update mutable fields of an existing expense."""
        existing = self.get_expense(expense_id, farmer_id)
        updates = payload.model_dump(exclude_unset=True)

        if not updates:
            return existing

        # If quantity or unit_price or total_amount updated, re-validate
        q = updates.get("quantity", existing.quantity)
        p = updates.get("unit_price", existing.unit_price)
        tot = updates.get("total_amount", existing.total_amount)

        if q is not None and p is not None and tot is not None:
            expected = UnitConversionService.round_currency(Decimal(str(q)) * Decimal(str(p)))
            supplied = UnitConversionService.round_currency(tot)
            if abs(expected - supplied) > Decimal("0.05"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Total amount mismatch: {q} * {p} = {expected}, but supplied {supplied}.",
                )

        success = self.repo.update_farm_expense(expense_id, farmer_id, updates)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update expense record.",
            )

        updated = self.get_expense(expense_id, farmer_id)
        self.repo.record_financial_audit(
            entity_type="farm_expense",
            entity_id=expense_id,
            farmer_id=farmer_id,
            action="UPDATE",
            changed_fields=updates,
            performed_by=user_id,
        )
        return updated

    def correct_expense(
        self,
        expense_id: str,
        farmer_id: str,
        payload: ExpenseCorrectionCreate,
        user_id: str,
    ) -> ExpenseCorrectionRecord:
        """Apply an append-only correction to an expense amount without erasing history."""
        existing = self.get_expense(expense_id, farmer_id)
        correction_id = f"cor_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()

        corr = ExpenseCorrectionRecord(
            correction_id=correction_id,
            original_expense_id=expense_id,
            farmer_id=farmer_id,
            reason=payload.reason,
            previous_amount=existing.total_amount,
            new_amount=float(UnitConversionService.round_currency(payload.new_amount)),
            corrected_by=user_id,
            created_at=now_str,
        )

        persisted = self.repo.create_expense_correction(corr)
        self.repo.record_financial_audit(
            entity_type="farm_expense",
            entity_id=expense_id,
            farmer_id=farmer_id,
            action="CORRECTION",
            changed_fields={
                "reason": payload.reason,
                "previous_amount": existing.total_amount,
                "new_amount": corr.new_amount,
            },
            performed_by=user_id,
        )
        logger.info(
            f"Applied correction {correction_id} to expense {expense_id}: "
            f"₹{existing.total_amount} -> ₹{corr.new_amount} (Reason: {payload.reason})"
        )
        return persisted

    def list_corrections(
        self, expense_id: str, farmer_id: str
    ) -> List[ExpenseCorrectionRecord]:
        """Fetch audit trail of corrections for an expense."""
        # Ensure expense exists and belongs to farmer
        self.get_expense(expense_id, farmer_id)
        return self.repo.list_expense_corrections(expense_id, farmer_id)

    def delete_expense(self, expense_id: str, farmer_id: str, user_id: str) -> bool:
        """Permanently delete an expense record."""
        existing = self.get_expense(expense_id, farmer_id)
        success = self.repo.delete_farm_expense(expense_id, farmer_id)
        if success:
            self.repo.record_financial_audit(
                entity_type="farm_expense",
                entity_id=expense_id,
                farmer_id=farmer_id,
                action="DELETE",
                changed_fields=existing.model_dump(),
                performed_by=user_id,
            )
        return success


_expense_service_instance: Optional[ExpenseService] = None


def get_expense_service(repo: Optional[FarmDatabaseRepository] = None) -> ExpenseService:
    """Singleton getter for ExpenseService."""
    global _expense_service_instance
    if _expense_service_instance is None or repo is not None:
        _expense_service_instance = ExpenseService(repo=repo)
    return _expense_service_instance

