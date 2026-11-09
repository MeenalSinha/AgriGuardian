"""Crop cycle budgeting, planned-versus-actual analysis, and threshold alerting."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import (
    BudgetItemCreate,
    BudgetItemRecord,
    BudgetVarianceItem,
    BudgetVarianceReport,
    CropCycleBudgetCreate,
    CropCycleBudgetRecord,
    CropCycleBudgetUpdate,
    ExpenseCategory,
)
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("budget_service")


class BudgetService:
    """Manages agricultural budgeting, category allocations, variance analysis, and proactive alerts."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def create_budget(
        self, farmer_id: str, payload: CropCycleBudgetCreate, user_id: str
    ) -> CropCycleBudgetRecord:
        """Create and persist an agricultural crop cycle budget."""
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

        now_str = datetime.now(timezone.utc).isoformat()
        budget_id = f"bdg_{uuid.uuid4().hex[:12]}"

        budget_rec = CropCycleBudgetRecord(
            budget_id=budget_id,
            farmer_id=farmer_id,
            farm_id=payload.farm_id,
            plot_id=payload.plot_id,
            crop_cycle_id=payload.crop_cycle_id,
            budget_name=payload.budget_name,
            season=payload.season,
            total_budget_amount=float(UnitConversionService.round_currency(payload.total_budget_amount)),
            currency=payload.currency,
            version_number=1,
            status="active",
            created_at=now_str,
            updated_at=now_str,
            items=[],
        )

        item_records: List[BudgetItemRecord] = []
        for it in payload.items:
            item_records.append(
                BudgetItemRecord(
                    item_id=f"bit_{uuid.uuid4().hex[:12]}",
                    budget_id=budget_id,
                    expense_category=it.expense_category,
                    planned_amount=float(UnitConversionService.round_currency(it.planned_amount)),
                    notes=it.notes,
                    created_at=now_str,
                    updated_at=now_str,
                )
            )

        persisted = self.repo.create_crop_cycle_budget(budget_rec, item_records)
        self.repo.record_financial_audit(
            entity_type="crop_cycle_budget",
            entity_id=budget_id,
            farmer_id=farmer_id,
            action="CREATE",
            changed_fields=budget_rec.model_dump(),
            performed_by=user_id,
        )
        logger.info(f"Created budget {budget_id} for cycle {payload.crop_cycle_id}: ₹{budget_rec.total_budget_amount}")
        return persisted

    def get_budget(self, budget_id: str, farmer_id: str) -> CropCycleBudgetRecord:
        """Fetch budget by ID with category breakdown."""
        res = self.repo.get_crop_cycle_budget(budget_id, farmer_id=farmer_id)
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget '{budget_id}' not found.",
            )
        budget, items = res
        budget.items = items
        return budget

    def get_budget_by_crop_cycle(
        self, crop_cycle_id: str, farmer_id: str
    ) -> Optional[CropCycleBudgetRecord]:
        """Fetch active budget for crop cycle."""
        res = self.repo.get_budget_by_crop_cycle(crop_cycle_id, farmer_id=farmer_id)
        if not res:
            return None
        budget, items = res
        budget.items = items
        return budget

    def list_budgets(
        self, farmer_id: str, farm_id: Optional[str] = None
    ) -> List[CropCycleBudgetRecord]:
        """List all budgets for farmer."""
        records = self.repo.list_crop_cycle_budgets(farmer_id, farm_id=farm_id)
        result = []
        for b, items in records:
            b.items = items
            result.append(b)
        return result

    def get_variance_report(
        self, crop_cycle_id: str, farmer_id: str
    ) -> BudgetVarianceReport:
        """Calculate Planned vs Actual expenditure variance across all categories."""
        budget = self.get_budget_by_crop_cycle(crop_cycle_id, farmer_id)
        if not budget:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active budget found for crop cycle '{crop_cycle_id}'.",
            )

        # Actual expenses recorded for this cycle
        actual_expenses = self.repo.list_farm_expenses(
            farmer_id=farmer_id, crop_cycle_id=crop_cycle_id, is_planned=False
        )

        actual_by_cat: Dict[str, Decimal] = {cat.value: Decimal("0.00") for cat in ExpenseCategory}
        total_actual_dec = Decimal("0.00")

        for exp in actual_expenses:
            amt = UnitConversionService.round_currency(exp.total_amount)
            cat_key = exp.expense_category.value
            actual_by_cat[cat_key] += amt
            total_actual_dec += amt

        planned_by_cat: Dict[str, Decimal] = {cat.value: Decimal("0.00") for cat in ExpenseCategory}
        for it in budget.items:
            cat_key = it.expense_category.value
            planned_by_cat[cat_key] += UnitConversionService.round_currency(it.planned_amount)

        variance_items: List[BudgetVarianceItem] = []
        for cat in ExpenseCategory:
            cat_key = cat.value
            p_amt = planned_by_cat.get(cat_key, Decimal("0.00"))
            a_amt = actual_by_cat.get(cat_key, Decimal("0.00"))
            var_amt = a_amt - p_amt
            var_pct = Decimal("0.0")
            if p_amt > Decimal("0.00"):
                var_pct = ((var_amt / p_amt) * Decimal("100.0")).quantize(
                    Decimal("0.1")
                )

            variance_items.append(
                BudgetVarianceItem(
                    expense_category=cat,
                    planned_amount=float(p_amt),
                    actual_amount=float(a_amt),
                    variance_amount=float(var_amt),
                    variance_percentage=float(var_pct),
                    is_over_budget=a_amt > p_amt,
                )
            )

        total_planned_dec = UnitConversionService.round_currency(budget.total_budget_amount)
        rem_budget = total_planned_dec - total_actual_dec
        if rem_budget < Decimal("0.00"):
            rem_budget = Decimal("0.00")

        over_budget = total_actual_dec - total_planned_dec
        if over_budget < Decimal("0.00"):
            over_budget = Decimal("0.00")

        tot_var_pct = Decimal("0.0")
        if total_planned_dec > Decimal("0.00"):
            tot_var_pct = (
                ((total_actual_dec - total_planned_dec) / total_planned_dec) * Decimal("100.0")
            ).quantize(Decimal("0.1"))

        is_exceeded = total_actual_dec > total_planned_dec

        # Check for threshold alerts (Phase 9 integration)
        self._check_budget_threshold_alerts(
            farmer_id=farmer_id,
            budget=budget,
            total_planned=total_planned_dec,
            total_actual=total_actual_dec,
            over_budget=over_budget,
        )

        return BudgetVarianceReport(
            budget_id=budget.budget_id,
            crop_cycle_id=crop_cycle_id,
            total_planned=float(total_planned_dec),
            total_actual=float(total_actual_dec),
            remaining_budget=float(rem_budget),
            amount_over_budget=float(over_budget),
            total_variance_pct=float(tot_var_pct),
            is_exceeded=is_exceeded,
            categories=variance_items,
        )

    def _check_budget_threshold_alerts(
        self,
        farmer_id: str,
        budget: CropCycleBudgetRecord,
        total_planned: Decimal,
        total_actual: Decimal,
        over_budget: Decimal,
    ) -> None:
        """Trigger proactive notification event if budget approaching limit or exceeded."""
        try:
            from backend.app.schemas.notification import NotificationCategory, NotificationEvent, NotificationPriority
            from backend.app.services.notifications.notification_event_bus import get_notification_event_bus

            bus = get_notification_event_bus(self.repo)

            if total_actual > total_planned and total_planned > Decimal("0.00"):
                event = NotificationEvent(
                    event_id=f"evt_bdg_exc_{budget.budget_id}_{datetime.now().strftime('%Y%m%d')}",
                    farmer_id=farmer_id,
                    category=NotificationCategory.AGRONOMIC_REMINDER,
                    priority=NotificationPriority.HIGH,
                    event_type="budget_exceeded",
                    dedup_key=f"bdg_exc_{budget.budget_id}",
                    payload={
                        "budget_name": budget.budget_name,
                        "crop_cycle_id": budget.crop_cycle_id,
                        "planned_amount": float(total_planned),
                        "actual_amount": float(total_actual),
                        "over_budget": float(over_budget),
                    },
                    title_en=f"⚠️ Budget Exceeded: {budget.budget_name}",
                    title_hi=f"⚠️ बजट सीमा पार: {budget.budget_name}",
                    body_en=f"Your expenses of ₹{float(total_actual):,.2f} have exceeded your planned budget of ₹{float(total_planned):,.2f} by ₹{float(over_budget):,.2f}.",
                    body_hi=f"आपका खर्च ₹{float(total_actual):,.2f} निर्धारित बजट ₹{float(total_planned):,.2f} से ₹{float(over_budget):,.2f} अधिक हो गया है।",
                )
                bus.publish(event)
        except Exception as e:
            logger.debug(f"Budget alert dispatch skipped: {e}")

    # Backward compatibility alias
    calculate_budget_variance = get_variance_report


_budget_service_instance: Optional[BudgetService] = None


def get_budget_service(repo: Optional[FarmDatabaseRepository] = None) -> BudgetService:
    """Singleton getter for BudgetService."""
    global _budget_service_instance
    if _budget_service_instance is None or repo is not None:
        _budget_service_instance = BudgetService(repo=repo)
    return _budget_service_instance

