"""Central profitability calculation engine, financial metrics, and farmer overview."""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import (
    ExpenseCategory,
    FarmerFinancialOverview,
    PaymentStatus,
    ProfitabilityMetrics,
)
from backend.app.schemas.farm import CropCycleStatus
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("profitability_service")


class ProfitabilityService:
    """Calculates transparent, reproducible financial metrics for crop cycles and farms."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def calculate_crop_cycle_profitability(
        self, crop_cycle_id: str, farmer_id: str
    ) -> ProfitabilityMetrics:
        """Calculate complete financial performance for a specific crop cycle."""
        # 1. Fetch Crop Cycle
        cycle = self.repo.get_crop_cycle(crop_cycle_id, farmer_id=farmer_id)
        if not cycle:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Crop cycle '{crop_cycle_id}' not found for farmer.",
            )

        # Fetch Farm & Plot context
        farm = self.repo.get_farm(cycle.farm_id, farmer_id=farmer_id)
        farm_name = farm.farm_name if farm else "Farm"

        area_acres = Decimal("1.0")
        if cycle.plot_id:
            plot = self.repo.get_plot(cycle.plot_id, farmer_id=farmer_id)
            if plot and plot.area_acres > 0:
                area_acres = Decimal(str(plot.area_acres))
        elif farm and farm.total_area_acres > 0:
            area_acres = Decimal(str(farm.total_area_acres))

        # 2. Fetch Actual Incurred Expenses
        expenses = self.repo.list_farm_expenses(
            farmer_id=farmer_id,
            crop_cycle_id=crop_cycle_id,
            is_planned=False,
        )

        cost_breakdown: Dict[str, float] = {cat.value: 0.0 for cat in ExpenseCategory}
        total_cost_dec = Decimal("0.00")

        for exp in expenses:
            amt_dec = UnitConversionService.round_currency(exp.total_amount)
            total_cost_dec += amt_dec
            cat_key = exp.expense_category.value
            cost_breakdown[cat_key] = float(
                UnitConversionService.round_currency(
                    Decimal(str(cost_breakdown.get(cat_key, 0.0))) + amt_dec
                )
            )

        # 3. Fetch Harvest Records
        harvests = self.repo.list_harvest_records(
            farmer_id=farmer_id, crop_cycle_id=crop_cycle_id
        )
        total_prod_qtl = Decimal("0.00")
        for h in harvests:
            qtl = UnitConversionService.convert_weight_to_quintals(h.harvested_quantity, h.quantity_unit)
            total_prod_qtl += qtl

        # 4. Fetch Actual Sales Records
        sales = self.repo.list_farm_sales(
            farmer_id=farmer_id, crop_cycle_id=crop_cycle_id
        )
        total_gross_rev = Decimal("0.00")
        total_net_rev = Decimal("0.00")
        total_deductions_dec = Decimal("0.00")
        total_sold_qtl = Decimal("0.00")

        for s in sales:
            total_gross_rev += UnitConversionService.round_currency(s.gross_amount)
            total_net_rev += UnitConversionService.round_currency(s.net_amount_received)
            total_deductions_dec += UnitConversionService.round_currency(
                s.deductions_charges + s.transport_charges + s.commission_charges
            )
            total_sold_qtl += UnitConversionService.convert_weight_to_quintals(
                s.quantity_sold, s.quantity_unit
            )

        # 5. Profitability & Margins
        net_profit_loss = UnitConversionService.round_currency(total_net_rev - total_cost_dec)
        is_profitable = net_profit_loss > Decimal("0.00")

        # Gross margin = Net Revenue - Direct variable inputs (seeds, fertilizers, protection, labor)
        variable_costs = (
            Decimal(str(cost_breakdown.get(ExpenseCategory.SEEDS_PLANTING.value, 0.0)))
            + Decimal(str(cost_breakdown.get(ExpenseCategory.FERTILIZER_SOIL.value, 0.0)))
            + Decimal(str(cost_breakdown.get(ExpenseCategory.CROP_PROTECTION.value, 0.0)))
            + Decimal(str(cost_breakdown.get(ExpenseCategory.LABOR.value, 0.0)))
            + Decimal(str(cost_breakdown.get(ExpenseCategory.MACHINERY_FUEL.value, 0.0)))
            + Decimal(str(cost_breakdown.get(ExpenseCategory.IRRIGATION_ENERGY.value, 0.0)))
        )
        gross_margin = UnitConversionService.round_currency(total_net_rev - variable_costs)

        # Normalized Metrics
        cost_per_acre: Optional[float] = None
        if area_acres > 0:
            cost_per_acre = float(UnitConversionService.round_currency(total_cost_dec / area_acres))

        cost_per_quintal: Optional[float] = None
        cost_per_kg: Optional[float] = None
        if total_prod_qtl > 0:
            cost_per_quintal = float(UnitConversionService.round_currency(total_cost_dec / total_prod_qtl))
            cost_per_kg = float(
                (Decimal(str(cost_per_quintal)) / Decimal("100.0")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )

        avg_realized_price: Optional[float] = None
        if total_sold_qtl > 0:
            avg_realized_price = float(UnitConversionService.round_currency(total_gross_rev / total_sold_qtl))

        break_even_price: Optional[float] = None
        if total_prod_qtl > 0:
            break_even_price = float(UnitConversionService.round_currency(total_cost_dec / total_prod_qtl))

        break_even_yield: Optional[float] = None
        if avg_realized_price and avg_realized_price > 0:
            break_even_yield = float(
                (total_cost_dec / Decimal(str(avg_realized_price))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            )

        return_on_cost_pct: Optional[float] = None
        if total_cost_dec > 0:
            return_on_cost_pct = float(
                ((net_profit_loss / total_cost_dec) * Decimal("100.0")).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            )

        # 6. Data Completeness & Provisional Flags
        missing_fields: List[str] = []
        is_provisional = False

        if cycle.status != CropCycleStatus.COMPLETED:
            is_provisional = True
            missing_fields.append("crop_cycle_in_progress")

        if not expenses:
            missing_fields.append("no_expenses_recorded")
            is_provisional = True

        if not harvests:
            missing_fields.append("no_harvest_recorded")
            is_provisional = True

        if not sales:
            missing_fields.append("no_sales_recorded")
            is_provisional = True
        elif total_sold_qtl < total_prod_qtl:
            missing_fields.append("unsold_produce_in_storage")

        # 7. Bilingual Explanations
        cost_fmt = UnitConversionService.format_inr(total_cost_dec)
        net_rev_fmt = UnitConversionService.format_inr(total_net_rev)
        net_pl_fmt = UnitConversionService.format_inr(net_profit_loss)
        status_label_en = "PROFIT" if is_profitable else "LOSS"
        status_label_hi = "लाभ" if is_profitable else "हानि"

        now_str = datetime.now(timezone.utc).isoformat()
        expl_en = (
            f"Cultivation Cost: {cost_fmt} ({len(expenses)} entries). "
            f"Net Realized Revenue: {net_rev_fmt} ({float(total_sold_qtl):.1f} qtl sold). "
            f"Net Outcome: {net_pl_fmt} ({status_label_en}). "
            f"{'Provisional: cycle still active.' if is_provisional else 'Finalized result.'}"
        )
        expl_hi = (
            f"खेती की कुल लागत: {cost_fmt} ({len(expenses)} प्रविष्टियाँ). "
            f"प्राप्त शुद्ध आय: {net_rev_fmt} ({float(total_sold_qtl):.1f} क्विंटल बिक्री). "
            f"शुद्ध परिणाम: {net_pl_fmt} ({status_label_hi}). "
            f"{'अस्थायी परिणाम: फसल चक्र अभी सक्रिय है।' if is_provisional else 'अंतिम परिणाम।'}"
        )

        return ProfitabilityMetrics(
            crop_cycle_id=crop_cycle_id,
            crop_name=cycle.crop_name,
            variety_name=cycle.variety_name,
            farm_id=cycle.farm_id,
            farm_name=farm_name,
            cultivated_area_acres=float(area_acres),
            is_provisional=is_provisional,
            missing_data_fields=missing_fields,
            total_actual_cost=float(total_cost_dec),
            cost_breakdown_by_category=cost_breakdown,
            total_gross_revenue=float(total_gross_rev),
            total_net_revenue=float(total_net_rev),
            total_deductions=float(total_deductions_dec),
            gross_margin=float(gross_margin),
            net_profit_or_loss=float(net_profit_loss),
            is_profitable=is_profitable,
            cost_per_acre=cost_per_acre,
            cost_per_quintal=cost_per_quintal,
            cost_per_kg=cost_per_kg,
            total_production_quintals=float(total_prod_qtl),
            total_sold_quintals=float(total_sold_qtl),
            average_realized_price_per_quintal=avg_realized_price,
            break_even_selling_price_per_quintal=break_even_price,
            break_even_production_quintals=break_even_yield,
            return_on_cultivation_cost_pct=return_on_cost_pct,
            calculation_timestamp=now_str,
            explanation_en=expl_en,
            explanation_hi=expl_hi,
        )

    def get_farmer_financial_overview(self, farmer_id: str) -> FarmerFinancialOverview:
        """Aggregate financial health across all farms and crop cycles for a farmer."""
        all_expenses = self.repo.list_farm_expenses(farmer_id=farmer_id)
        all_sales = self.repo.list_farm_sales(farmer_id=farmer_id)
        all_cycles = self.repo.get_all_crop_cycles_by_farmer(farmer_id=farmer_id)

        tot_exp = Decimal("0.00")
        tot_paid = Decimal("0.00")
        tot_unpaid = Decimal("0.00")

        for exp in all_expenses:
            if not exp.is_planned:
                amt = UnitConversionService.round_currency(exp.total_amount)
                tot_exp += amt
                if exp.payment_status == PaymentStatus.PAID:
                    tot_paid += amt
                else:
                    tot_unpaid += amt

        tot_rev = Decimal("0.00")
        for sale in all_sales:
            tot_rev += UnitConversionService.round_currency(sale.net_amount_received)

        net_bal = UnitConversionService.round_currency(tot_rev - tot_exp)

        active_costs = Decimal("0.00")
        completed_profit = Decimal("0.00")
        provisional_count = 0
        completed_count = 0

        for cycle in all_cycles:
            try:
                metrics = self.calculate_crop_cycle_profitability(cycle.cycle_id, farmer_id)
                if metrics.is_provisional:
                    provisional_count += 1
                    active_costs += Decimal(str(metrics.total_actual_cost))
                else:
                    completed_count += 1
                    completed_profit += Decimal(str(metrics.net_profit_or_loss))
            except Exception as e:
                logger.warning(f"Could not compute cycle metrics for {cycle.cycle_id}: {e}")

        recent_expenses = all_expenses[:10]
        recent_sales = all_sales[:10]

        return FarmerFinancialOverview(
            farmer_id=farmer_id,
            total_cultivation_expenses=float(tot_exp),
            total_paid_expenses=float(tot_paid),
            total_outstanding_expenses=float(tot_unpaid),
            total_realized_revenue=float(tot_rev),
            net_balance=float(net_bal),
            active_crop_cycles_cost=float(active_costs),
            completed_crop_cycles_profit=float(completed_profit),
            provisional_cycles_count=provisional_count,
            completed_cycles_count=completed_count,
            recent_expenses=recent_expenses,
            recent_sales=recent_sales,
        )


_profitability_service_instance: Optional[ProfitabilityService] = None


def get_profitability_service(repo: Optional[FarmDatabaseRepository] = None) -> ProfitabilityService:
    """Singleton getter for ProfitabilityService."""
    global _profitability_service_instance
    if _profitability_service_instance is None or repo is not None:
        _profitability_service_instance = ProfitabilityService(repo=repo)
    return _profitability_service_instance

