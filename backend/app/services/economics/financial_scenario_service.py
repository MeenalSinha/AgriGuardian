"""Financial scenario simulator, sensitivity analysis, and multi-crop economic comparison."""

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import (
    CropComparisonItem,
    CropComparisonReport,
    CropComparisonRequest,
    FinancialScenarioRecord,
    ScenarioResult,
    ScenarioSimulateRequest,
    ScenarioType,
)
from backend.app.services.economics.unit_conversion_service import UnitConversionService

logger = get_logger("financial_scenario_service")


class FinancialScenarioService:
    """Provides transparent, deterministic scenario modeling and comparative crop economics."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def simulate_scenario(self, req: ScenarioSimulateRequest) -> ScenarioResult:
        """Execute deterministic financial projection with sensitivity analysis."""
        area_dec = UnitConversionService.to_decimal(req.planned_area_acres)
        cost_per_acre_dec = UnitConversionService.to_decimal(req.expected_cost_per_acre)
        yield_per_acre_dec = UnitConversionService.to_decimal(req.expected_yield_quintals_per_acre)
        price_dec = UnitConversionService.to_decimal(req.expected_price_per_quintal)
        loss_pct_dec = UnitConversionService.to_decimal(req.expected_harvest_loss_pct) / Decimal("100.0")
        trans_cost_dec = UnitConversionService.to_decimal(req.transport_cost_per_quintal)

        # Totals
        total_cost = UnitConversionService.round_currency(area_dec * cost_per_acre_dec)
        gross_prod = (area_dec * yield_per_acre_dec).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        saleable_prod = (gross_prod * (Decimal("1.0") - loss_pct_dec)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        expected_gross_rev = UnitConversionService.round_currency(saleable_prod * price_dec)
        transport_total = UnitConversionService.round_currency(saleable_prod * trans_cost_dec)
        expected_net_rev = UnitConversionService.round_currency(expected_gross_rev - transport_total)
        expected_net_return = UnitConversionService.round_currency(expected_net_rev - total_cost)

        # Break-even metrics
        break_even_price = Decimal("0.00")
        if saleable_prod > Decimal("0.00"):
            break_even_price = UnitConversionService.round_currency(total_cost / saleable_prod)

        break_even_yield = Decimal("0.00")
        if area_dec > Decimal("0.00") and price_dec > Decimal("0.00"):
            break_even_yield = (total_cost / (area_dec * price_dec)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        roi_pct = Decimal("0.0")
        if total_cost > Decimal("0.00"):
            roi_pct = ((expected_net_return / total_cost) * Decimal("100.0")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Sensitivity Analysis (-10%, +10% price and yield)
        sensitivity = {
            "price_minus_10pct": float(
                UnitConversionService.round_currency(
                    (saleable_prod * (price_dec * Decimal("0.90"))) - transport_total - total_cost
                )
            ),
            "price_plus_10pct": float(
                UnitConversionService.round_currency(
                    (saleable_prod * (price_dec * Decimal("1.10"))) - transport_total - total_cost
                )
            ),
            "yield_minus_10pct": float(
                UnitConversionService.round_currency(
                    ((saleable_prod * Decimal("0.90")) * price_dec)
                    - ((saleable_prod * Decimal("0.90")) * trans_cost_dec)
                    - total_cost
                )
            ),
            "yield_plus_10pct": float(
                UnitConversionService.round_currency(
                    ((saleable_prod * Decimal("1.10")) * price_dec)
                    - ((saleable_prod * Decimal("1.10")) * trans_cost_dec)
                    - total_cost
                )
            ),
        }

        return ScenarioResult(
            scenario_name=req.scenario_name,
            scenario_type=req.scenario_type,
            crop_name=req.crop_name,
            planned_area_acres=float(area_dec),
            total_expected_cost=float(total_cost),
            gross_production_quintals=float(gross_prod),
            saleable_production_quintals=float(saleable_prod),
            expected_gross_revenue=float(expected_gross_rev),
            expected_net_revenue=float(expected_net_rev),
            expected_net_return=float(expected_net_return),
            gross_margin=float(expected_net_rev - total_cost),
            break_even_price_per_quintal=float(break_even_price),
            break_even_yield_quintals_per_acre=float(break_even_yield),
            return_on_cost_pct=float(roi_pct),
            assumptions_summary={
                "cost_per_acre": float(cost_per_acre_dec),
                "yield_per_acre": float(yield_per_acre_dec),
                "expected_price": float(price_dec),
                "harvest_loss_pct": float(loss_pct_dec * Decimal("100.0")),
                "transport_cost_per_quintal": float(trans_cost_dec),
                "mandi_reference_price": req.mandi_reference_price,
            },
            sensitivity_analysis=sensitivity,
        )

    def save_scenario(
        self,
        farmer_id: str,
        req: ScenarioSimulateRequest,
        user_id: str,
        result: Optional[ScenarioResult] = None,
    ) -> FinancialScenarioRecord:
        """Simulate and persist a reproducible financial scenario."""
        if result is None:
            result = self.simulate_scenario(req)
        scenario_id = f"scn_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()

        record_data = {
            "scenario_id": scenario_id,
            "farmer_id": farmer_id,
            "farm_id": req.farm_id,
            "plot_id": req.plot_id,
            "crop_cycle_id": req.crop_cycle_id,
            "crop_name": req.crop_name,
            "scenario_name": req.scenario_name,
            "scenario_type": req.scenario_type.value,
            "assumptions_json": json.dumps(req.model_dump()),
            "results_json": json.dumps(result.model_dump()),
            "version_number": 1,
            "created_at": now_str,
            "updated_at": now_str,
        }

        self.repo.create_financial_scenario(record_data)
        self.repo.record_financial_audit(
            entity_type="financial_scenario",
            entity_id=scenario_id,
            farmer_id=farmer_id,
            action="CREATE",
            changed_fields=record_data,
            performed_by=user_id,
        )
        logger.info(f"Saved scenario {scenario_id} ({req.scenario_name}) for farmer {farmer_id}")
        return FinancialScenarioRecord(**record_data)

    def get_saved_scenario(
        self, scenario_id: str, farmer_id: str
    ) -> FinancialScenarioRecord:
        """Fetch saved scenario by ID."""
        data = self.repo.get_financial_scenario(scenario_id, farmer_id=farmer_id)
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Financial scenario '{scenario_id}' not found.",
            )
        return FinancialScenarioRecord(**data)

    def list_saved_scenarios(
        self, farmer_id: str, farm_id: Optional[str] = None, crop_name: Optional[str] = None
    ) -> List[FinancialScenarioRecord]:
        """List saved scenarios for farmer."""
        records = self.repo.list_financial_scenarios(farmer_id, farm_id=farm_id, crop_name=crop_name)
        return [FinancialScenarioRecord(**r) for r in records]

    # Aliases
    get_scenario = get_saved_scenario
    list_scenarios = list_saved_scenarios

    def delete_scenario(self, scenario_id: str, farmer_id: str, user_id: str) -> bool:
        """Delete saved scenario."""
        existing = self.get_saved_scenario(scenario_id, farmer_id)
        success = self.repo.delete_financial_scenario(scenario_id, farmer_id)
        if success:
            self.repo.record_financial_audit(
                entity_type="financial_scenario",
                entity_id=scenario_id,
                farmer_id=farmer_id,
                action="DELETE",
                changed_fields=existing.model_dump(),
                performed_by=user_id,
            )
        return success

    def compare_crops(self, req: CropComparisonRequest) -> CropComparisonReport:
        """Compare the economics of multiple alternative crops for a plot."""
        if not req.crops:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one crop must be provided for economic comparison.",
            )

        area_dec = UnitConversionService.to_decimal(req.plot_area_acres)
        items: List[CropComparisonItem] = []

        for c in req.crops:
            cost_dec = UnitConversionService.to_decimal(c.estimated_cost_per_acre)
            yield_dec = UnitConversionService.to_decimal(c.expected_yield_per_acre)
            price_dec = UnitConversionService.to_decimal(c.expected_price_per_quintal)

            gross_rev_per_acre = UnitConversionService.round_currency(yield_dec * price_dec)
            net_return_per_acre = UnitConversionService.round_currency(gross_rev_per_acre - cost_dec)

            break_even_p = Decimal("0.00")
            if yield_dec > Decimal("0.00"):
                break_even_p = UnitConversionService.round_currency(cost_dec / yield_dec)

            roi = Decimal("0.0")
            if cost_dec > Decimal("0.00"):
                roi = ((net_return_per_acre / cost_dec) * Decimal("100.0")).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )

            items.append(
                CropComparisonItem(
                    crop_name=c.crop_name,
                    variety_name=c.variety_name,
                    estimated_cost_per_acre=float(cost_dec),
                    expected_yield_per_acre=float(yield_dec),
                    expected_price_per_quintal=float(price_dec),
                    gross_revenue_per_acre=float(gross_rev_per_acre),
                    net_return_per_acre=float(net_return_per_acre),
                    break_even_price=float(break_even_p),
                    roi_pct=float(roi),
                    data_source=c.data_source,
                )
            )

        # Sort to determine best return and best ROI
        best_return_crop = max(items, key=lambda x: x.net_return_per_acre).crop_name
        best_roi_crop = max(items, key=lambda x: x.roi_pct).crop_name

        return CropComparisonReport(
            season=req.season,
            plot_area_acres=float(area_dec),
            comparison_items=items,
            best_net_return_crop=best_return_crop,
            best_roi_crop=best_roi_crop,
        )


_financial_scenario_service_instance: Optional[FinancialScenarioService] = None


def get_financial_scenario_service(repo: Optional[FarmDatabaseRepository] = None) -> FinancialScenarioService:
    """Singleton getter for FinancialScenarioService."""
    global _financial_scenario_service_instance
    if _financial_scenario_service_instance is None or repo is not None:
        _financial_scenario_service_instance = FinancialScenarioService(repo=repo)
    return _financial_scenario_service_instance

