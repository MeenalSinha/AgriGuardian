"""
AgriGuardian Phase 11 — Economics Master Service Facade
Aggregates all farm economics sub-services:
  - Expenses & Labor Tracking
  - Harvest & Yield Tracking
  - Sales & Revenue Tracking
  - Profitability & Break-Even Intelligence
  - Budgeting & Variance Analysis
  - Financial Scenarios & Crop Comparison
  - Regional Financial Benchmarks
"""

import logging
from typing import Optional, List, Dict, Any

from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import FinancialBenchmarkRecord
from backend.app.services.economics.expense_service import ExpenseService, get_expense_service
from backend.app.services.economics.harvest_service import HarvestService, get_harvest_service
from backend.app.services.economics.sales_service import SalesService, get_sales_service
from backend.app.services.economics.profitability_service import ProfitabilityService, get_profitability_service
from backend.app.services.economics.budget_service import BudgetService, get_budget_service
from backend.app.services.economics.financial_scenario_service import FinancialScenarioService, get_financial_scenario_service

logger = logging.getLogger("agri_guard.economics_service")


class EconomicsService:
    """Master facade for AgriGuardian Phase 11 Farm Economics & Profitability."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()
        self.expenses: ExpenseService = get_expense_service(self.repo)
        self.harvests: HarvestService = get_harvest_service(self.repo)
        self.sales: SalesService = get_sales_service(self.repo)
        self.profitability: ProfitabilityService = get_profitability_service(self.repo)
        self.budgets: BudgetService = get_budget_service(self.repo)
        self.scenarios: FinancialScenarioService = get_financial_scenario_service(self.repo)

    # -------------------------------------------------------------------------
    # Benchmarks
    # -------------------------------------------------------------------------
    def list_benchmarks(
        self,
        crop_name: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        season: Optional[str] = None,
    ) -> List[FinancialBenchmarkRecord]:
        """List regional/ICAR/CACP financial benchmarks."""
        return self.repo.list_financial_benchmarks(
            crop_name=crop_name,
            state=state,
            district=district,
            season=season,
        )

    def create_benchmark(self, benchmark: FinancialBenchmarkRecord) -> FinancialBenchmarkRecord:
        """Record a regional financial benchmark."""
        return self.repo.create_financial_benchmark(benchmark)


_economics_service_instance: Optional[EconomicsService] = None


def get_economics_service(repo: Optional[FarmDatabaseRepository] = None) -> EconomicsService:
    """Singleton factory for EconomicsService."""
    global _economics_service_instance
    if _economics_service_instance is None or repo is not None:
        _economics_service_instance = EconomicsService(repo=repo)
    return _economics_service_instance
