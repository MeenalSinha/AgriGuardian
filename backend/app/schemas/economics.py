"""Pydantic schemas and data models for Phase 11 Farm Economics & Profitability Intelligence."""

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExpenseCategory(str, Enum):
    """Authoritative expense categories for agricultural accounting."""
    SEEDS_PLANTING = "seeds_planting"
    FERTILIZER_SOIL = "fertilizer_soil"
    CROP_PROTECTION = "crop_protection"
    LABOR = "labor"
    MACHINERY_FUEL = "machinery_fuel"
    IRRIGATION_ENERGY = "irrigation_energy"
    LAND_OPERATIONAL = "land_operational"
    OTHER = "other"


class PaymentStatus(str, Enum):
    """Payment completion status for expenses and sales."""
    PAID = "paid"
    UNPAID = "unpaid"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ScenarioType(str, Enum):
    """Scenario simulation assumptions."""
    CONSERVATIVE = "conservative"
    BASE = "base"
    OPTIMISTIC = "optimistic"


# -----------------------------------------------------------------------------
# EXPENSE MODELS
# -----------------------------------------------------------------------------

class FarmExpenseCreate(BaseModel):
    """Payload to create an agricultural expense entry."""
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    expense_category: ExpenseCategory
    subcategory: Optional[str] = None
    expense_description: str
    input_name: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_amount: float = Field(..., gt=0, description="Total amount in currency units")
    currency: str = "INR"
    purchase_date: str  # YYYY-MM-DD
    payment_date: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PAID
    is_planned: bool = False
    supplier_vendor: Optional[str] = None
    invoice_attachment_id: Optional[str] = None
    notes: Optional[str] = None
    entry_source: str = "manual"


class FarmExpenseUpdate(BaseModel):
    """Payload to update an existing expense entry."""
    expense_category: Optional[ExpenseCategory] = None
    subcategory: Optional[str] = None
    expense_description: Optional[str] = None
    input_name: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_amount: Optional[float] = Field(None, gt=0)
    purchase_date: Optional[str] = None
    payment_date: Optional[str] = None
    payment_status: Optional[PaymentStatus] = None
    is_planned: Optional[bool] = None
    supplier_vendor: Optional[str] = None
    notes: Optional[str] = None


class FarmExpenseRecord(BaseModel):
    """Full persisted agricultural expense record."""
    expense_id: str
    farmer_id: str
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    expense_category: ExpenseCategory
    subcategory: Optional[str] = None
    expense_description: str
    input_name: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_amount: float
    currency: str = "INR"
    purchase_date: str
    payment_date: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PAID
    is_planned: bool = False
    supplier_vendor: Optional[str] = None
    invoice_attachment_id: Optional[str] = None
    notes: Optional[str] = None
    entry_source: str = "manual"
    version_number: int = 1
    created_at: str
    updated_at: str


class ExpenseCorrectionCreate(BaseModel):
    """Payload to create an auditable correction entry on an expense."""
    reason: str
    new_amount: float = Field(..., gt=0)


class ExpenseCorrectionRecord(BaseModel):
    """Immutable audit correction record."""
    correction_id: str
    original_expense_id: str
    farmer_id: str
    reason: str
    previous_amount: float
    new_amount: float
    corrected_by: str
    created_at: str


# -----------------------------------------------------------------------------
# HARVEST & PRODUCTION MODELS
# -----------------------------------------------------------------------------

class FarmHarvestCreate(BaseModel):
    """Payload to record a crop harvest event."""
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    harvest_date: str  # YYYY-MM-DD
    crop_name: str
    variety_name: Optional[str] = None
    harvested_quantity: float = Field(..., gt=0, description="Gross quantity harvested")
    quantity_unit: str = "quintal"
    quality_grade: Optional[str] = None  # Grade A, Grade B, Fair Average Quality (FAQ)
    damaged_quantity: float = Field(default=0.0, ge=0)
    rejected_quantity: float = Field(default=0.0, ge=0)
    stored_quantity: float = Field(default=0.0, ge=0)
    sold_quantity: float = Field(default=0.0, ge=0)
    retained_quantity: float = Field(default=0.0, ge=0)  # Self-consumption or seeds
    moisture_percentage: Optional[float] = None
    notes: Optional[str] = None
    attachment_id: Optional[str] = None


class FarmHarvestUpdate(BaseModel):
    """Payload to update an existing harvest record."""
    harvest_date: Optional[str] = None
    harvested_quantity: Optional[float] = Field(None, gt=0)
    quantity_unit: Optional[str] = None
    quality_grade: Optional[str] = None
    damaged_quantity: Optional[float] = Field(None, ge=0)
    rejected_quantity: Optional[float] = Field(None, ge=0)
    stored_quantity: Optional[float] = Field(None, ge=0)
    sold_quantity: Optional[float] = Field(None, ge=0)
    retained_quantity: Optional[float] = Field(None, ge=0)
    moisture_percentage: Optional[float] = None
    notes: Optional[str] = None


class FarmHarvestRecord(BaseModel):
    """Full persisted harvest record."""
    harvest_id: str
    farmer_id: str
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    harvest_date: str
    crop_name: str
    variety_name: Optional[str] = None
    harvested_quantity: float
    quantity_unit: str = "quintal"
    quality_grade: Optional[str] = None
    damaged_quantity: float = 0.0
    rejected_quantity: float = 0.0
    stored_quantity: float = 0.0
    sold_quantity: float = 0.0
    retained_quantity: float = 0.0
    moisture_percentage: Optional[float] = None
    notes: Optional[str] = None
    attachment_id: Optional[str] = None
    version_number: int = 1
    created_at: str
    updated_at: str


# -----------------------------------------------------------------------------
# SALES & REVENUE MODELS
# -----------------------------------------------------------------------------

class FarmSaleCreate(BaseModel):
    """Payload to record an actual produce sale."""
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    harvest_id: Optional[str] = None
    crop_name: str
    quantity_sold: float = Field(..., gt=0)
    quantity_unit: str = "quintal"
    realized_price_per_unit: float = Field(..., gt=0)
    gross_amount: Optional[float] = None  # Calculated if omitted: quantity * price
    deductions_charges: float = Field(default=0.0, ge=0)  # Commission, cess, loading
    net_amount_received: Optional[float] = None  # Calculated if omitted: gross - deductions
    currency: str = "INR"
    sale_date: str  # YYYY-MM-DD
    buyer_name: Optional[str] = None
    mandi_id: Optional[str] = None
    quality_grade: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PAID
    payment_received_date: Optional[str] = None
    transport_charges: float = Field(default=0.0, ge=0)
    commission_charges: float = Field(default=0.0, ge=0)
    attachment_id: Optional[str] = None
    notes: Optional[str] = None


class FarmSaleUpdate(BaseModel):
    """Payload to update an existing sale record."""
    quantity_sold: Optional[float] = Field(None, gt=0)
    realized_price_per_unit: Optional[float] = Field(None, gt=0)
    deductions_charges: Optional[float] = Field(None, ge=0)
    sale_date: Optional[str] = None
    buyer_name: Optional[str] = None
    payment_status: Optional[PaymentStatus] = None
    payment_received_date: Optional[str] = None
    transport_charges: Optional[float] = Field(None, ge=0)
    commission_charges: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class FarmSaleRecord(BaseModel):
    """Full persisted farm sale record."""
    sale_id: str
    farmer_id: str
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    harvest_id: Optional[str] = None
    crop_name: str
    quantity_sold: float
    quantity_unit: str = "quintal"
    realized_price_per_unit: float
    gross_amount: float
    deductions_charges: float = 0.0
    net_amount_received: float
    currency: str = "INR"
    sale_date: str
    buyer_name: Optional[str] = None
    mandi_id: Optional[str] = None
    quality_grade: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PAID
    payment_received_date: Optional[str] = None
    transport_charges: float = 0.0
    commission_charges: float = 0.0
    attachment_id: Optional[str] = None
    notes: Optional[str] = None
    version_number: int = 1
    created_at: str
    updated_at: str


# -----------------------------------------------------------------------------
# BUDGET MODELS
# -----------------------------------------------------------------------------

class BudgetItemCreate(BaseModel):
    """Category budget item."""
    expense_category: ExpenseCategory
    planned_amount: float = Field(..., ge=0)
    notes: Optional[str] = None


class BudgetItemRecord(BaseModel):
    """Persisted category budget item."""
    item_id: str
    budget_id: str
    expense_category: ExpenseCategory
    planned_amount: float
    notes: Optional[str] = None
    created_at: str
    updated_at: str


class CropCycleBudgetCreate(BaseModel):
    """Payload to create an entire crop cycle budget."""
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    budget_name: str
    season: Optional[str] = None
    total_budget_amount: float = Field(..., gt=0)
    currency: str = "INR"
    items: List[BudgetItemCreate] = []


class CropCycleBudgetUpdate(BaseModel):
    """Payload to update an existing budget."""
    budget_name: Optional[str] = None
    total_budget_amount: Optional[float] = Field(None, gt=0)
    status: Optional[str] = None
    items: Optional[List[BudgetItemCreate]] = None


class CropCycleBudgetRecord(BaseModel):
    """Persisted crop cycle budget header."""
    budget_id: str
    farmer_id: str
    farm_id: str
    plot_id: Optional[str] = None
    crop_cycle_id: str
    budget_name: str
    season: Optional[str] = None
    total_budget_amount: float
    currency: str = "INR"
    version_number: int = 1
    status: str = "active"
    created_at: str
    updated_at: str
    items: List[BudgetItemRecord] = []


class BudgetVarianceItem(BaseModel):
    """Comparison item between budget and actual expenditure."""
    expense_category: ExpenseCategory
    planned_amount: float
    actual_amount: float
    variance_amount: float  # actual - planned (positive = over budget)
    variance_percentage: float  # % over/under
    is_over_budget: bool


class BudgetVarianceReport(BaseModel):
    """Full budget vs actual report for a crop cycle."""
    budget_id: str
    crop_cycle_id: str
    total_planned: float
    total_actual: float
    remaining_budget: float
    amount_over_budget: float
    total_variance_pct: float
    is_exceeded: bool
    categories: List[BudgetVarianceItem]


# -----------------------------------------------------------------------------
# PROFITABILITY & FINANCIAL OVERVIEW MODELS
# -----------------------------------------------------------------------------

class ProfitabilityMetrics(BaseModel):
    """Comprehensive, transparent financial metrics for a crop cycle."""
    crop_cycle_id: str
    crop_name: str
    variety_name: Optional[str] = None
    farm_id: str
    farm_name: str
    cultivated_area_acres: float
    is_provisional: bool = False
    missing_data_fields: List[str] = []
    
    # Costs
    total_actual_cost: float
    cost_breakdown_by_category: Dict[str, float] = {}
    
    # Revenues
    total_gross_revenue: float
    total_net_revenue: float
    total_deductions: float
    
    # Margins & Profits
    gross_margin: float  # Revenue minus variable cultivation cost
    net_profit_or_loss: float  # Net Revenue minus Total Cost
    is_profitable: bool
    
    # Normalized Unit Metrics
    cost_per_acre: Optional[float] = None
    cost_per_quintal: Optional[float] = None
    cost_per_kg: Optional[float] = None
    
    # Production & Sales Quantities
    total_production_quintals: float = 0.0
    total_sold_quintals: float = 0.0
    average_realized_price_per_quintal: Optional[float] = None
    
    # Break-Even Analysis
    break_even_selling_price_per_quintal: Optional[float] = None
    break_even_production_quintals: Optional[float] = None
    
    # Return on Investment
    return_on_cultivation_cost_pct: Optional[float] = None
    
    # Explanation & Provenance
    calculation_timestamp: str
    explanation_en: str
    explanation_hi: str


class FarmerFinancialOverview(BaseModel):
    """High-level financial status summary for a farmer."""
    farmer_id: str
    total_cultivation_expenses: float
    total_paid_expenses: float
    total_outstanding_expenses: float
    total_realized_revenue: float
    net_balance: float
    active_crop_cycles_cost: float
    completed_crop_cycles_profit: float
    provisional_cycles_count: int
    completed_cycles_count: int
    recent_expenses: List[FarmExpenseRecord] = []
    recent_sales: List[FarmSaleRecord] = []


# -----------------------------------------------------------------------------
# SCENARIO SIMULATOR & CROP COMPARISON MODELS
# -----------------------------------------------------------------------------

class ScenarioSimulateRequest(BaseModel):
    """Input parameters to simulate financial outcomes."""
    crop_name: str
    farm_id: Optional[str] = None
    plot_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    scenario_name: str
    scenario_type: ScenarioType = ScenarioType.BASE
    planned_area_acres: float = Field(..., gt=0)
    expected_cost_per_acre: float = Field(..., gt=0)
    expected_yield_quintals_per_acre: float = Field(..., gt=0)
    expected_price_per_quintal: float = Field(..., gt=0)
    expected_harvest_loss_pct: float = Field(default=0.0, ge=0, le=100)
    transport_cost_per_quintal: float = Field(default=0.0, ge=0)
    mandi_reference_id: Optional[str] = None
    mandi_reference_price: Optional[float] = None


class ScenarioResult(BaseModel):
    """Deterministic, transparent result of a scenario simulation."""
    scenario_name: str
    scenario_type: ScenarioType
    crop_name: str
    planned_area_acres: float
    total_expected_cost: float
    gross_production_quintals: float
    saleable_production_quintals: float
    expected_gross_revenue: float
    expected_net_revenue: float
    expected_net_return: float
    gross_margin: float
    break_even_price_per_quintal: float
    break_even_yield_quintals_per_acre: float
    return_on_cost_pct: float
    assumptions_summary: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]  # Impact of -10%, +10% price/yield


class FinancialScenarioRecord(BaseModel):
    """Persisted scenario for reproducibility."""
    scenario_id: str
    farmer_id: str
    farm_id: Optional[str] = None
    plot_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    crop_name: str
    scenario_name: str
    scenario_type: ScenarioType = ScenarioType.BASE
    assumptions_json: str
    results_json: str
    version_number: int = 1
    created_at: str
    updated_at: str


class CropComparisonItem(BaseModel):
    """Single crop evaluation in a multi-crop comparison."""
    crop_name: str
    variety_name: Optional[str] = None
    estimated_cost_per_acre: float
    expected_yield_per_acre: float
    expected_price_per_quintal: float
    gross_revenue_per_acre: float
    net_return_per_acre: float
    break_even_price: float
    roi_pct: float
    data_source: str  # "farmer_history", "benchmark", "assumption"


class CropComparisonRequest(BaseModel):
    """Request to compare 2 or more prospective crops for a plot."""
    plot_area_acres: float = Field(..., gt=0)
    season: str
    state: str
    district: str
    crops: List[CropComparisonItem]


class CropComparisonReport(BaseModel):
    """Comparative economic assessment between multiple crops."""
    season: str
    plot_area_acres: float
    comparison_items: List[CropComparisonItem]
    best_net_return_crop: str
    best_roi_crop: str
    disclaimer: str = (
        "Agricultural scenario comparisons are based strictly on supplied assumptions and historical records. "
        "They do not constitute guaranteed profit forecasts or investment advice."
    )


# -----------------------------------------------------------------------------
# BENCHMARK REFERENCE MODELS
# -----------------------------------------------------------------------------

class FinancialBenchmarkRecord(BaseModel):
    """Verified external cost & yield benchmark."""
    benchmark_id: str
    crop_name: str
    state: str
    district: Optional[str] = None
    reporting_period: str
    cost_per_acre: float
    expected_yield_per_acre: float
    yield_unit: str = "quintal"
    source_name: str
    source_doc: Optional[str] = None
    verification_status: str = "verified"
    created_at: str
    updated_at: str

