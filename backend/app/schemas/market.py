"""Standardized agricultural market data models, master catalogs, and analytical schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CommodityCategory(str, Enum):
    """Broad agricultural commodity categories."""
    CEREALS = "cereals"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    OILSEEDS = "oilseeds"
    PULSES = "pulses"
    FIBER = "fiber"
    SUGARCANE = "sugarcane"
    SPICES = "spices"


class PriceTrendDirection(str, Enum):
    """Statistical classification of market price trajectory."""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class MarketDataSource(str, Enum):
    """Authorized agricultural market intelligence data origins."""
    AGMARKNET = "AGMARKNET (Directorate of Marketing & Inspection, MoA&FW)"
    ENAM = "e-NAM (National Agriculture Market, Small Farmers' Agri-Business Consortium)"
    CACP_MSP = "CACP / DA&FW (Commission for Agricultural Costs & Prices, GoI)"
    STATE_APMC = "State Agricultural Marketing Board / APMC Daily Bulletin"


# ==============================================================================
# MASTER REFERENCE SCHEMAS
# ==============================================================================

class CommodityRecord(BaseModel):
    """Standardized commodity master catalog entry."""
    commodity_id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'CMD-WHEAT')")
    canonical_name: str = Field(..., description="Canonical standard English name")
    hindi_name: str = Field(..., description="Standard Hindi name in Devanagari script")
    scientific_name: Optional[str] = Field(None, description="Botanical/biological name")
    category: CommodityCategory = Field(..., description="Commodity category")
    standard_unit: str = Field("₹/quintal", description="Standard price unit")
    aliases: List[str] = Field(default_factory=list, description="Multilingual search synonyms")
    is_active: bool = Field(True)


class VarietyRecord(BaseModel):
    """Standardized commodity variety reference entry."""
    variety_id: str = Field(..., description="Unique variety identifier (e.g. 'VAR-TOM-HYBRID')")
    commodity_id: str = Field(..., description="Parent commodity ID")
    variety_name: str = Field(..., description="Market variety name")
    synonyms: List[str] = Field(default_factory=list)
    grade: str = Field("Standard / FAQ", description="Fair Average Quality (FAQ) or commercial grade")


class MandiRecord(BaseModel):
    """Standardized APMC market master catalog entry."""
    market_id: str = Field(..., description="Unique market identifier (e.g. 'MKT-UP-MEERUT')")
    market_name: str = Field(..., description="Market/mandi name")
    apmc_name: str = Field(..., description="Regulated APMC authority name")
    state: str = Field(..., description="Indian state")
    district: str = Field(..., description="District")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Verified market latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Verified market longitude")
    is_active: bool = Field(True)


# ==============================================================================
# OBSERVATION & TRANSACTION SCHEMAS
# ==============================================================================

class MarketPriceObservation(BaseModel):
    """Validated daily market price observation for a commodity in an APMC mandi."""
    observation_id: str = Field(..., description="Unique deterministic identifier")
    commodity_id: str = Field(...)
    commodity_name: str = Field(...)
    variety_id: Optional[str] = None
    variety_name: str = Field("Standard / Other", description="Observed variety or FAQ")
    market_id: str = Field(...)
    market_name: str = Field(...)
    state: str = Field(...)
    district: str = Field(...)
    reporting_date: str = Field(..., description="Market transaction/reporting date (YYYY-MM-DD)")
    
    min_price: float = Field(..., ge=0.0, description="Lowest reported price in ₹/quintal")
    max_price: float = Field(..., ge=0.0, description="Highest reported price in ₹/quintal")
    modal_price: float = Field(..., ge=0.0, description="Most commonly transacted price in ₹/quintal")
    
    unit: str = Field("₹/quintal", description="Standard price unit")
    arrivals_tonnes: Optional[float] = Field(None, ge=0.0, description="Commodity arrivals in metric tonnes")
    traded_quantity_tonnes: Optional[float] = Field(None, ge=0.0, description="Traded volume in metric tonnes")
    
    source: MarketDataSource = Field(..., description="Authoritative data provider")
    retrieval_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_live: bool = Field(False, description="True ONLY if real-time intraday auction, False for daily closing bulletin")
    data_quality_status: str = Field("verified", description="'verified', 'flagged', 'outlier'")

    @field_validator("modal_price")
    @classmethod
    def validate_modal_within_bounds(cls, v: float, info: Any) -> float:
        min_p = info.data.get("min_price")
        max_p = info.data.get("max_price")
        if min_p is not None and v < min_p:
            raise ValueError(f"Modal price ({v}) cannot be less than minimum price ({min_p}).")
        if max_p is not None and v > max_p:
            raise ValueError(f"Modal price ({v}) cannot be greater than maximum price ({max_p}).")
        return round(v, 2)


class MSPRecord(BaseModel):
    """Official Minimum Support Price (MSP) policy benchmark."""
    msp_id: str = Field(..., description="Unique MSP record ID (e.g. 'MSP-WHEAT-2024-25')")
    crop: str = Field(..., description="Commodity name")
    commodity_id: str = Field(...)
    marketing_season: str = Field(..., description="Marketing season (e.g. 'Rabi Marketing Season (RMS) 2024-25')")
    msp_price: float = Field(..., ge=0.0, description="Official MSP in ₹/quintal")
    unit: str = Field("₹/quintal")
    announcement_date: str = Field(..., description="Cabinet Committee on Economic Affairs (CCEA) approval date")
    effective_period: str = Field(..., description="Procurement operating window")
    official_source: str = Field(
        "Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        description="Statutory issuing authority"
    )
    official_notification_ref: str = Field(..., description="Government gazette/press bulletin citation")
    eligible_variety: str = Field("Fair Average Quality (FAQ)", description="Statutory quality prerequisite")
    procurement_agencies: List[str] = Field(
        default_factory=lambda: ["FCI", "NAFED", "State Procurement Agencies"],
        description="Designated public procurement entities"
    )
    limitations_disclaimer: str = Field(
        "MSP is an indicative government procurement benchmark, not a statutory floor price guaranteed at private APMC auctions.",
        description="Regulatory transparency disclaimer"
    )


# ==============================================================================
# ANALYTICS & DECISION SUPPORT SCHEMAS
# ==============================================================================

class NearbyMandiPrice(BaseModel):
    """Mandi price comparison entry paired with geographic straight-line distance."""
    market_id: str
    market_name: str
    state: str
    district: str
    distance_km: float = Field(..., ge=0.0, description="Straight-line distance from farmer location in km")
    modal_price: float
    min_price: float
    max_price: float
    variety: str
    reporting_date: str
    data_freshness: str
    arrivals_tonnes: Optional[float] = None
    source: str
    is_live: bool = False


class MarketTrendAnalysis(BaseModel):
    """Historical time-series trend and volatility analytics."""
    commodity_id: str
    commodity_name: str
    market_id: str
    market_name: str
    analysis_window_days: int
    observations_count: int
    latest_price: float
    previous_price: Optional[float] = None
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None
    period_average: float
    period_median: float
    period_min: float
    period_max: float
    price_range: float
    volatility_std: float = Field(..., description="Sample standard deviation of modal price")
    trend_direction: PriceTrendDirection
    trend_explanation: str
    data_completeness_pct: float
    history_points: List[Dict[str, Any]] = Field(default_factory=list)


class PriceForecastPoint(BaseModel):
    """Single-day predicted future price with prediction uncertainty interval."""
    forecast_date: str = Field(..., description="Future date (YYYY-MM-DD)")
    predicted_modal_price: float = Field(..., description="Point forecast in ₹/quintal")
    lower_interval_80: float = Field(..., description="Lower 80% confidence interval bound")
    upper_interval_80: float = Field(..., description="Upper 80% confidence interval bound")
    lower_interval_95: float = Field(..., description="Lower 95% confidence interval bound")
    upper_interval_95: float = Field(..., description="Upper 95% confidence interval bound")


class PriceForecastResult(BaseModel):
    """Comprehensive time-series forecast payload with model evaluation metrics."""
    commodity_id: str
    commodity_name: str
    market_id: str
    market_name: str
    variety_name: str
    forecast_horizon_days: int
    model_name: str = Field(..., description="Forecasting model (e.g. 'Holt-Winters Exponential Smoothing', 'Rolling SMA')")
    model_version: str = Field("1.0.0")
    training_observations_count: int
    last_historical_date: str
    last_observed_modal_price: float
    forecast_points: List[PriceForecastPoint] = Field(default_factory=list)
    backtest_mae: float = Field(..., description="Out-of-sample Mean Absolute Error (₹/quintal)")
    backtest_rmse: float = Field(..., description="Out-of-sample Root Mean Squared Error (₹/quintal)")
    backtest_directional_accuracy_pct: float = Field(..., description="Percentage of correct directional movement predictions")
    baseline_naive_mae: float = Field(..., description="Persistence baseline MAE for comparative benchmark")
    model_outperformed_baseline: bool = Field(True)
    confidence_tier: str = Field("Medium", description="'High', 'Medium', 'Low', 'Insufficient_Data'")
    statutory_disclaimer: str = Field(
        "Forecasting is decision-support based strictly on historical market patterns. Future prices are subject to weather shocks, policy changes, and supply variations and are NEVER guaranteed.",
        description="Mandatory economic disclaimer"
    )


class NetRealizationResult(BaseModel):
    """Comparative net financial return calculation factoring logistics and handling costs."""
    market_id: str
    market_name: str
    distance_km: float
    commodity_name: str
    quantity_quintals: float
    reference_modal_price: float
    gross_value_inr: float
    transport_cost_inr: float
    handling_fees_inr: float
    total_estimated_costs_inr: float
    estimated_net_realization_inr: float
    net_realization_per_quintal: float
    assumptions: List[str] = Field(default_factory=list)
    disclaimer: str = Field(
        "Estimated net realization is an analytical calculation based on user-supplied costs and reference modal prices. Actual auction settlement depends on individual lot quality, moisture, grading, and weighment."
    )
