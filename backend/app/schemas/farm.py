"""Pydantic schemas for Farmer Identity, Multi-Farm Management, Crop Lifecycle Tracking, and Farm Context."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class UserRole(str, Enum):
    FARMER = "farmer"
    FIELD_OFFICER = "field_officer"
    AGRONOMIST = "agronomist"
    ADMINISTRATOR = "administrator"


class AreaUnit(str, Enum):
    ACRE = "acre"
    HECTARE = "hectare"
    BIGHA = "bigha"
    GUNTHA = "guntha"


class OwnershipType(str, Enum):
    OWNED = "owned"
    LEASED = "leased"
    SHARECROPPING = "sharecropping"


class SoilType(str, Enum):
    ALLUVIAL = "alluvial"
    BLACK_COTTON = "black_cotton"
    RED_LOAM = "red_loam"
    LATERITE = "laterite"
    SANDY_LOAM = "sandy_loam"
    CLAY_LOAM = "clay_loam"
    SALINE = "saline"
    PEATY = "peaty"
    UNKNOWN = "unknown"


class IrrigationMethod(str, Enum):
    DRIP = "drip"
    SPRINKLER = "sprinkler"
    FLOOD_FURROW = "flood_furrow"
    BASIN = "basin"
    RAINFED = "rainfed"


class WaterAvailability(str, Enum):
    ABUNDANT = "abundant"
    ADEQUATE = "adequate"
    SCARCE = "scarce"
    SEASONAL = "seasonal"


class PumpingSource(str, Enum):
    SOLAR_PUMP = "solar_pump"
    ELECTRIC_GRID = "electric_grid"
    DIESEL_GENERATOR = "diesel_generator"
    GRAVITY_FLOW = "gravity_flow"
    MANUAL = "manual"
    NONE = "none"


class CropSeason(str, Enum):
    KHARIF = "kharif"
    RABI = "rabi"
    ZAID = "zaid"


class CropGrowthStage(str, Enum):
    SOWING_GERMINATION = "sowing_germination"
    SEEDLING_EMERGENCE = "seedling_emergence"
    VEGETATIVE_TILLERING = "vegetative_tillering"
    FLOWERING_REPRODUCTIVE = "flowering_reproductive"
    GRAIN_FILLING_MATURITY = "grain_filling_maturity"
    HARVEST_READY = "harvest_ready"
    HARVESTED = "harvested"


class CropCycleStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ActivityType(str, Enum):
    LAND_PREPARATION = "land_preparation"
    SOWING = "sowing"
    TRANSPLANTING = "transplanting"
    IRRIGATION = "irrigation"
    FERTILIZER_APPLICATION = "fertilizer_application"
    PESTICIDE_SPRAY = "pesticide_spray"
    WEEDING = "weeding"
    INTER_CULTIVATION = "inter_cultivation"
    HARVESTING = "harvesting"
    POST_HARVEST_STORAGE = "post_harvest_storage"
    SOIL_TESTING = "soil_testing"
    MARKET_SALE = "market_sale"


class ActivityStatus(str, Enum):
    PLANNED = "planned"
    DUE = "due"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class LiteracyPreference(str, Enum):
    TEXT = "text"
    VOICE_PRIMARY = "voice_primary"
    AUDIO_VISUAL = "audio_visual"


# --- Authentication & User Schemas ---

class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username or mobile number")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: str = Field(..., min_length=2, max_length=100, description="Farmer full name")
    phone_number: Optional[str] = Field(default=None, description="10-digit mobile number")
    preferred_language: str = Field(default="hi-IN", description="Preferred language code (e.g. hi-IN, en-IN, ta-IN)")
    role: Optional[UserRole] = Field(default=UserRole.FARMER, description="Account role (public registration restricted to FARMER)")

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        """Enforce password complexity: at least 8 chars, containing letters and digits."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        has_alpha = any(c.isalpha() for c in v)
        has_digit = any(c.isdigit() for c in v)
        if not (has_alpha and has_digit):
            raise ValueError("Password must contain both letters and digits for security.")
        return v

    @field_validator("role")
    @classmethod
    def validate_registration_role(cls, v: Optional[UserRole]) -> UserRole:
        """Prevent privilege escalation attacks on public self-registration."""
        if v is not None and v != UserRole.FARMER:
            raise ValueError(
                f"Unauthorized role assignment: Self-registration for '{v.value}' is prohibited. "
                "Privileged accounts must be provisioned by a platform administrator."
            )
        return UserRole.FARMER


class UserLoginRequest(BaseModel):
    username: str = Field(..., description="Username or mobile number")
    password: str = Field(..., description="Account password")


class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    farmer_id: str
    username: str
    full_name: str
    role: UserRole
    preferred_language: str
    profile_completion_pct: float = 0.0


class UserRecord(BaseModel):
    user_id: str
    username: str
    password_hash: str
    salt: str
    role: UserRole
    created_at: str
    last_login: Optional[str] = None


# --- Farmer Profile Schemas ---

class FarmerProfileBase(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    phone_number: Optional[str] = None
    preferred_language: str = "hi-IN"
    experience_years: Optional[int] = Field(default=None, ge=0, le=70)
    literacy_preference: LiteracyPreference = LiteracyPreference.VOICE_PRIMARY
    kisan_credit_card_holder: bool = False
    soil_health_card_available: bool = False


class FarmerProfileCreate(FarmerProfileBase):
    pass


class FarmerProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    preferred_language: Optional[str] = None
    experience_years: Optional[int] = None
    literacy_preference: Optional[LiteracyPreference] = None
    kisan_credit_card_holder: Optional[bool] = None
    soil_health_card_available: Optional[bool] = None


class FarmerProfile(FarmerProfileBase):
    farmer_id: str
    user_id: str
    profile_completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str
    updated_at: str

    @property
    def farmer_name(self) -> str:
        """Convenience alias for full_name."""
        return self.full_name


# --- Farm & Multi-Farm Schemas ---

class FarmBase(BaseModel):
    farm_name: str = Field(..., min_length=1, max_length=100, description="Name or identifier for the farm")
    state: str = Field(..., min_length=2, max_length=50)
    district: str = Field(..., min_length=2, max_length=50)
    tehsil_taluka: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    total_area: float = Field(..., gt=0.0, description="Total land area in specified area_unit")
    area_unit: AreaUnit = AreaUnit.ACRE
    ownership_type: OwnershipType = OwnershipType.OWNED
    is_active_farm: bool = Field(default=False, description="Whether this is the currently active farm context")
    notes: Optional[str] = None


class FarmCreate(FarmBase):
    pass


class FarmUpdate(BaseModel):
    farm_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    tehsil_taluka: Optional[str] = None
    village: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    total_area: Optional[float] = None
    area_unit: Optional[AreaUnit] = None
    ownership_type: Optional[OwnershipType] = None
    is_active_farm: Optional[bool] = None
    notes: Optional[str] = None


class Farm(FarmBase):
    farm_id: str
    farmer_id: str
    total_area_acres: float = Field(..., description="Standardized area in acres for agronomic calculations")
    created_at: str
    updated_at: str
    version_number: int = 1

    @property
    def id(self) -> str:
        return self.farm_id

    @property
    def total_acres(self) -> float:
        return self.total_area_acres


# --- Plot / Field Schemas ---

class PlotBase(BaseModel):
    plot_name: str = Field(..., min_length=1, max_length=100)
    area: float = Field(..., gt=0.0)
    area_unit: AreaUnit = AreaUnit.ACRE
    soil_type: SoilType = SoilType.ALLUVIAL
    irrigation_source: str = Field(default="Borewell", description="Source (e.g. Borewell, Canal, Rain)")
    drainage_type: str = Field(default="Good", description="Drainage capacity (Good, Moderate, Poor)")
    slope: str = Field(default="Flat", description="Topography (Flat, Gentle, Steep)")
    notes: Optional[str] = None


class PlotCreate(PlotBase):
    pass


class PlotUpdate(BaseModel):
    plot_name: Optional[str] = None
    area: Optional[float] = None
    area_unit: Optional[AreaUnit] = None
    soil_type: Optional[SoilType] = None
    irrigation_source: Optional[str] = None
    drainage_type: Optional[str] = None
    slope: Optional[str] = None
    notes: Optional[str] = None


class Plot(PlotBase):
    plot_id: str
    farm_id: str
    farmer_id: str
    area_acres: float = Field(..., description="Standardized area in acres")
    created_at: str
    updated_at: str
    version_number: int = 1

    @property
    def id(self) -> str:
        return self.plot_id


# --- Crop Cycle Schemas ---

class CropCycleBase(BaseModel):
    crop_name: str = Field(..., min_length=2, max_length=50)
    variety_name: Optional[str] = Field(default=None, max_length=100)
    season: CropSeason = CropSeason.KHARIF
    sowing_date: str = Field(..., description="Date of sowing or nursery seed germination (YYYY-MM-DD)")
    transplanting_date: Optional[str] = Field(default=None, description="Date of field transplantation (YYYY-MM-DD)")
    estimated_harvest_date: Optional[str] = Field(default=None, description="Target harvest date (YYYY-MM-DD)")
    target_yield_quintals: Optional[float] = Field(default=None, gt=0.0)
    notes: Optional[str] = None


class CropCycleCreate(CropCycleBase):
    current_stage: CropGrowthStage = CropGrowthStage.SOWING_GERMINATION


class CropCycleUpdate(BaseModel):
    crop_name: Optional[str] = None
    variety_name: Optional[str] = None
    season: Optional[CropSeason] = None
    sowing_date: Optional[str] = None
    transplanting_date: Optional[str] = None
    estimated_harvest_date: Optional[str] = None
    actual_harvest_date: Optional[str] = None
    current_stage: Optional[CropGrowthStage] = None
    stage_confirmed_by_farmer: Optional[bool] = None
    target_yield_quintals: Optional[float] = None
    actual_yield_quintals: Optional[float] = None
    status: Optional[CropCycleStatus] = None
    notes: Optional[str] = None


class CropStageTransitionRequest(BaseModel):
    new_stage: CropGrowthStage
    actual_date: Optional[str] = None
    yield_quintals: Optional[float] = None
    notes: Optional[str] = None


class CropCycle(CropCycleBase):
    cycle_id: str
    plot_id: str
    farm_id: str
    farmer_id: str
    current_stage: CropGrowthStage
    estimated_stage: CropGrowthStage = CropGrowthStage.SOWING_GERMINATION
    days_since_sowing: int = 0
    stage_confirmed_by_farmer: bool = False
    actual_harvest_date: Optional[str] = None
    actual_yield_quintals: Optional[float] = None
    status: CropCycleStatus = CropCycleStatus.ACTIVE
    created_at: str
    updated_at: str
    version_number: int = 1


# --- Soil & Irrigation Profile Schemas ---

class SoilProfileBase(BaseModel):
    soil_type: SoilType = SoilType.ALLUVIAL
    ph_level: Optional[float] = Field(default=None, ge=3.0, le=11.0, description="Soil pH (3.0 - 11.0)")
    organic_carbon_pct: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Organic Carbon %")
    nitrogen_kg_ha: Optional[float] = Field(default=None, ge=0.0, description="Available Nitrogen (kg/ha)")
    phosphorus_kg_ha: Optional[float] = Field(default=None, ge=0.0, description="Available Phosphorus (kg/ha)")
    potassium_kg_ha: Optional[float] = Field(default=None, ge=0.0, description="Available Potassium (kg/ha)")
    electrical_conductivity: Optional[float] = Field(default=None, ge=0.0, description="EC in dS/m")
    micronutrients: Dict[str, Any] = Field(default_factory=dict, description="Zinc, Iron, Boron, etc.")
    is_lab_tested: bool = Field(default=False, description="Whether data is from an accredited soil lab or SHC")
    test_date: Optional[str] = None
    soil_health_card_number: Optional[str] = None
    testing_lab_name: Optional[str] = None


class SoilProfileCreate(SoilProfileBase):
    pass


class SoilProfileUpdate(BaseModel):
    soil_type: Optional[SoilType] = None
    ph_level: Optional[float] = None
    organic_carbon_pct: Optional[float] = None
    nitrogen_kg_ha: Optional[float] = None
    phosphorus_kg_ha: Optional[float] = None
    potassium_kg_ha: Optional[float] = None
    electrical_conductivity: Optional[float] = None
    micronutrients: Optional[Dict[str, Any]] = None
    is_lab_tested: Optional[bool] = None
    test_date: Optional[str] = None
    soil_health_card_number: Optional[str] = None
    testing_lab_name: Optional[str] = None


class SoilProfile(SoilProfileBase):
    soil_id: str
    plot_id: str
    farmer_id: str
    updated_at: str


class IrrigationProfileBase(BaseModel):
    primary_source: str = Field(default="Borewell", description="Primary source: Borewell, Canal, River, Pond, Rain")
    secondary_source: Optional[str] = None
    irrigation_method: IrrigationMethod = IrrigationMethod.DRIP
    water_availability: WaterAvailability = WaterAvailability.ADEQUATE
    pumping_source: PumpingSource = PumpingSource.ELECTRIC_GRID
    notes: Optional[str] = None


class IrrigationProfileCreate(IrrigationProfileBase):
    pass


class IrrigationProfileUpdate(BaseModel):
    primary_source: Optional[str] = None
    secondary_source: Optional[str] = None
    irrigation_method: Optional[IrrigationMethod] = None
    water_availability: Optional[WaterAvailability] = None
    pumping_source: Optional[PumpingSource] = None
    notes: Optional[str] = None


class IrrigationProfile(IrrigationProfileBase):
    irrigation_id: str
    plot_id: str
    farmer_id: str
    updated_at: str


# --- Farm Activity & Calendar Schemas ---

class FarmActivityBase(BaseModel):
    activity_type: ActivityType
    activity_title: str = Field(..., min_length=2, max_length=150)
    description: Optional[str] = None
    scheduled_date: str = Field(..., description="Target scheduled date (YYYY-MM-DD)")
    stage_tag: Optional[str] = Field(default=None, description="Linked crop growth stage tag")
    cost_incurred: Optional[float] = Field(default=None, ge=0.0)
    notes: Optional[str] = None


class FarmActivityCreate(FarmActivityBase):
    pass


class FarmActivityUpdate(BaseModel):
    activity_title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None
    completed_date: Optional[str] = None
    status: Optional[ActivityStatus] = None
    stage_tag: Optional[str] = None
    cost_incurred: Optional[float] = None
    notes: Optional[str] = None


class FarmActivity(FarmActivityBase):
    activity_id: str
    farm_id: str
    plot_id: Optional[str] = None
    cycle_id: Optional[str] = None
    farmer_id: str
    completed_date: Optional[str] = None
    status: ActivityStatus = ActivityStatus.PLANNED
    created_at: str
    updated_at: str
    version_number: int = 1


# --- Unified Farm Context Schema ---

class FarmContext(BaseModel):
    """Complete contextual representation of the farmer, their active farm, plot, and active crop."""
    is_authenticated: bool = False
    farmer: Optional[FarmerProfile] = None
    active_farm: Optional[Farm] = None
    all_farms_count: int = 0
    active_plots: List[Plot] = Field(default_factory=list)
    current_cycle: Optional[CropCycle] = None
    all_active_cycles: List[CropCycle] = Field(default_factory=list)
    soil_profile: Optional[SoilProfile] = None
    irrigation_profile: Optional[IrrigationProfile] = None
    due_activities: List[FarmActivity] = Field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None

    @property
    def has_active_farm(self) -> bool:
        return self.active_farm is not None

    @property
    def has_active_crop(self) -> bool:
        return self.current_cycle is not None

