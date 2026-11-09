"""Schemas for farm location management and spatial precision."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class LocationPrecision(str, Enum):
    """Categorized spatial accuracy level for agro-meteorological modeling."""
    EXACT_GPS = "exact_gps"
    MANUAL_PIN = "manual_pin"
    VILLAGE_CENTROID = "village_centroid"
    DISTRICT_CENTROID = "district_centroid"


# Curated agricultural districts database across major Indian states
PRESET_AGRICULTURAL_DISTRICTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Uttar Pradesh": {
        "Meerut": {"lat": 28.9845, "lon": 77.7064, "elevation": 219.0},
        "Varanasi": {"lat": 25.3176, "lon": 82.9739, "elevation": 81.0},
        "Agra": {"lat": 27.1767, "lon": 78.0081, "elevation": 171.0},
        "Bareilly": {"lat": 28.3670, "lon": 79.4304, "elevation": 168.0},
        "Lucknow": {"lat": 26.8467, "lon": 80.9462, "elevation": 123.0},
    },
    "Punjab": {
        "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "elevation": 244.0},
        "Amritsar": {"lat": 31.6340, "lon": 74.8723, "elevation": 234.0},
        "Patiala": {"lat": 30.3398, "lon": 76.3869, "elevation": 250.0},
        "Bathinda": {"lat": 30.2110, "lon": 74.9455, "elevation": 201.0},
    },
    "Haryana": {
        "Karnal": {"lat": 29.6857, "lon": 76.9905, "elevation": 253.0},
        "Hisar": {"lat": 29.1492, "lon": 75.7217, "elevation": 215.0},
        "Ambala": {"lat": 30.3782, "lon": 76.7767, "elevation": 264.0},
        "Sirsa": {"lat": 29.5349, "lon": 75.0298, "elevation": 205.0},
    },
    "Karnataka": {
        "Bengaluru Rural": {"lat": 13.2291, "lon": 77.5684, "elevation": 920.0},
        "Kolar": {"lat": 13.1367, "lon": 78.1291, "elevation": 822.0},
        "Belagavi": {"lat": 15.8497, "lon": 74.4977, "elevation": 762.0},
        "Mandya": {"lat": 12.5229, "lon": 76.8976, "elevation": 678.0},
        "Dharwad": {"lat": 15.4589, "lon": 75.0078, "elevation": 750.0},
    },
    "Maharashtra": {
        "Pune": {"lat": 18.5204, "lon": 73.8567, "elevation": 560.0},
        "Nashik": {"lat": 19.9975, "lon": 73.7898, "elevation": 600.0},
        "Nagpur": {"lat": 21.1458, "lon": 79.0882, "elevation": 310.0},
        "Solapur": {"lat": 17.6599, "lon": 75.9064, "elevation": 458.0},
        "Aurangabad": {"lat": 19.8762, "lon": 75.3433, "elevation": 568.0},
    },
    "Tamil Nadu": {
        "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "elevation": 411.0},
        "Madurai": {"lat": 9.9252, "lon": 78.1198, "elevation": 101.0},
        "Thanjavur": {"lat": 10.7870, "lon": 79.1378, "elevation": 59.0},
        "Salem": {"lat": 11.6643, "lon": 78.1460, "elevation": 278.0},
    },
    "Andhra Pradesh": {
        "Guntur": {"lat": 16.3067, "lon": 80.4365, "elevation": 33.0},
        "Krishna": {"lat": 16.1875, "lon": 81.1389, "elevation": 14.0},
        "Kurnool": {"lat": 15.8281, "lon": 78.0373, "elevation": 273.0},
        "Anantapur": {"lat": 14.6819, "lon": 77.6006, "elevation": 335.0},
    },
    "Madhya Pradesh": {
        "Indore": {"lat": 22.7196, "lon": 75.8577, "elevation": 553.0},
        "Bhopal": {"lat": 23.2599, "lon": 77.4126, "elevation": 527.0},
        "Ujjain": {"lat": 23.1765, "lon": 75.7885, "elevation": 494.0},
        "Jabalpur": {"lat": 23.1815, "lon": 79.9864, "elevation": 411.0},
    },
    "Rajasthan": {
        "Jaipur": {"lat": 26.9124, "lon": 75.7873, "elevation": 431.0},
        "Kota": {"lat": 25.2138, "lon": 75.8648, "elevation": 271.0},
        "Ganganagar": {"lat": 29.9038, "lon": 73.8772, "elevation": 178.0},
        "Alwar": {"lat": 27.5530, "lon": 76.6346, "elevation": 271.0},
    },
    "Bihar": {
        "Patna": {"lat": 25.5941, "lon": 85.1376, "elevation": 53.0},
        "Muzaffarpur": {"lat": 26.1209, "lon": 85.3647, "elevation": 60.0},
        "Bhagalpur": {"lat": 25.2425, "lon": 86.9842, "elevation": 52.0},
        "Gaya": {"lat": 24.7914, "lon": 85.0002, "elevation": 111.0},
    },
    "West Bengal": {
        "Nadia": {"lat": 23.4710, "lon": 88.5565, "elevation": 14.0},
        "Bardhaman": {"lat": 23.2324, "lon": 87.8615, "elevation": 30.0},
        "Hooghly": {"lat": 22.9083, "lon": 88.3967, "elevation": 17.0},
    },
    "Gujarat": {
        "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "elevation": 53.0},
        "Rajkot": {"lat": 22.3039, "lon": 70.8022, "elevation": 128.0},
        "Surat": {"lat": 21.1702, "lon": 72.8311, "elevation": 13.0},
        "Vadodara": {"lat": 22.3072, "lon": 73.1812, "elevation": 39.0},
    },
}


class LocationRecord(BaseModel):
    """Schema for validated farm location context."""
    location_id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'LOC-MEERUT-01')")
    display_name: str = Field(..., description="Human-readable location label (e.g. 'Meerut, Uttar Pradesh')")
    state: str = Field(..., description="Indian state name")
    district: str = Field(..., description="District name")
    subdistrict: Optional[str] = Field(None, description="Tehsil or sub-district block")
    village: Optional[str] = Field(None, description="Village or local revenue town")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate in decimal degrees")
    elevation_m: Optional[float] = Field(None, description="Elevation above mean sea level in meters")
    timezone: str = Field("Asia/Kolkata", description="IANA Time Zone identifier")
    precision: str = Field(
        "district_centroid",
        description="Spatial precision: 'exact_gps', 'village_centroid', 'district_centroid', 'manual_pin'"
    )
    source: str = Field(
        "manual",
        description="Location acquisition source: 'gps', 'manual', 'map_click', 'preset'"
    )
    is_confirmed: bool = Field(True, description="Whether the farmer confirmed this location")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude {v} is outside valid range (-90.0 to 90.0).")
        return round(v, 4)

    @field_validator("longitude")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude {v} is outside valid range (-180.0 to 180.0).")
        return round(v, 4)

    @property
    def is_approximate(self) -> bool:
        """Returns True if the location is a regional or district centroid rather than exact field plot."""
        return self.precision in ("district_centroid", "village_centroid")

    @property
    def precision_label(self) -> str:
        """Human-readable explanation of spatial accuracy."""
        if self.precision == "exact_gps":
            return "Exact GPS Coordinate (Field Level)"
        elif self.precision == "manual_pin":
            return "Map Pin Selection (Farm Vicinity)"
        elif self.precision == "village_centroid":
            return "Village Level (~1-3 km approximate)"
        return "District Centroid (Regional Forecast)"
