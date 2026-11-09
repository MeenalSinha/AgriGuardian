"""Soil Health and Irrigation Profile Service with ICAR NPK rating diagnostics."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import (
    IrrigationProfile,
    IrrigationProfileCreate,
    IrrigationProfileUpdate,
    SoilProfile,
    SoilProfileCreate,
    SoilProfileUpdate,
)

logger = get_logger("SoilIrrigationService")


class SoilIrrigationService:
    """Manages plot soil fertility profiles, laboratory Soil Health Card tests, and irrigation configurations."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    # =========================================================================
    # SOIL PROFILE MANAGEMENT & ICAR DIAGNOSTICS
    # =========================================================================

    def save_soil_profile(
        self,
        plot_id: str,
        farmer_id: str,
        profile_data: SoilProfileCreate,
    ) -> SoilProfile:
        """Create or update soil profile for a plot verifying tenant ownership."""
        plot = self.repo.get_plot(plot_id=plot_id, farmer_id=farmer_id)
        if not plot:
            raise ValueError(f"Plot '{plot_id}' not found or does not belong to farmer '{farmer_id}'.")

        soil_id = f"sol_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()

        soil = SoilProfile(
            soil_id=soil_id,
            plot_id=plot_id,
            farmer_id=farmer_id,
            soil_type=profile_data.soil_type,
            ph_level=profile_data.ph_level,
            organic_carbon_pct=profile_data.organic_carbon_pct,
            nitrogen_kg_ha=profile_data.nitrogen_kg_ha,
            phosphorus_kg_ha=profile_data.phosphorus_kg_ha,
            potassium_kg_ha=profile_data.potassium_kg_ha,
            electrical_conductivity=profile_data.electrical_conductivity,
            micronutrients=profile_data.micronutrients,
            is_lab_tested=profile_data.is_lab_tested,
            test_date=profile_data.test_date,
            soil_health_card_number=profile_data.soil_health_card_number,
            testing_lab_name=profile_data.testing_lab_name,
            updated_at=now_str,
        )

        saved = self.repo.save_soil_profile(soil)
        logger.info(f"Saved soil profile for plot {plot_id} (Lab tested: {soil.is_lab_tested})")
        return saved

    def get_soil_profile(self, plot_id: str, farmer_id: str) -> Optional[SoilProfile]:
        """Fetch soil profile for a plot verifying tenant ownership."""
        return self.repo.get_soil_profile_by_plot(plot_id=plot_id, farmer_id=farmer_id)

    def diagnose_soil_fertility(self, profile: SoilProfile) -> Dict[str, Any]:
        """Classify NPK, pH, and Organic Carbon against standard ICAR/Soil Health Card benchmarks."""
        diagnostics = {
            "ph_category": "Unknown",
            "organic_carbon_category": "Unknown",
            "nitrogen_category": "Unknown",
            "phosphorus_category": "Unknown",
            "potassium_category": "Unknown",
            "recommendations": [],
        }

        # pH assessment
        if profile.ph_level is not None:
            if profile.ph_level < 6.0:
                diagnostics["ph_category"] = "Strongly Acidic"
                diagnostics["recommendations"].append("Apply agricultural lime (calcium carbonate) to buffer acidic soil pH.")
            elif 6.0 <= profile.ph_level < 6.5:
                diagnostics["ph_category"] = "Slightly Acidic"
            elif 6.5 <= profile.ph_level <= 7.5:
                diagnostics["ph_category"] = "Neutral (Optimal for most field crops)"
            elif 7.5 < profile.ph_level <= 8.5:
                diagnostics["ph_category"] = "Moderately Alkaline"
            else:
                diagnostics["ph_category"] = "Strongly Alkaline"
                diagnostics["recommendations"].append("Apply agricultural gypsum to reduce soil sodicity and lower pH.")

        # Organic Carbon assessment (ICAR benchmark: < 0.5% Low, 0.5-0.75% Medium, > 0.75% High)
        if profile.organic_carbon_pct is not None:
            if profile.organic_carbon_pct < 0.5:
                diagnostics["organic_carbon_category"] = "Low (< 0.5%)"
                diagnostics["recommendations"].append("Incorporate Farmyard Manure (FYM) or green manure to boost soil organic carbon.")
            elif profile.organic_carbon_pct <= 0.75:
                diagnostics["organic_carbon_category"] = "Medium (0.5% - 0.75%)"
            else:
                diagnostics["organic_carbon_category"] = "High (> 0.75%)"

        # Nitrogen (kg/ha: < 280 Low, 280-560 Medium, > 560 High)
        if profile.nitrogen_kg_ha is not None:
            if profile.nitrogen_kg_ha < 280:
                diagnostics["nitrogen_category"] = "Low (< 280 kg/ha)"
                diagnostics["recommendations"].append("Increase basal nitrogen application and consider split urea top-dressing.")
            elif profile.nitrogen_kg_ha <= 560:
                diagnostics["nitrogen_category"] = "Medium (280 - 560 kg/ha)"
            else:
                diagnostics["nitrogen_category"] = "High (> 560 kg/ha)"
                diagnostics["recommendations"].append("Avoid excess nitrogen application to prevent lodging and foliar pest surges.")

        # Phosphorus (kg/ha P2O5: < 10 Low, 10-25 Medium, > 25 High)
        if profile.phosphorus_kg_ha is not None:
            if profile.phosphorus_kg_ha < 10:
                diagnostics["phosphorus_category"] = "Low (< 10 kg/ha)"
                diagnostics["recommendations"].append("Apply Single Super Phosphate (SSP) or DAP at basal sowing stage.")
            elif profile.phosphorus_kg_ha <= 25:
                diagnostics["phosphorus_category"] = "Medium (10 - 25 kg/ha)"
            else:
                diagnostics["phosphorus_category"] = "High (> 25 kg/ha)"

        # Potassium (kg/ha K2O: < 110 Low, 110-280 Medium, > 280 High)
        if profile.potassium_kg_ha is not None:
            if profile.potassium_kg_ha < 110:
                diagnostics["potassium_category"] = "Low (< 110 kg/ha)"
                diagnostics["recommendations"].append("Apply Muriate of Potash (MOP) to enhance stalk strength and drought resilience.")
            elif profile.potassium_kg_ha <= 280:
                diagnostics["potassium_category"] = "Medium (110 - 280 kg/ha)"
            else:
                diagnostics["potassium_category"] = "High (> 280 kg/ha)"

        return diagnostics

    # =========================================================================
    # IRRIGATION PROFILE MANAGEMENT
    # =========================================================================

    def save_irrigation_profile(
        self,
        plot_id: str,
        farmer_id: str,
        profile_data: IrrigationProfileCreate,
    ) -> IrrigationProfile:
        """Create or update irrigation profile for a plot verifying tenant ownership."""
        plot = self.repo.get_plot(plot_id=plot_id, farmer_id=farmer_id)
        if not plot:
            raise ValueError(f"Plot '{plot_id}' not found or does not belong to farmer '{farmer_id}'.")

        irrigation_id = f"irr_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()

        irrigation = IrrigationProfile(
            irrigation_id=irrigation_id,
            plot_id=plot_id,
            farmer_id=farmer_id,
            primary_source=profile_data.primary_source.strip(),
            secondary_source=profile_data.secondary_source.strip() if profile_data.secondary_source else None,
            irrigation_method=profile_data.irrigation_method,
            water_availability=profile_data.water_availability,
            pumping_source=profile_data.pumping_source,
            notes=profile_data.notes,
            updated_at=now_str,
        )

        saved = self.repo.save_irrigation_profile(irrigation)
        logger.info(f"Saved irrigation profile for plot {plot_id} (Method: {irrigation.irrigation_method.value})")
        return saved

    def get_irrigation_profile(self, plot_id: str, farmer_id: str) -> Optional[IrrigationProfile]:
        """Fetch irrigation profile for a plot verifying tenant ownership."""
        return self.repo.get_irrigation_profile_by_plot(plot_id=plot_id, farmer_id=farmer_id)

