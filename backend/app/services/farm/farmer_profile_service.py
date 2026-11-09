"""Farmer Profile Service managing identity attributes, preferences, and completion metrics."""

import uuid
from datetime import datetime
from typing import Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import FarmerProfile, FarmerProfileUpdate

logger = get_logger("FarmerProfileService")


class FarmerProfileService:
    """Manages farmer personal profile, literacy/voice preferences, and progressive onboarding score."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def get_profile(self, farmer_id: str) -> Optional[FarmerProfile]:
        """Fetch profile and compute dynamic completion percentage."""
        profile = self.repo.get_farmer_profile(farmer_id)
        if not profile:
            return None
        return self._enrich_completion(profile)

    def update_profile(self, farmer_id: str, update_data: FarmerProfileUpdate) -> Optional[FarmerProfile]:
        """Update profile fields and persist updated completion score."""
        profile = self.repo.get_farmer_profile(farmer_id)
        if not profile:
            return None

        updated_dict = profile.model_dump()
        for field, val in update_data.model_dump(exclude_unset=True).items():
            if val is not None:
                updated_dict[field] = val

        updated_dict["updated_at"] = datetime.now().isoformat()
        enriched = self._enrich_completion(FarmerProfile(**updated_dict))
        return self.repo.update_farmer_profile(enriched)

    def _enrich_completion(self, profile: FarmerProfile) -> FarmerProfile:
        """Calculate progressive profile completion percentage across profile, farm, plot, and crop."""
        score = 0.0
        # 1. Base personal attributes (max 40%)
        if profile.full_name:
            score += 15.0
        if profile.phone_number:
            score += 10.0
        if profile.preferred_language:
            score += 5.0
        if profile.experience_years is not None:
            score += 5.0
        if profile.soil_health_card_available or profile.kisan_credit_card_holder:
            score += 5.0

        # 2. Farm, plot, and crop presence (max 60%)
        farms = self.repo.get_farms_by_farmer(profile.farmer_id)
        if farms:
            score += 20.0
            plots = self.repo.get_plots_by_farmer(profile.farmer_id)
            if plots:
                score += 20.0
                # Check for active crop cycle
                cycles = self.repo.get_all_crop_cycles_by_farmer(profile.farmer_id, status="active")
                if cycles:
                    score += 20.0

        profile.profile_completion_pct = min(100.0, round(score, 1))
        return profile
