"""Agricultural Knowledge Retrieval Service grounded in ICAR agronomic standards."""

import json
import os
from typing import Any, Dict, List, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("KnowledgeService")


class KnowledgeService:
    """Provides grounded agronomic guidelines, crop cycles, and IPM disease recommendations."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.knowledge_path = os.path.join(self.settings.data_dir, "crop_knowledge", "crops.json")
        self.crops_db: Dict[str, Any] = {}
        self.load_knowledge()

    def load_knowledge(self) -> None:
        """Load crop knowledge profiles from JSON store."""
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.crops_db = data.get("crops", {})
                logger.info(f"Loaded knowledge profiles for {len(self.crops_db)} crops from {self.knowledge_path}")
            except Exception as e:
                logger.error(f"Failed to load crop knowledge from {self.knowledge_path}: {e}")
                self.crops_db = {}
        else:
            logger.warning(f"Crop knowledge file not found at {self.knowledge_path}")
            self.crops_db = {}

    def get_crop_profile(self, crop_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve agronomic profile for a specific crop."""
        key = crop_name.strip().lower() if crop_name else ""
        return self.crops_db.get(key)

    def find_relevant_guidance(self, crop_name: str, query: str) -> Dict[str, Any]:
        """Find matching disease symptoms, irrigation guidance, or cultural practices for query."""
        profile = self.get_crop_profile(crop_name)
        if not profile:
            return {
                "crop": crop_name,
                "found": False,
                "summary": "Standard agricultural monitoring recommended.",
            }

        q_lower = query.lower()
        matched_disease = None
        matched_irrigation = None

        # Check for irrigation queries
        if any(w in q_lower for w in ["water", "irrigate", "irrigation", "सिंचाई", "पानी"]):
            matched_irrigation = profile.get("critical_irrigation_stages", [])

        # Check for matching disease symptoms
        for d in profile.get("common_diseases", []):
            d_symptoms = d.get("symptoms", "").lower()
            d_name = d.get("name", "").lower()
            if any(term in q_lower for term in d_name.split()) or any(term in q_lower for term in d_symptoms.split()):
                matched_disease = d
                break

        return {
            "crop": profile.get("name", crop_name),
            "season": profile.get("season", "N/A"),
            "found": True,
            "soil_type": profile.get("soil_type", "N/A"),
            "matched_disease": matched_disease,
            "irrigation_stages": matched_irrigation,
        }
