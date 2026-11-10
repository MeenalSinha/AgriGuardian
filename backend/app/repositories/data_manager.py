"""Advisory persistence and repository layer."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("DataManager")

COLUMNS = [
    "timestamp",
    "farmer_name",
    "crop",
    "location",
    "symptoms",
    "risk",
    "advisory_hi",
    "advisory_en",
    "steps",
    "weather_summary",
    "market_summary",
    "field_visit_required",
    "status",
    "feedback",
    "language",
    "phone_number",
    "image_analysis",
]


class DataManager:
    """Manages CSV persistence for farmer advisories and case histories."""

    def __init__(self, csv_path: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.csv_path = csv_path or os.path.join(self.settings.data_dir, "advisories.csv")
        self.ensure_csv_exists()

    def ensure_csv_exists(self) -> None:
        """Create empty CSV file with proper headers if not already existing."""
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=COLUMNS)
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Initialized advisory CSV at {self.csv_path}")

    def save_advisory(self, data: Dict[str, Any]) -> bool:
        """Persist advisory record to CSV."""
        try:
            steps = data.get("three_step_plan", [])
            steps_json = json.dumps(steps) if isinstance(steps, list) else str(steps)

            img_analysis = data.get("image_analysis")
            img_json = json.dumps(img_analysis) if isinstance(img_analysis, dict) else "{}"

            record = {
                "timestamp": datetime.now().isoformat(),
                "farmer_name": data.get("farmer_name", "Farmer"),
                "crop": data.get("crop", "General"),
                "location": data.get("location", "India"),
                "symptoms": data.get("symptoms", ""),
                "risk": data.get("risk", "Low"),
                "advisory_hi": data.get("advisory_hi", ""),
                "advisory_en": data.get("advisory_en", ""),
                "steps": steps_json,
                "weather_summary": data.get("weather_summary", ""),
                "market_summary": data.get("market_summary", ""),
                "field_visit_required": data.get("field_visit_required", False),
                "status": "pending",
                "feedback": "",
                "language": data.get("language", "hi"),
                "phone_number": data.get("phone_number", ""),
                "image_analysis": img_json,
            }

            self.ensure_csv_exists()
            df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
            new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            new_df.to_csv(self.csv_path, index=False)
            logger.info(f"Advisory saved for {record['farmer_name']} ({record['crop']})")
            return True
        except Exception as e:
            logger.error(f"Failed to save advisory to {self.csv_path}: {e}", exc_info=True)
            return False

    def get_advisories(self) -> pd.DataFrame:
        """Load all saved advisories."""
        try:
            self.ensure_csv_exists()
            return pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        except Exception as e:
            logger.error(f"Failed to read advisories from {self.csv_path}: {e}")
            return pd.DataFrame(columns=COLUMNS)

    def update_status(self, index: int, status: str, feedback: str = "") -> bool:
        """Update review status or expert feedback for a specific advisory row."""
        try:
            df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
            if 0 <= index < len(df):
                df = df.astype(object)
                df.loc[index, "status"] = str(status)
                if feedback:
                    df.loc[index, "feedback"] = str(feedback)
                df.to_csv(self.csv_path, index=False)
                logger.info(f"Updated status for advisory #{index} -> {status}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update status for advisory #{index}: {e}")
            return False

