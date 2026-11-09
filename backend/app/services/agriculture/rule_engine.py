"""Dynamic Rule Engine for agricultural symptom and weather risk assessment."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger("RuleEngine")


class DynamicRuleEngine:
    """Evaluates crop symptoms and environmental triggers against expert agricultural rules."""

    def __init__(self, rules_file: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.rules_file = rules_file or os.path.join(self.settings.config_dir, "rules.json")
        self.rules: List[Dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        """Load rules from configuration file or create default rules."""
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.rules = data.get("rules", [])
                logger.info(f"Loaded {len(self.rules)} rules from {self.rules_file}")
            except Exception as e:
                logger.error(f"Error loading rules from {self.rules_file}: {e}")
                self._load_fallback_rules()
        else:
            self._create_default_rules()

    def _create_default_rules(self) -> None:
        """Create baseline agricultural diagnostic rules."""
        Path(self.rules_file).parent.mkdir(parents=True, exist_ok=True)
        default_rules = [
            {
                "id": "FUNGAL_001",
                "name": "Fungal Risk - Yellowing + Rain",
                "category": "disease",
                "risk_level": "High",
                "conditions": {
                    "symptoms_keywords": ["yellow", "yellowing"],
                    "weather_conditions": {"rain_recent": True},
                },
                "suggestion": "Possible fungal infection due to excess moisture. Improve soil drainage and consider bio-fungicide.",
                "active": True,
            },
            {
                "id": "WATER_001",
                "name": "Water Stress - Wilting + High Temp",
                "category": "environmental",
                "risk_level": "High",
                "conditions": {
                    "symptoms_keywords": ["wilt", "wilting", "drooping"],
                    "weather_conditions": {"temp_high": True},
                },
                "suggestion": "Severe water stress detected. Irrigate in early morning or evening and apply organic mulch.",
                "active": True,
            },
            {
                "id": "DISEASE_001",
                "name": "Leaf Spot Disease",
                "category": "disease",
                "risk_level": "Medium",
                "conditions": {
                    "symptoms_keywords": ["spot", "spots"],
                    "symptoms_keywords_additional": ["brown", "black"],
                },
                "suggestion": "Foliar leaf spot suspected. Remove severely affected leaves and avoid overhead sprinkler watering.",
                "active": True,
            },
            {
                "id": "PEST_001",
                "name": "Pest Infestation",
                "category": "pest",
                "risk_level": "Medium",
                "conditions": {
                    "symptoms_keywords": ["pest", "insect", "holes", "chewed", "bugs"],
                },
                "suggestion": "Insect pest infestation detected. Spray 5% neem seed kernel extract (NSKE) as organic repellent.",
                "active": True,
            },
        ]

        try:
            with open(self.rules_file, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "last_updated": datetime.now().isoformat(), "rules": default_rules}, f, indent=2)
            self.rules = default_rules
            logger.info(f"Initialized default rules file at {self.rules_file}")
        except Exception as e:
            logger.error(f"Failed to create default rules file: {e}")
            self.rules = default_rules

    def _load_fallback_rules(self) -> None:
        """Load in-memory fallback rules when file I/O fails."""
        self.rules = [
            {
                "id": "FALLBACK_001",
                "name": "General Moisture Stress",
                "category": "environmental",
                "risk_level": "Medium",
                "conditions": {"symptoms_keywords": ["dry", "wilting"]},
                "suggestion": "Check soil moisture at root depth before irrigating.",
                "active": True,
            }
        ]

    def evaluate(self, symptoms: str, weather: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate symptoms against active rules and weather context."""
        symptoms_lower = symptoms.lower() if symptoms else ""
        triggered_rules = []
        highest_risk = "Low"

        risk_priority = {"Low": 1, "Medium": 2, "High": 3}

        for rule in self.rules:
            if not rule.get("active", True):
                continue

            conditions = rule.get("conditions", {})
            keywords = conditions.get("symptoms_keywords", [])
            add_keywords = conditions.get("symptoms_keywords_additional", [])
            weather_conds = conditions.get("weather_conditions", {})

            # Check primary keyword match
            keyword_match = any(kw.lower() in symptoms_lower for kw in keywords) if keywords else True

            # Check secondary keyword match if required
            add_match = any(kw.lower() in symptoms_lower for kw in add_keywords) if add_keywords else True

            # Check weather condition match
            weather_match = True
            for cond_key, cond_val in weather_conds.items():
                if weather.get(cond_key) != cond_val:
                    weather_match = False
                    break

            if keyword_match and add_match and weather_match:
                triggered_rules.append(rule)
                rule_risk = rule.get("risk_level", "Low")
                if risk_priority.get(rule_risk, 1) > risk_priority.get(highest_risk, 1):
                    highest_risk = rule_risk

        return {
            "risk": highest_risk,
            "triggered_rules": triggered_rules,
            "rule_count": len(triggered_rules),
        }

