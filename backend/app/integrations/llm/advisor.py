"""LLM Agricultural Advisory Service with safety guardrails and robust fallbacks."""

import json
import os
from typing import Any, Dict, List, Optional
import httpx
from openai import OpenAI
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import LLMAPIError
from backend.app.core.logging import get_logger

logger = get_logger("LLMAdvisor")

SYSTEM_PROMPT = """You are an expert, empathetic agricultural advisor helping Indian farmers.

IMPORTANT AGRICULTURAL SAFETY RULES:
1. Provide practical, sustainable agricultural guidance (cultural, biological, and agronomic).
2. DO NOT prescribe precise chemical dosages or hazardous pesticide mixtures without expert supervision.
3. If risk is High or uncertain, clearly advise the farmer to consult a local Krishi Vigyan Kendra (KVK) or Agriculture Extension Officer.
4. Keep advice culturally appropriate, encouraging, and easy to understand.
5. Always return your response in strict JSON format.
"""


class LLMAdvisorService:
    """Service generating contextual agricultural advisories via LLM with fallback safety."""

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None):
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.openai_api_key

        self.client: Optional[OpenAI] = None
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                # Use custom http_client to avoid openai 1.30 + httpx 0.28 'proxies' deprecation issue
                custom_http = httpx.Client(timeout=self.settings.llm_timeout_seconds)
                self.client = OpenAI(api_key=self.api_key, http_client=custom_http)
                logger.info("OpenAI client initialized successfully for LLM advisory.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Will rely on rule-based fallbacks.")

    def generate_advisory(
        self,
        crop: str,
        symptoms: str,
        weather: Dict[str, Any],
        market: Dict[str, Any],
        rule_results: Dict[str, Any],
        farmer_name: str = "Farmer",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate structured agricultural advisory with safety guardrails."""

        context = f"""Farmer: {farmer_name}
Crop: {crop}
Reported Symptoms / Question: {symptoms}
Weather Context: {weather.get('summary', 'Normal')}
Market Context: {market.get('summary', 'Standard')}
Rule-Engine Assessment: {rule_results.get('risk', 'Medium')}
Triggered Rules: {', '.join([r.get('name', '') for r in rule_results.get('triggered_rules', [])])}
"""

        user_prompt = f"""INPUT INFORMATION:
{context}

TASK: Generate a comprehensive agricultural advisory.
Structure your JSON response exactly as follows:
{{
  "risk": "Low" | "Medium" | "High",
  "advisory_hi": "2-3 simple, compassionate sentences in Hindi explaining the situation and practical advice",
  "advisory_en": "2-3 clear sentences in English with practical agricultural advice",
  "three_step_plan": ["Immediate Step 1", "Corrective Step 2", "Follow-up / Preventive Step 3"],
  "justification": "One clear sentence explaining the reasoning behind this assessment",
  "field_visit_required": true | false
}}
"""

        if not self.client:
            logger.info("OpenAI client is not configured; generating rule-based fallback advisory.")
            return self._generate_fallback_advisory(crop, symptoms, rule_results)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject conversation history if available
        if conversation_history:
            for msg in conversation_history[-4:]:
                messages.append(msg)

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )

            raw_text = response.choices[0].message.content.strip()

            # Parse JSON safely
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            result = json.loads(raw_text)
            logger.info(f"LLM advisory successfully generated for {farmer_name} ({crop})")
            return result

        except Exception as e:
            logger.error(f"LLM advisory call failed ({e}); invoking safe fallback advisory.", exc_info=True)
            return self._generate_fallback_advisory(crop, symptoms, rule_results)

    def _generate_fallback_advisory(
        self, crop: str, symptoms: str, rule_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate safe, rule-derived agricultural advisory when LLM is offline."""
        risk = rule_results.get("risk", "Medium")
        triggered = rule_results.get("triggered_rules", [])

        if triggered:
            suggestions = [r.get("suggestion", "") for r in triggered[:3] if r.get("suggestion")]
        else:
            suggestions = [
                f"Inspect your {crop} crop daily in the early morning for any symptom spread.",
                "Ensure proper field drainage and avoid over-irrigation during humid periods.",
                "Consult the nearest Krishi Vigyan Kendra (KVK) officer if symptoms persist.",
            ]

        # Ensure we always have 3 steps
        while len(suggestions) < 3:
            suggestions.append("Maintain good farm sanitation and remove diseased plant residues.")

        return {
            "risk": risk,
            "advisory_hi": f"आपकी {crop} फसल में '{symptoms}' के लक्षण देखे गए हैं। कृपया खेत की नियमित निगरानी रखें और नमी को नियंत्रित करें।",
            "advisory_en": f"Your {crop} crop shows symptoms of '{symptoms}'. Please maintain balanced moisture and inspect the field closely.",
            "three_step_plan": suggestions,
            "justification": f"Assessed as {risk} risk based on symptom pattern analysis and current weather factors.",
            "field_visit_required": (risk == "High"),
        }

