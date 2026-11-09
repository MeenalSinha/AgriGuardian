"""LLM Agricultural Advisory Service with evidence grounding, safety guardrails, and robust fallbacks."""

import json
import os
from typing import Any, Dict, List, Optional
import httpx
from openai import OpenAI
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import LLMAPIError
from backend.app.core.logging import get_logger
from backend.app.schemas.evidence import EvidenceExcerpt
from backend.app.schemas.knowledge import AtomicFact

logger = get_logger("LLMAdvisor")

SYSTEM_PROMPT = """You are an expert, empathetic agricultural advisor helping Indian farmers.

IMPORTANT AGRICULTURAL SAFETY RULES:
1. Provide practical, sustainable agricultural guidance (cultural, biological, and agronomic).
2. DO NOT prescribe precise chemical dosages or hazardous pesticide mixtures without expert supervision.
3. If risk is High or uncertain, clearly advise the farmer to consult a local Krishi Vigyan Kendra (KVK) or Agriculture Extension Officer.
4. Keep advice culturally appropriate, encouraging, and easy to understand.
5. Always return your response in strict JSON format.
"""

GROUNDED_SYSTEM_PROMPT = """You are an expert, trustworthy agricultural advisor helping Indian farmers.
Your advice must be strictly grounded in the verified agricultural evidence provided.

CRITICAL SAFETY & GROUNDING RULES:
1. ONLY recommend practices, cultural sanitation, or bio-controls that are supported by the provided evidence facts.
2. DO NOT hallucinate, invent, or prescribe chemical dosages, unverified brand names, or off-label pesticides.
3. Explicitly cite the issuing authoritative organization (e.g., ICAR-IIHR, CIBRC, IMD, TNAU) for key recommendations.
4. If risk is High, disease stage is Severe, or information is missing, advise the farmer to consult their local Krishi Vigyan Kendra (KVK).
5. Deliver empathetic, clear advice suitable for Indian farmers.
6. Always return your response in strict JSON format.
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
        retrieved_facts: Optional[List[Any]] = None,
        evidence_excerpts: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured agricultural advisory with safety guardrails."""
        if retrieved_facts or evidence_excerpts:
            return self.generate_grounded_advisory(
                crop=crop,
                farmer_query=symptoms,
                weather=weather,
                market=market,
                retrieved_facts=retrieved_facts or [],
                evidence_excerpts=evidence_excerpts or [],
                rule_results=rule_results,
                farmer_name=farmer_name,
                conversation_history=conversation_history,
            )

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

    def generate_grounded_advisory(
        self,
        crop: str,
        farmer_query: str,
        weather: Dict[str, Any],
        market: Dict[str, Any],
        retrieved_facts: List[Any],
        evidence_excerpts: List[Any],
        rule_results: Optional[Dict[str, Any]] = None,
        farmer_name: str = "Farmer",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        language: str = "hi",
    ) -> Dict[str, Any]:
        """Generate strictly grounded advisory referencing verified facts and evidence."""
        rules = rule_results or {}
        evidence_bullets = []
        for ev in evidence_excerpts:
            if isinstance(ev, dict):
                org = ev.get("organization", "Agricultural Authority")
                text = ev.get("excerpt_text", "")
                ev_id = ev.get("evidence_id", "")
            else:
                org = getattr(ev, "organization", "Agricultural Authority")
                text = getattr(ev, "excerpt_text", "")
                ev_id = getattr(ev, "evidence_id", "")
            evidence_bullets.append(f"[{ev_id} | {org}] {text}")

        facts_bullets = []
        for f in retrieved_facts:
            if isinstance(f, dict):
                pred = f.get("predicate", "")
                val = f.get("value", "")
            else:
                pred = getattr(f, "predicate", "")
                val = getattr(f, "value", "")
            facts_bullets.append(f"- ({pred}): {val}")

        evidence_str = "\n".join(evidence_bullets) if evidence_bullets else "No specific document excerpts available."
        facts_str = "\n".join(facts_bullets) if facts_bullets else "Standard agronomic practices apply."

        context = f"""Farmer: {farmer_name}
Crop: {crop}
Query / Symptoms: {farmer_query}
Weather: {weather.get('summary', 'Normal')}
Market: {market.get('summary', 'Standard')}
Rule Risk: {rules.get('risk', 'Medium')}

VERIFIED FACTS:
{facts_str}

STATUTORY & SCIENTIFIC EVIDENCE EXCERPTS:
{evidence_str}
"""

        user_prompt = f"""INPUT INFORMATION:
{context}

TASK: Generate a strictly evidence-grounded agricultural advisory.
Every recommended action MUST cite the source organization (e.g., ICAR-IIHR, CIBRC, IMD, TNAU) or trace to the facts above.
DO NOT fabricate chemical dosages or unapproved pesticides.

Structure your JSON response exactly as follows:
{{
  "risk": "Low" | "Medium" | "High",
  "advisory_hi": "2-3 simple, compassionate sentences in Hindi explaining the situation and practical advice grounded in facts",
  "advisory_en": "2-3 clear sentences in English with practical agricultural advice grounded in facts",
  "three_step_plan": [
    "Immediate Step 1 (Cultural / Sanitation / Drainage)",
    "Corrective Step 2 (Biological / Protectant as authorized)",
    "Follow-up / Preventive Step 3 (Monitoring / KVK visit)"
  ],
  "justification": "One clear sentence referencing the evidence source(s) and environmental rationale",
  "field_visit_required": true | false
}}
"""

        if not self.client:
            logger.info("OpenAI client offline; generating evidence-grounded fallback advisory.")
            return self._generate_grounded_fallback(crop, farmer_query, retrieved_facts, evidence_excerpts, rules)

        messages = [{"role": "system", "content": GROUNDED_SYSTEM_PROMPT}]

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
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            result = json.loads(raw_text)
            logger.info(f"Grounded LLM advisory successfully generated for {farmer_name} ({crop})")
            return result

        except Exception as e:
            logger.error(f"Grounded LLM call failed ({e}); falling back to deterministic evidence advisory.", exc_info=True)
            return self._generate_grounded_fallback(crop, farmer_query, retrieved_facts, evidence_excerpts, rules)

    def _generate_grounded_fallback(
        self,
        crop: str,
        query: str,
        retrieved_facts: List[Any],
        evidence_excerpts: List[Any],
        rule_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate deterministic, evidence-grounded advisory from facts when LLM is offline."""
        risk = rule_results.get("risk", "Medium")
        steps: List[str] = []
        authorities = set()

        for ev in evidence_excerpts:
            org = ev.get("organization") if isinstance(ev, dict) else getattr(ev, "organization", None)
            if org:
                authorities.add(org)

        # Build steps directly from facts
        cultural_actions = []
        bio_actions = []
        precaution_actions = []

        for f in retrieved_facts:
            pred = f.get("predicate", "") if isinstance(f, dict) else getattr(f, "predicate", "")
            val = f.get("value", "") if isinstance(f, dict) else getattr(f, "value", "")

            if "cultural" in pred or "sanitation" in pred:
                cultural_actions.append(val)
            elif "biological" in pred or "organic" in pred:
                bio_actions.append(val)
            elif "chemical" in pred or "irrigation" in pred:
                precaution_actions.append(val)

        if cultural_actions:
            steps.append(f"Cultural Practice: {cultural_actions[0]}")
        else:
            steps.append(f"Inspect {crop} foliage daily; remove infected leaves and debris to prevent spore spread.")

        if bio_actions:
            steps.append(f"Biological Control: {bio_actions[0]}")
        elif precaution_actions:
            steps.append(f"Protective Measure: {precaution_actions[0]}")
        else:
            steps.append("Maintain optimal field drainage and avoid overhead sprinkler irrigation.")

        if precaution_actions and len(steps) < 3:
            steps.append(f"Guideline: {precaution_actions[0]}")
        else:
            steps.append("For severe or persistent symptoms, consult the local Krishi Vigyan Kendra (KVK) for on-site diagnosis.")

        org_cite = ", ".join(sorted(authorities)) if authorities else "ICAR / CIBRC guidelines"

        return {
            "risk": risk,
            "advisory_hi": f"आपकी {crop} फसल के लिए {org_cite} के अनुशंसित दिशानिर्देशों के अनुसार सुरक्षित उपाय अपनाएं। अधिक प्रकोप होने पर निकटतम केवीके (KVK) से संपर्क करें।",
            "advisory_en": f"Evidence-based guidance for your {crop} based on verified protocols from {org_cite}. Implement sanitation and consult local KVK if symptoms persist.",
            "three_step_plan": steps[:3],
            "justification": f"Grounded in verified statutory and agronomic guidelines from {org_cite}.",
            "field_visit_required": (risk == "High"),
        }

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


