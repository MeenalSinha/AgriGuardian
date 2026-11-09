"""AgriGuardian System Orchestrator coordinating all agricultural reasoning services."""

import time
from datetime import datetime
from typing import Any, Dict, Optional
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.integrations.llm.advisor import LLMAdvisorService
from backend.app.integrations.market.market_service import MarketService
from backend.app.integrations.weather.open_meteo import WeatherService
from backend.app.repositories.data_manager import DataManager
from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector
from backend.app.services.agriculture.recommendation_engine import RecommendationEngine
from backend.app.services.agriculture.rule_engine import DynamicRuleEngine
from backend.app.services.analytics.dashboard import EnhancedDashboardAnalytics
from backend.app.services.farm.farm_context_service import FarmContextService
from backend.app.services.reporting.pdf_generator import PDFReportGenerator
from backend.app.services.weather.weather_intelligence import WeatherIntelligenceService

logger = get_logger("Orchestrator")


class AgriGuardianSystem:
    """Core AgriGuardian application orchestrator."""

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None):
        self.settings = settings or get_settings()
        self.rule_engine = DynamicRuleEngine(settings=self.settings)
        self.weather_service = WeatherService(settings=self.settings)
        self.weather_intel = WeatherIntelligenceService(settings=self.settings, weather_service=self.weather_service)
        self.market_service = MarketService(settings=self.settings)
        self.disease_detector = PlantDiseaseDetector(settings=self.settings)
        self.recommendation_engine = RecommendationEngine(settings=self.settings)
        self.llm_advisor = LLMAdvisorService(settings=self.settings, api_key=api_key)
        self.data_manager = DataManager(settings=self.settings)
        self.pdf_generator = PDFReportGenerator(settings=self.settings)
        self.analytics = EnhancedDashboardAnalytics(self.data_manager)
        self.farm_context = FarmContextService()
        logger.info("AgriGuardianSystem initialized successfully.")

    def process_request(
        self,
        farmer_name: str,
        crop: str,
        location: str,
        symptoms: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        image_path: Optional[str] = None,
        language: str = "hi",
        phone_number: Optional[str] = None,
        confidence_threshold: float = 0.3,
        farmer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process complete advisory request combining multi-source intelligence."""
        # Auto-enrich from farm context if farmer_id is provided
        farm_ctx = None
        if farmer_id:
            farm_ctx = self.farm_context.resolve_context(farmer_id=farmer_id)
            if farm_ctx and farm_ctx.active_farm:
                lat = lat or farm_ctx.active_farm.latitude
                lon = lon or farm_ctx.active_farm.longitude
                if not location or location == "Delhi, India":
                    location = f"{farm_ctx.active_farm.district}, {farm_ctx.active_farm.state}"
            if farm_ctx and farm_ctx.current_cycle:
                crop = crop or farm_ctx.current_cycle.crop_name

        logger.info(f"Processing request for farmer '{farmer_name}' (Crop: {crop}, Location: {location})")

        timing = {}
        overall_start = time.time()

        # Step 1: Weather Intelligence & Crop Climate Risk Assessment
        t0 = time.time()
        loc = None
        if lat is not None and lon is not None:
            loc = self.weather_intel.location_manager.resolve_coordinates(
                latitude=lat, longitude=lon, source="manual_pin"
            )
        elif location:
            for state_name, dists in self.weather_intel.location_manager.presets.items():
                for dist_name in dists:
                    if dist_name.lower() in location.lower():
                        loc = self.weather_intel.location_manager.resolve_preset_district(state_name, dist_name)
                        break
                if loc:
                    break
        if not loc:
            loc = self.weather_intel.location_manager.get_default_location()

        weather_intel = self.weather_intel.get_weather_intelligence(
            location=loc,
            crop=crop,
            forecast_days=7,
        )
        weather_report = weather_intel["report"]
        weather_risk = weather_intel["risk_assessment"]
        weather_alerts = weather_intel["active_alerts"]
        weather = weather_report.to_legacy_context()
        timing["weather"] = time.time() - t0

        # Step 2: Market data
        t0 = time.time()
        market = self.market_service.get_market_data(crop=crop, location=location)
        timing["market"] = time.time() - t0

        # Step 3: Image analysis if provided
        image_analysis = None
        severity_explanation = None
        effective_symptoms = symptoms

        if image_path:
            t0 = time.time()
            image_analysis = self.disease_detector.predict(
                image_path,
                confidence_threshold=confidence_threshold,
                weather_context=weather.model_dump(),
            )
            timing["image_ml"] = time.time() - t0

            if image_analysis.get("disease") not in ("Unable to detect", "Invalid or Unreadable Image", "Uncertain / Unrecognized Foliar Symptom"):
                effective_symptoms = f"{symptoms}. Image analysis reveals: {image_analysis['disease']} (Severity: {image_analysis.get('severity', 'Unknown')})"
                diag_data = image_analysis.get("diagnosis_result")
                if diag_data and "severity_assessment" in diag_data:
                    sev = diag_data["severity_assessment"]
                    severity_explanation = {
                        "score": sev.get("severity_score_percent", 0.0),
                        "final_score": sev.get("severity_score_percent", 0.0),
                        "level": sev.get("severity_stage", "Unknown"),
                        "severity": sev.get("severity_stage", "Unknown"),
                        "factors": sev.get("environmental_warnings", []),
                        "weather_factors": sev.get("environmental_warnings", []),
                        "weather_multiplier": sev.get("environmental_risk_multiplier", 1.0),
                        "clinical_summary": sev.get("clinical_summary", ""),
                    }
                else:
                    severity_explanation = self.disease_detector.calculate_severity(
                        image_analysis["disease"],
                        image_analysis["confidence"],
                        weather.model_dump(),
                    )

        # Step 4: Rule Engine
        t0 = time.time()
        rule_results = self.rule_engine.evaluate(effective_symptoms, weather.model_dump())
        timing["rules"] = time.time() - t0

        # Step 5: Phase 3 Recommendation Engine (Evidence Retrieval, Applicability, Conflict Arbitration & Safety)
        t0 = time.time()
        diag_dict = image_analysis.get("diagnosis_result") if image_analysis else None
        weather_ctx_dict = {
            **weather.model_dump(),
            "active_alerts": [a.model_dump() for a in weather_alerts],
        }
        recommendation = self.recommendation_engine.generate_recommendation(
            query=effective_symptoms,
            crop=crop.lower(),
            language=language,
            state=None,
            weather_context=weather_ctx_dict,
            image_diagnosis=diag_dict,
        )
        timing["recommendation"] = time.time() - t0

        # Step 6: Grounded LLM Advisory
        t0 = time.time()
        retrieved_facts = [
            f for f, _ in self.recommendation_engine.retrieval_service.retrieve(
                query=effective_symptoms, crop=crop.lower(), top_k=4
            )
        ]
        advisory = self.llm_advisor.generate_advisory(
            crop=crop,
            symptoms=effective_symptoms,
            weather=weather.model_dump(),
            market=market.model_dump(),
            rule_results=rule_results,
            farmer_name=farmer_name,
            retrieved_facts=retrieved_facts,
            evidence_excerpts=recommendation.supporting_evidence,
        )
        timing["llm"] = time.time() - t0
        timing["total"] = time.time() - overall_start

        # Compile final result
        result = {
            "farmer_name": farmer_name,
            "crop": crop,
            "location": location,
            "symptoms": symptoms,
            "risk": advisory.get("risk", "Medium"),
            "advisory_hi": advisory.get("advisory_hi", ""),
            "advisory_en": advisory.get("advisory_en", ""),
            "three_step_plan": advisory.get("three_step_plan", []),
            "justification": advisory.get("justification", ""),
            "field_visit_required": advisory.get("field_visit_required", False) or recommendation.expert_referral_required,
            "weather_summary": weather.summary,
            "market_summary": market.summary,
            "weather_details": weather.model_dump(),
            "weather_report": weather_report.model_dump(),
            "weather_risk_assessment": weather_risk.model_dump(),
            "weather_alerts": [a.model_dump() for a in weather_alerts],
            "market_details": market.model_dump(),
            "rule_results": rule_results,
            "image_analysis": image_analysis,
            "severity_explanation": severity_explanation,
            "recommendation": recommendation.model_dump(),
            "supporting_evidence": [e.model_dump() for e in recommendation.supporting_evidence],
            "expert_referral_required": recommendation.expert_referral_required,
            "expert_referral_details": recommendation.expert_referral_details,
            "audio_file": None,
            "language": language,
            "phone_number": phone_number,
            "timestamp": datetime.now().isoformat(),
            "timing": timing,
        }

        # Step 6: Save to CSV repository
        self.data_manager.save_advisory(result)
        logger.info(f"Advisory completed for {farmer_name} with risk: {result['risk']} in {timing['total']:.2f}s")
        return result

    def generate_pdf_report(self, result: Dict[str, Any]) -> str:
        """Generate PDF report for advisory result."""
        return self.pdf_generator.generate_report(result)

