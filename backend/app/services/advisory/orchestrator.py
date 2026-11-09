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
from backend.app.services.agriculture.rule_engine import DynamicRuleEngine
from backend.app.services.analytics.dashboard import EnhancedDashboardAnalytics
from backend.app.services.reporting.pdf_generator import PDFReportGenerator

logger = get_logger("Orchestrator")


class AgriGuardianSystem:
    """Core AgriGuardian application orchestrator."""

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None):
        self.settings = settings or get_settings()
        self.rule_engine = DynamicRuleEngine(settings=self.settings)
        self.weather_service = WeatherService(settings=self.settings)
        self.market_service = MarketService(settings=self.settings)
        self.disease_detector = PlantDiseaseDetector(settings=self.settings)
        self.llm_advisor = LLMAdvisorService(settings=self.settings, api_key=api_key)
        self.data_manager = DataManager(settings=self.settings)
        self.pdf_generator = PDFReportGenerator(settings=self.settings)
        self.analytics = EnhancedDashboardAnalytics(self.data_manager)
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
    ) -> Dict[str, Any]:
        """Process complete advisory request combining multi-source intelligence."""
        logger.info(f"Processing request for farmer '{farmer_name}' (Crop: {crop}, Location: {location})")

        timing = {}
        overall_start = time.time()

        # Step 1: Weather data
        t0 = time.time()
        weather = self.weather_service.get_weather(lat=lat, lon=lon, location_name=location)
        timing["weather"] = time.time() - t0

        # Step 2: Market data
        t0 = time.time()
        market = self.market_service.get_market_data(crop=crop)
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
                severity_explanation = self.disease_detector.calculate_severity(
                    image_analysis["disease"],
                    image_analysis["confidence"],
                    weather.model_dump(),
                )

        # Step 4: Rule Engine
        t0 = time.time()
        rule_results = self.rule_engine.evaluate(effective_symptoms, weather.model_dump())
        timing["rules"] = time.time() - t0

        # Step 5: LLM Advisory
        t0 = time.time()
        advisory = self.llm_advisor.generate_advisory(
            crop=crop,
            symptoms=effective_symptoms,
            weather=weather.model_dump(),
            market=market.model_dump(),
            rule_results=rule_results,
            farmer_name=farmer_name,
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
            "field_visit_required": advisory.get("field_visit_required", False),
            "weather_summary": weather.summary,
            "market_summary": market.summary,
            "weather_details": weather.model_dump(),
            "market_details": market.model_dump(),
            "rule_results": rule_results,
            "image_analysis": image_analysis,
            "severity_explanation": severity_explanation,
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

