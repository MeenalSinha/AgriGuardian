"""High-level Weather Intelligence service integrating location, weather retrieval, risk evaluation, and alerts."""

from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.integrations.weather.open_meteo import WeatherService
from backend.app.schemas.location import LocationRecord
from backend.app.schemas.weather import WeatherReport
from backend.app.schemas.weather_alert import WeatherAlert, WeatherRiskAssessment
from backend.app.services.weather.alert_manager import AlertManager, alert_manager
from backend.app.services.weather.location_manager import LocationManager
from backend.app.services.weather.weather_risk_engine import WeatherRiskEngine

logger = get_logger("WeatherIntelligenceService")


class WeatherIntelligenceService:
    """Unified service for agricultural weather intelligence, risk assessment, and advisory alerting."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        weather_service: Optional[WeatherService] = None,
        risk_engine: Optional[WeatherRiskEngine] = None,
        loc_manager: Optional[LocationManager] = None,
        alt_manager: Optional[AlertManager] = None,
    ):
        self.settings = settings or get_settings()
        self.weather_service = weather_service or WeatherService(settings=self.settings)
        self.risk_engine = risk_engine or WeatherRiskEngine()
        self.location_manager = loc_manager or LocationManager()
        self.alert_manager = alt_manager or alert_manager

    def get_weather_intelligence(
        self,
        location: Optional[LocationRecord] = None,
        crop: Optional[str] = None,
        crop_stage: Optional[str] = None,
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """Generate comprehensive agricultural weather intelligence package.
        
        Args:
            location: Target location (falls back to default preset if omitted).
            crop: Optional crop focus ('tomato', 'wheat', etc.).
            crop_stage: Optional growth stage ('flowering', 'grain_filling', etc.).
            forecast_days: Forecast horizon in days (default: 7).
            
        Returns:
            Dictionary containing 'report', 'risk_assessment', 'active_alerts', and 'location'.
        """
        loc = location or self.location_manager.get_default_location()

        # 1. Fetch normalized weather report with historical reanalysis
        report = self.weather_service.get_weather_report(location=loc, forecast_days=forecast_days)

        # 2. Evaluate evidence-grounded agricultural risks
        risk_assessment = self.risk_engine.evaluate_risks(
            report=report,
            crop=crop,
            crop_stage=crop_stage,
        )

        # 3. Register generated alerts with AlertManager
        managed_alerts = self.alert_manager.register_alerts(risk_assessment.active_alerts)
        risk_assessment.active_alerts = managed_alerts

        return {
            "location": loc,
            "report": report,
            "risk_assessment": risk_assessment,
            "active_alerts": managed_alerts,
        }

