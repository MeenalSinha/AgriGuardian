"""Schemas for agricultural advisory requests, contextual factors, and outputs."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class WeatherContext(BaseModel):
    """Normalized weather conditions for agricultural reasoning."""
    temperature: float
    humidity: float
    precipitation: float
    rain_recent: bool = False
    temp_high: bool = False
    temp_very_high: bool = False
    humidity_high: bool = False
    forecast: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    is_live: bool = True


class MarketContext(BaseModel):
    """Market price information with source transparency."""
    crop: str
    current_price: float
    trend: str
    min_price: float
    max_price: float
    last_week_price: Optional[float] = None
    price_change: Optional[float] = None
    change_percent: Optional[float] = None
    forecast: Optional[str] = None
    demand: Optional[str] = None
    insight: Optional[str] = None
    summary: str
    source: str = "Indicative Reference Benchmark"
    is_live: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RuleEvaluationResult(BaseModel):
    """Result from the agricultural rule engine."""
    risk: str
    triggered_rules: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.8
    evaluation_time: float = 0.0


class ImageAnalysisResult(BaseModel):
    """Result of crop disease detection and explainability."""
    disease: str
    confidence: float
    top_3: List[Dict[str, Any]] = Field(default_factory=list)
    severity: str
    severity_score_percent: Optional[float] = None
    recommendation: str
    model_used: str = "MobileNetV3-Small"
    model_version: str = "2.1.0-prod"
    device: str = "cpu"
    inference_time: float = 0.0
    gradcam_image: Optional[str] = None
    gradcam_base64: Optional[str] = None
    is_simulated: bool = False
    diagnosis_result: Optional[Dict[str, Any]] = None


class AdvisoryResult(BaseModel):
    """Comprehensive agricultural advisory result."""
    farmer_name: str
    crop: str
    location: str
    symptoms: str
    risk: str
    advisory_hi: str
    advisory_en: str
    three_step_plan: List[str]
    justification: str
    field_visit_required: bool
    weather_summary: str
    market_summary: str
    weather_details: Optional[Dict[str, Any]] = None
    market_details: Optional[Dict[str, Any]] = None
    rule_results: Optional[Dict[str, Any]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    severity_explanation: Optional[Dict[str, Any]] = None
    audio_file: Optional[str] = None
    language: str = "hi"
    phone_number: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    timing: Dict[str, float] = Field(default_factory=dict)

