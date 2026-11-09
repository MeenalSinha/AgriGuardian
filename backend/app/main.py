"""AgriGuardian — FastAPI Headless REST Backend.

Provides programmatic access to multilingual voice intelligence,
crop diagnostics, weather context, and agricultural advisory services.
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.app.core.config import SUPPORTED_LANGUAGES, TIMBRE_V25_VOICES, get_settings
from backend.app.core.exceptions import GnaniAPIError, GnaniAuthError
from backend.app.core.logging import get_logger, setup_logging
from backend.app.integrations.gnani.audio_utils import save_temporary_audio
from backend.app.schemas.advisory import AdvisoryResult
from backend.app.schemas.conversation import ConversationSession, ConversationTurn
from backend.app.schemas.disease import DiagnosisResult
from backend.app.schemas.speech import (
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
    TranscriptionResult,
)
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import MarketTrendAnalysis, PriceForecastResult
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector
from backend.app.services.agriculture.knowledge_service import KnowledgeService
from backend.app.services.conversation.engine import ConversationEngine
from backend.app.services.market.market_analytics import MarketAnalyticsService
from backend.app.services.market.market_intelligence import MarketIntelligenceService
from backend.app.services.market.net_realization_calculator import NetRealizationCalculator
from backend.app.services.market.price_forecaster import PriceForecaster

# Initialize settings and structured logging
settings = get_settings()
setup_logging(log_level=settings.log_level, logs_dir=settings.logs_dir)
logger = get_logger("FastAPIServer")

# Initialize FastAPI app with OpenAPI metadata
app = FastAPI(
    title="AgriGuardian API",
    description="Multilingual Voice Intelligence & Agricultural Advisory API powered by Gnani AI.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services singletons
system = AgriGuardianSystem(settings=settings)
conversation_engine = ConversationEngine(system=system, settings=settings)
knowledge_service = KnowledgeService(settings=settings)
market_intel = MarketIntelligenceService()

# In-memory sessions store (can be backed by Redis / PostgreSQL in Phase 2)
sessions_store: Dict[str, ConversationSession] = {}


# ==============================================================================
# SCHEMAS FOR API REQUESTS
# ==============================================================================

class NetRealizationRequest(BaseModel):
    """Payload for calculating net farmgate realization."""
    market_id: str
    market_name: str
    commodity_name: str
    distance_km: float = Field(..., ge=0.0)
    quantity_quintals: float = Field(..., gt=0.0)
    reference_modal_price: float = Field(..., gt=0.0)
    transport_cost_override: Optional[float] = None
    handling_fee_override: Optional[float] = None
    transport_rate_per_km_qtl: Optional[float] = None
# ==============================================================================

class ConversationTurnRequest(BaseModel):
    """Payload for submitting a conversation turn."""
    session_id: Optional[str] = None
    farmer_name: str = "Farmer"
    user_query: str
    crop: Optional[str] = "Tomato"
    location: Optional[str] = "Delhi, India"
    language_code: str = "hi-IN"
    input_type: str = "text"
    image_path: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    synthesize_voice: bool = True


class AdvisoryGenerationRequest(BaseModel):
    """Payload for generating a comprehensive crop advisory."""
    farmer_name: str = "Rajesh Kumar"
    crop: str = "Wheat"
    location: str = "Meerut, UP"
    symptoms: str = "Yellowing leaves with spots"
    lat: Optional[float] = 28.6139
    lon: Optional[float] = 77.2090
    language: str = "hi"
    phone_number: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Payload for evidence-grounded recommendation generation."""
    query: str
    crop: str = "tomato"
    language: str = "hi-IN"
    state: Optional[str] = None
    crop_stage: Optional[str] = None
    days_after_sowing: Optional[int] = None
    weather_context: Optional[Dict[str, Any]] = None
    image_diagnosis: Optional[Dict[str, Any]] = None


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/", tags=["General"])
def root():
    """API Root greeting and system information."""
    return {
        "app_name": settings.app_name,
        "version": "2.0.0",
        "status": "online",
        "documentation": "/docs",
        "gnani_models": {
            "stt": settings.gnani_prisma_model,
            "tts": settings.gnani_timbre_model,
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """System health check and integration availability."""
    return {
        "status": "healthy",
        "gnani_configured": settings.is_gnani_configured,
        "llm_configured": settings.is_llm_configured,
        "open_meteo": "active",
        "storage": "available",
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
    }


@app.get("/api/v1/languages", tags=["Voice"])
def get_languages():
    """Retrieve verified Indian language catalog supported by Gnani AI."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "voices": TIMBRE_V25_VOICES,
    }


@app.post("/api/v1/voice/transcribe", response_model=TranscriptionResult, tags=["Voice"])
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language_code: str = Form("hi-IN"),
):
    """Transcribe farmer audio question using Gnani Prisma v2.5."""
    try:
        content = await audio_file.read()
        ext = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
        temp_audio_path = save_temporary_audio(content, prefix="api_upload", ext=ext)

        result = conversation_engine.transcribe_audio(
            audio_file_path=temp_audio_path,
            language_code=language_code,
        )
        return result
    except GnaniAuthError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"API transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/voice/synthesize", response_model=SpeechSynthesisResult, tags=["Voice"])
def synthesize_speech(request: SpeechSynthesisRequest):
    """Synthesize agricultural guidance into natural speech using Gnani Timbre v2.5."""
    try:
        result = conversation_engine.timbre_client.synthesize(
            text=request.text,
            language_code=request.language_code,
            voice=request.voice,
            speed=request.speed,
        )
        return result
    except GnaniAuthError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        logger.error(f"API synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/conversation/turn", tags=["Conversation"])
def process_conversation_turn(req: ConversationTurnRequest):
    """Process a full conversation turn with intent routing and agricultural reasoning."""
    # Retrieve or create session
    sess_id = req.session_id or f"sess_{len(sessions_store) + 1}"
    if sess_id not in sessions_store:
        sessions_store[sess_id] = ConversationSession(
            session_id=sess_id,
            farmer_name=req.farmer_name,
            crop=req.crop,
            location=req.location,
            language=req.language_code,
        )
    session = sessions_store[sess_id]

    turn, tts_result = conversation_engine.process_turn(
        session=session,
        user_query=req.user_query,
        input_type=req.input_type,
        image_path=req.image_path,
        crop=req.crop,
        location=req.location,
        lat=req.lat,
        lon=req.lon,
        synthesize_voice=req.synthesize_voice,
    )

    return {
        "session_id": sess_id,
        "turn": turn.model_dump(),
        "tts_result": tts_result.model_dump() if tts_result else None,
        "conversation_length": len(session.turns),
    }


@app.post("/api/v1/advisory/generate", tags=["Advisory"])
def generate_advisory(req: AdvisoryGenerationRequest):
    """Generate complete multi-source agricultural advisory."""
    try:
        result = system.process_request(
            farmer_name=req.farmer_name,
            crop=req.crop,
            location=req.location,
            symptoms=req.symptoms,
            lat=req.lat,
            lon=req.lon,
            language=req.language,
            phone_number=req.phone_number,
        )
        return result
    except Exception as e:
        logger.error(f"Advisory generation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/dashboard/metrics", tags=["Dashboard"])
def get_dashboard_metrics():
    """Retrieve operational metrics and outbreak risk distributions for field officers."""
    return {
        "risk_distribution": system.analytics.get_risk_distribution(),
        "crop_wise_issues": system.analytics.get_crop_wise_issues(),
        "growth_metrics": system.analytics.get_growth_metrics(),
        "field_visit_stats": system.analytics.get_field_visit_stats(),
    }


@app.get("/api/v1/knowledge/{crop}", tags=["Knowledge"])
def get_crop_knowledge(crop: str):
    """Retrieve ICAR agronomic profile and critical stages for crop."""
    profile = knowledge_service.get_crop_profile(crop)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No profile found for crop: {crop}")
    return profile


@app.get("/api/v1/disease/classes", tags=["Disease Intelligence"])
def get_disease_classes():
    """Retrieve list of supported crop disease classes with agronomic taxonomy."""
    manifest = system.disease_detector.manifest
    return {
        "classes": manifest.get("classes", []),
        "total_classes": len(manifest.get("classes", [])),
        "target_crop": "Tomato (Solanum lycopersicum)",
    }


@app.get("/api/v1/disease/model-info", tags=["Disease Intelligence"])
def get_model_info():
    """Retrieve diagnostic vision model metadata, checksum, and evaluation metrics."""
    return system.disease_detector.manifest


@app.post("/api/v1/disease/diagnose", response_model=DiagnosisResult, tags=["Disease Intelligence"])
async def diagnose_disease(
    image_file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
):
    """Diagnose crop disease from uploaded leaf photo with genuine Grad-CAM and severity."""
    try:
        content = await image_file.read()
        ext = os.path.splitext(image_file.filename or "leaf.jpg")[1] or ".jpg"
        temp_img_path = save_temporary_audio(content, prefix="leaf_diag", ext=ext)

        result = system.disease_detector.diagnose(
            image_path=temp_img_path,
            confidence_threshold=confidence_threshold,
        )
        return result
    except Exception as e:
        logger.error(f"API diagnosis error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/knowledge/sources", tags=["Knowledge Base"])
def get_knowledge_sources():
    """Retrieve all authoritative agricultural sources registered in AgriGuardian."""
    repo = system.recommendation_engine.retrieval_service.source_repo
    sources = repo.get_all()
    return {
        "sources": [s.model_dump() for s in sources],
        "count": len(sources),
    }


@app.get("/api/v1/knowledge/facts", tags=["Knowledge Base"])
def get_knowledge_facts(crop: Optional[str] = None):
    """Retrieve verified atomic facts filtered optionally by crop."""
    repo = system.recommendation_engine.retrieval_service.fact_repo
    if crop:
        facts = repo.get_by_crop(crop)
    else:
        facts = repo.get_all()
    return {
        "facts": [f.model_dump() for f in facts],
        "count": len(facts),
    }


@app.get("/api/v1/knowledge/manifest", tags=["Knowledge Base"])
def get_knowledge_manifest():
    """Retrieve knowledge manifest with catalog metrics and integrity signatures."""
    manifest_path = os.path.join(os.getcwd(), "data", "knowledge_manifest.json")
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "uninitialized", "sources_count": 0, "facts_count": 0}


@app.post("/api/v1/recommendation/generate", tags=["Recommendation Engine"])
def generate_recommendation(req: RecommendationRequest):
    """Generate evidence-grounded, context-aware personalized recommendation."""
    try:
        rec = system.recommendation_engine.generate_recommendation(
            query=req.query,
            crop=req.crop,
            language=req.language,
            state=req.state,
            crop_stage=req.crop_stage,
            days_after_sowing=req.days_after_sowing,
            weather_context=req.weather_context,
            image_diagnosis=req.image_diagnosis,
        )
        return rec.model_dump()
    except Exception as e:
        logger.error(f"Recommendation generation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ==============================================================================
# PHASE 4: WEATHER INTELLIGENCE & AGRICULTURAL CLIMATE RISKS
# ==============================================================================

@app.get("/api/v1/location/presets", tags=["Weather Intelligence"])
def get_location_presets():
    """Retrieve supported Indian agricultural states and preset district centroids."""
    return {
        "presets": system.weather_intel.location_manager.presets,
        "states": system.weather_intel.location_manager.get_supported_states(),
    }


@app.get("/api/v1/location/resolve", tags=["Weather Intelligence"])
def resolve_location(
    state: Optional[str] = None,
    district: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    source: str = "manual",
    consent: bool = False,
):
    """Resolve location metadata and precision classification."""
    try:
        if state and district:
            loc = system.weather_intel.location_manager.resolve_preset_district(state, district)
        elif lat is not None and lon is not None:
            loc = system.weather_intel.location_manager.resolve_coordinates(
                latitude=lat,
                longitude=lon,
                source=source,
                consent_granted=consent,
            )
        else:
            loc = system.weather_intel.location_manager.get_default_location()
        return loc.model_dump()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/weather/intelligence", tags=["Weather Intelligence"])
def get_weather_intelligence(
    state: Optional[str] = None,
    district: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    crop: Optional[str] = None,
    crop_stage: Optional[str] = None,
    forecast_days: int = 7,
):
    """Get complete agricultural weather report, risk assessment, and active alerts."""
    try:
        loc = None
        if state and district:
            loc = system.weather_intel.location_manager.resolve_preset_district(state, district)
        elif lat is not None and lon is not None:
            loc = system.weather_intel.location_manager.resolve_coordinates(latitude=lat, longitude=lon)
        else:
            loc = system.weather_intel.location_manager.get_default_location()

        intel = system.weather_intel.get_weather_intelligence(
            location=loc,
            crop=crop,
            crop_stage=crop_stage,
            forecast_days=forecast_days,
        )
        return {
            "location": intel["location"].model_dump(),
            "report": intel["report"].model_dump(),
            "risk_assessment": intel["risk_assessment"].model_dump(),
            "active_alerts": [a.model_dump() for a in intel["active_alerts"]],
        }
    except Exception as e:
        logger.error(f"Weather intelligence error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/weather/alerts", tags=["Weather Intelligence"])
def get_weather_alerts(district: Optional[str] = None):
    """Retrieve active and updated agricultural weather alerts."""
    alerts = system.weather_intel.alert_manager.get_active_alerts(district=district)
    return {
        "alerts": [a.model_dump() for a in alerts],
        "count": len(alerts),
    }


@app.post("/api/v1/weather/alerts/{alert_id}/acknowledge", tags=["Weather Intelligence"])
def acknowledge_weather_alert(alert_id: str):
    """Acknowledge an in-app agricultural weather alert."""
    success = system.weather_intel.alert_manager.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{alert_id}' not found.")
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/api/v1/weather/alerts/{alert_id}/dismiss", tags=["Weather Intelligence"])
def dismiss_weather_alert(alert_id: str):
    """Dismiss an in-app agricultural weather alert."""
    success = system.weather_intel.alert_manager.dismiss_alert(alert_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert '{alert_id}' not found.")
    return {"status": "dismissed", "alert_id": alert_id}


# ==============================================================================
# PHASE 5: AGRICULTURAL MARKET INTELLIGENCE & PRICE FORECASTING ENDPOINTS
# ==============================================================================

@app.get("/api/v1/market/commodities", tags=["Market Intelligence"])
def list_commodities():
    """Retrieve catalog of standardized agricultural commodities with Hindi names and aliases."""
    commodities = market_catalog.get_all_commodities()
    return {
        "commodities": [c.model_dump() for c in commodities],
        "count": len(commodities),
    }


@app.get("/api/v1/market/mandis", tags=["Market Intelligence"])
def list_mandis(state: Optional[str] = None, district: Optional[str] = None):
    """Retrieve registered APMC mandis across 12 states with verified GPS coordinates."""
    mandis = market_catalog.get_all_mandis(state=state, district=district)
    return {
        "mandis": [m.model_dump() for m in mandis],
        "count": len(mandis),
    }


@app.get("/api/v1/market/prices/latest", tags=["Market Intelligence"])
def get_latest_market_price(
    commodity: str,
    market_id: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
):
    """Fetch latest verified modal, min, and max price observation for a commodity."""
    try:
        obs = market_intel.provider.get_latest_price(
            commodity_id=commodity,
            market_id=market_id,
            state=state,
            district=district,
        )
        if not obs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No market price bulletin found for commodity '{commodity}'.",
            )
        return obs.model_dump()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error fetching latest market price: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/market/nearby", tags=["Market Intelligence"])
def discover_nearby_mandis(
    commodity: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    location: Optional[str] = None,
    radius_km: float = 120.0,
):
    """Discover nearby APMC mandis trading a commodity with straight-line distances."""
    try:
        nearby = market_intel.nearby_service.discover_nearby_prices(
            commodity_id=commodity,
            latitude=lat,
            longitude=lon,
            location_query=location,
            radius_km=radius_km,
        )
        return {
            "commodity": commodity,
            "nearby_mandis": [m.model_dump() for m in nearby],
            "count": len(nearby),
        }
    except Exception as e:
        logger.error(f"Error discovering nearby mandis: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/market/history", tags=["Market Intelligence"])
def get_market_history_and_trend(
    commodity: str,
    market_id: str,
    days: int = 90,
):
    """Retrieve sequential historical observations, statistical metrics, and trend trajectory."""
    try:
        obs_list = market_intel.provider.get_historical_observations(
            commodity_id=commodity,
            market_id=market_id,
            days=days,
        )
        if not obs_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No historical price records found for commodity '{commodity}' in market '{market_id}'.",
            )
        trend = market_intel.forecaster._fit_holt  # test helper
        trend_analysis = MarketAnalyticsService.analyze_trend(
            observations=obs_list,
            window_days=min(len(obs_list), days),
        )
        return {
            "commodity": commodity,
            "market_id": market_id,
            "observations_count": len(obs_list),
            "trend_analysis": trend_analysis.model_dump(),
            "observations": [o.model_dump() for o in obs_list],
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error retrieving market history: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/market/msp", tags=["Market Intelligence"])
def get_msp_benchmark(commodity: str, current_modal_price: Optional[float] = None):
    """Retrieve statutory MSP benchmark and comparative evaluation."""
    try:
        if current_modal_price is not None:
            comparison = market_intel.msp_provider.compare_price_with_msp(
                commodity_query=commodity,
                current_modal_price=current_modal_price,
            )
            return comparison
        else:
            msp_rec = market_intel.msp_provider.get_msp(commodity)
            if not msp_rec:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No statutory MSP policy benchmark found for '{commodity}'.",
                )
            return msp_rec.model_dump()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error retrieving MSP benchmark: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/market/forecast", tags=["Market Intelligence"])
def get_price_forecast(
    commodity: str,
    market_id: str,
    horizon_days: int = 14,
):
    """Generate 7-14 day time-series forecast with 80% & 95% intervals and backtest metrics."""
    try:
        obs_list = market_intel.provider.get_historical_observations(
            commodity_id=commodity,
            market_id=market_id,
            days=90,
        )
        if len(obs_list) < PriceForecaster.MIN_OBSERVATIONS_REQUIRED:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Insufficient sequential data: price forecasting requires at least {PriceForecaster.MIN_OBSERVATIONS_REQUIRED} observations (available: {len(obs_list)}).",
            )
        forecast_res = market_intel.forecaster.forecast_prices(
            observations=obs_list,
            horizon_days=horizon_days,
        )
        return forecast_res.model_dump()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error generating price forecast: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/market/calculate-net-realization", tags=["Market Intelligence"])
def calculate_net_realization(req: NetRealizationRequest):
    """Calculate transparent net farmgate realization factoring logistics, labor, and market fees."""
    try:
        res = NetRealizationCalculator.calculate(
            market_id=req.market_id,
            market_name=req.market_name,
            commodity_name=req.commodity_name,
            distance_km=req.distance_km,
            quantity_quintals=req.quantity_quintals,
            reference_modal_price=req.reference_modal_price,
            transport_cost_override=req.transport_cost_override,
            handling_fee_override=req.handling_fee_override,
            transport_rate_per_km_qtl=req.transport_rate_per_km_qtl,
        )
        return res.model_dump()
    except Exception as e:
        logger.error(f"Net realization calculation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/market/alerts", tags=["Market Intelligence"])
def get_market_alerts(commodity: Optional[str] = None, market_id: Optional[str] = None):
    """Retrieve active market alerts (price crash, surge, below MSP)."""
    alerts = market_intel.alert_manager.get_active_alerts(
        commodity_id=commodity,
        market_id=market_id,
    )
    return {
        "alerts": [a.model_dump() for a in alerts],
        "count": len(alerts),
    }


@app.post("/api/v1/market/alerts/{alert_id}/acknowledge", tags=["Market Intelligence"])
def acknowledge_market_alert(alert_id: str):
    """Acknowledge an in-app agricultural market alert."""
    success = market_intel.alert_manager.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Market alert '{alert_id}' not found.")
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/api/v1/market/alerts/{alert_id}/dismiss", tags=["Market Intelligence"])
def dismiss_market_alert(alert_id: str):
    """Dismiss an in-app agricultural market alert."""
    success = market_intel.alert_manager.dismiss_alert(alert_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Market alert '{alert_id}' not found.")
    return {"status": "dismissed", "alert_id": alert_id}




