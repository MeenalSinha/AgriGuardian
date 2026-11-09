"""AgriGuardian — FastAPI Headless REST Backend.

Provides programmatic access to multilingual voice intelligence,
crop diagnostics, weather context, and agricultural advisory services.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
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
from backend.app.schemas.farm import (
    Farm,
    FarmActivity,
    FarmActivityCreate,
    FarmContext,
    FarmCreate,
    FarmerProfile,
    FarmerProfileUpdate,
    FarmUpdate,
    IrrigationProfile,
    IrrigationProfileCreate,
    Plot,
    PlotCreate,
    PlotUpdate,
    SoilProfile,
    SoilProfileCreate,
    UserLoginRequest,
    UserLoginResponse,
    UserRecord,
    UserRegistrationRequest,
    CropCycle,
    CropCycleCreate,
    CropStageTransitionRequest,
    UserRole,
)
from backend.app.schemas.analytics import (
    AdminDashboardOverview,
    AdvisoryCase,
    CaseFollowUpCreate,
    CaseFollowUpRecord,
    FarmerDashboardOverview,
    FieldOfficerDashboardOverview,
    OfficerTerritoryAssignRequest,
    OfficerTerritoryScope,
)
from backend.app.schemas.expert import (
    CaseDossier,
    CaseReviewStatus,
    ExpertAccountStatus,
    ExpertCredentialCreate,
    ExpertCredentialRecord,
    ExpertDecisionSubmission,
    ExpertProfile,
    ExpertProfileCreate,
    ExpertProfileUpdate,
    ExpertProvisionRequest,
    ExpertReviewComment,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
    ExpertVerificationAction,
    FarmerVerificationBadgeInfo,
    KnowledgeImprovementProposal,
    ReviewPriority,
    SecondaryReviewSubmission,
)
from backend.app.services.analytics.metric_definitions import METRIC_CATALOG
from backend.app.services.auth.auth_service import AuthService, AuthenticationError
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_context_service import FarmContextService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService
from backend.app.services.farm.soil_irrigation_service import SoilIrrigationService

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

# Phase 6 Farm & Identity Singletons
auth_service = AuthService(settings=settings)
farmer_profile_service = FarmerProfileService()
farm_service = FarmService()
plot_service = PlotService()
crop_cycle_service = CropCycleService()
soil_irrigation_service = SoilIrrigationService()
crop_calendar_service = CropCalendarService()
farm_context_service = FarmContextService(
    profile_service=farmer_profile_service,
    farm_service=farm_service,
    plot_service=plot_service,
    cycle_service=crop_cycle_service,
    soil_service=soil_irrigation_service,
    calendar_service=crop_calendar_service,
)


def get_current_farmer_auth(authorization: Optional[str] = Header(None)) -> Tuple[UserRecord, FarmerProfile]:
    """FastAPI dependency to extract and validate authenticated farmer bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization bearer token required in headers.",
        )
    token = authorization.replace("Bearer ", "").strip()
    try:
        return auth_service.get_current_farmer(token)
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def get_current_user_auth(authorization: Optional[str] = Header(None)) -> UserRecord:
    """FastAPI dependency to extract and validate authenticated user record (any role)."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization bearer token required in headers.",
        )
    token = authorization.replace("Bearer ", "").strip()
    try:
        return auth_service.get_current_user(token)
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def require_roles(allowed_roles: List[UserRole]):
    """Factory dependency enforcing specific user roles."""
    def dependency(user: UserRecord = Depends(get_current_user_auth)) -> UserRecord:
        if user.role not in allowed_roles:
            role_names = [r.value for r in allowed_roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: User '{user.username}' has role '{user.role.value}'. Requires one of: {role_names}",
            )
        return user
    return dependency

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


# ==============================================================================
# PHASE 6: FARMER IDENTITY & AUTHENTICATION ENDPOINTS
# ==============================================================================

@app.post("/api/v1/auth/register", response_model=UserLoginResponse, status_code=status.HTTP_201_CREATED, tags=["Farmer Identity & Multi-Farm"])
def register_farmer(req: UserRegistrationRequest):
    """Register a new farmer account and initialize their personalized profile."""
    try:
        user, profile, token = auth_service.register_farmer(req)
        return UserLoginResponse(
            access_token=token,
            token_type="bearer",
            user_id=user.user_id,
            farmer_id=profile.farmer_id,
            username=user.username,
            full_name=profile.full_name,
            role=user.role,
            preferred_language=profile.preferred_language,
            profile_completion_pct=profile.profile_completion_pct,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Farmer registration error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed.")


@app.post("/api/v1/auth/login", response_model=UserLoginResponse, tags=["Farmer Identity & Multi-Farm"])
def login_farmer(req: UserLoginRequest):
    """Authenticate farmer with password and receive signed bearer token."""
    try:
        return auth_service.authenticate_user(req)
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@app.get("/api/v1/auth/me", response_model=FarmerProfile, tags=["Farmer Identity & Multi-Farm"])
def get_current_user_profile(auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth)):
    """Fetch profile of currently authenticated farmer."""
    _, profile = auth
    enriched = farmer_profile_service.get_profile(profile.farmer_id)
    return enriched or profile


# ==============================================================================
# FARMER PROFILE ENDPOINTS
# ==============================================================================

@app.get("/api/v1/farmers/profile", response_model=FarmerProfile, tags=["Farmer Identity & Multi-Farm"])
def get_farmer_profile(auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth)):
    """Retrieve full profile with progressive onboarding completion percentage."""
    _, profile = auth
    p = farmer_profile_service.get_profile(profile.farmer_id)
    if not p:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")
    return p


@app.put("/api/v1/farmers/profile", response_model=FarmerProfile, tags=["Farmer Identity & Multi-Farm"])
def update_farmer_profile(
    req: FarmerProfileUpdate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Update profile attributes and recalculate completion score."""
    _, profile = auth
    updated = farmer_profile_service.update_profile(profile.farmer_id, req)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Failed to update profile.")
    return updated


# ==============================================================================
# MULTI-FARM MANAGEMENT ENDPOINTS (Strict Multi-User Tenancy)
# ==============================================================================

@app.get("/api/v1/farms", response_model=List[Farm], tags=["Farmer Identity & Multi-Farm"])
def list_farmer_farms(auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth)):
    """List all farms owned by authenticated farmer."""
    _, profile = auth
    return farm_service.list_farms(profile.farmer_id)


@app.post("/api/v1/farms", response_model=Farm, status_code=status.HTTP_201_CREATED, tags=["Farmer Identity & Multi-Farm"])
def create_farm(
    req: FarmCreate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Create a new farm under the authenticated farmer's ownership."""
    _, profile = auth
    try:
        return farm_service.create_farm(profile.farmer_id, req)
    except Exception as e:
        logger.error(f"Create farm error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/farms/{farm_id}", response_model=Farm, tags=["Farmer Identity & Multi-Farm"])
def get_farm(
    farm_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Retrieve specific farm enforcing tenant isolation."""
    _, profile = auth
    farm = farm_service.get_farm(farm_id=farm_id, farmer_id=profile.farmer_id)
    if not farm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found or unauthorized.")
    return farm


@app.put("/api/v1/farms/{farm_id}", response_model=Farm, tags=["Farmer Identity & Multi-Farm"])
def update_farm(
    farm_id: str,
    req: FarmUpdate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Update farm details enforcing tenant ownership."""
    _, profile = auth
    updated = farm_service.update_farm(farm_id=farm_id, farmer_id=profile.farmer_id, update_data=req)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found or unauthorized.")
    return updated


@app.delete("/api/v1/farms/{farm_id}", tags=["Farmer Identity & Multi-Farm"])
def delete_farm(
    farm_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Delete a farm and associated plots/cycles enforcing tenant ownership."""
    _, profile = auth
    success = farm_service.delete_farm(farm_id=farm_id, farmer_id=profile.farmer_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found or unauthorized.")
    return {"status": "deleted", "farm_id": farm_id}


@app.post("/api/v1/farms/{farm_id}/activate", response_model=Farm, tags=["Farmer Identity & Multi-Farm"])
def activate_farm(
    farm_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Set specified farm as active context for weather, advisory, and market discovery."""
    _, profile = auth
    activated = farm_service.set_active_farm(farm_id=farm_id, farmer_id=profile.farmer_id)
    if not activated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farm not found or unauthorized.")
    return activated


# ==============================================================================
# PLOT / FIELD ENDPOINTS
# ==============================================================================

@app.get("/api/v1/farms/{farm_id}/plots", response_model=List[Plot], tags=["Farmer Identity & Multi-Farm"])
def list_farm_plots(
    farm_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """List all plots on a farm verifying tenant ownership."""
    _, profile = auth
    return plot_service.list_plots_by_farm(farm_id=farm_id, farmer_id=profile.farmer_id)


@app.post("/api/v1/farms/{farm_id}/plots", response_model=Plot, status_code=status.HTTP_201_CREATED, tags=["Farmer Identity & Multi-Farm"])
def create_plot(
    farm_id: str,
    req: PlotCreate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Add a plot/field inside a farm."""
    _, profile = auth
    try:
        return plot_service.create_plot(farm_id=farm_id, farmer_id=profile.farmer_id, plot_data=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/plots/{plot_id}", response_model=Plot, tags=["Farmer Identity & Multi-Farm"])
def get_plot(
    plot_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Retrieve plot details verifying tenant ownership."""
    _, profile = auth
    plot = plot_service.get_plot(plot_id=plot_id, farmer_id=profile.farmer_id)
    if not plot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plot not found or unauthorized.")
    return plot


@app.put("/api/v1/plots/{plot_id}", response_model=Plot, tags=["Farmer Identity & Multi-Farm"])
def update_plot(
    plot_id: str,
    req: PlotUpdate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Update plot specifications verifying tenant ownership."""
    _, profile = auth
    updated = plot_service.update_plot(plot_id=plot_id, farmer_id=profile.farmer_id, update_data=req)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plot not found or unauthorized.")
    return updated


@app.delete("/api/v1/plots/{plot_id}", tags=["Farmer Identity & Multi-Farm"])
def delete_plot(
    plot_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Delete a plot verifying tenant ownership."""
    _, profile = auth
    success = plot_service.delete_plot(plot_id=plot_id, farmer_id=profile.farmer_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plot not found or unauthorized.")
    return {"status": "deleted", "plot_id": plot_id}


# ==============================================================================
# CROP LIFECYCLE & STAGE ENDPOINTS
# ==============================================================================

@app.get("/api/v1/farms/{farm_id}/crop-cycles", response_model=List[CropCycle], tags=["Farmer Identity & Multi-Farm"])
def list_farm_active_cycles(
    farm_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """List active crop cycles on specified farm."""
    _, profile = auth
    return crop_cycle_service.get_active_cycles_by_farm(farm_id=farm_id, farmer_id=profile.farmer_id)


@app.post("/api/v1/plots/{plot_id}/crop-cycles", response_model=CropCycle, status_code=status.HTTP_201_CREATED, tags=["Farmer Identity & Multi-Farm"])
def create_crop_cycle(
    plot_id: str,
    req: CropCycleCreate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Start a new crop cycle on a plot."""
    _, profile = auth
    try:
        return crop_cycle_service.create_crop_cycle(plot_id=plot_id, farmer_id=profile.farmer_id, cycle_data=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/crop-cycles/{cycle_id}", response_model=CropCycle, tags=["Farmer Identity & Multi-Farm"])
def get_crop_cycle(
    cycle_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Get crop cycle with real-time days elapsed and estimated growth stage."""
    _, profile = auth
    cycle = crop_cycle_service.get_crop_cycle(cycle_id=cycle_id, farmer_id=profile.farmer_id)
    if not cycle:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop cycle not found or unauthorized.")
    return cycle


@app.post("/api/v1/crop-cycles/{cycle_id}/transition", response_model=CropCycle, tags=["Farmer Identity & Multi-Farm"])
def transition_crop_stage(
    cycle_id: str,
    req: CropStageTransitionRequest,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Transition crop growth stage (e.g., Vegetative -> Flowering -> Harvested) preserving history."""
    _, profile = auth
    updated = crop_cycle_service.transition_stage(cycle_id=cycle_id, farmer_id=profile.farmer_id, req=req)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop cycle not found or unauthorized.")
    return updated


# ==============================================================================
# SOIL & IRRIGATION PROFILE ENDPOINTS
# ==============================================================================

@app.get("/api/v1/plots/{plot_id}/soil", response_model=Optional[SoilProfile], tags=["Farmer Identity & Multi-Farm"])
def get_plot_soil(
    plot_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Fetch soil fertility and NPK profile for plot."""
    _, profile = auth
    return soil_irrigation_service.get_soil_profile(plot_id=plot_id, farmer_id=profile.farmer_id)


@app.post("/api/v1/plots/{plot_id}/soil", response_model=SoilProfile, tags=["Farmer Identity & Multi-Farm"])
def save_plot_soil(
    plot_id: str,
    req: SoilProfileCreate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Save or update soil test and NPK record for plot."""
    _, profile = auth
    try:
        return soil_irrigation_service.save_soil_profile(plot_id=plot_id, farmer_id=profile.farmer_id, profile_data=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/plots/{plot_id}/irrigation", response_model=Optional[IrrigationProfile], tags=["Farmer Identity & Multi-Farm"])
def get_plot_irrigation(
    plot_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Fetch irrigation system details for plot."""
    _, profile = auth
    return soil_irrigation_service.get_irrigation_profile(plot_id=plot_id, farmer_id=profile.farmer_id)


@app.post("/api/v1/plots/{plot_id}/irrigation", response_model=IrrigationProfile, tags=["Farmer Identity & Multi-Farm"])
def save_plot_irrigation(
    plot_id: str,
    req: IrrigationProfileCreate,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Save or update irrigation system specifications for plot."""
    _, profile = auth
    try:
        return soil_irrigation_service.save_irrigation_profile(plot_id=plot_id, farmer_id=profile.farmer_id, profile_data=req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ==============================================================================
# FARM ACTIVITIES & CALENDAR ENDPOINTS
# ==============================================================================

@app.get("/api/v1/farms/{farm_id}/activities", response_model=List[FarmActivity], tags=["Farmer Identity & Multi-Farm"])
def list_farm_activities(
    farm_id: str,
    status: Optional[str] = None,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """List farm activities synchronized with current due status."""
    _, profile = auth
    return crop_calendar_service.list_activities(farmer_id=profile.farmer_id, farm_id=farm_id, status=status)


@app.post("/api/v1/farms/{farm_id}/activities", response_model=FarmActivity, status_code=status.HTTP_201_CREATED, tags=["Farmer Identity & Multi-Farm"])
def schedule_farm_activity(
    farm_id: str,
    req: FarmActivityCreate,
    plot_id: Optional[str] = None,
    cycle_id: Optional[str] = None,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Schedule an agronomic activity on the farm."""
    _, profile = auth
    try:
        return crop_calendar_service.create_activity(
            farm_id=farm_id,
            farmer_id=profile.farmer_id,
            activity_data=req,
            plot_id=plot_id,
            cycle_id=cycle_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/activities/{activity_id}/complete", response_model=FarmActivity, tags=["Farmer Identity & Multi-Farm"])
def mark_activity_completed(
    activity_id: str,
    cost: Optional[float] = None,
    notes: Optional[str] = None,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Mark a farm activity as completed."""
    _, profile = auth
    completed = crop_calendar_service.complete_activity(
        activity_id=activity_id,
        farmer_id=profile.farmer_id,
        cost_incurred=cost,
        notes=notes,
    )
    if not completed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Activity not found or unauthorized.")
    return completed


@app.post("/api/v1/crop-cycles/{cycle_id}/generate-tasks", response_model=List[FarmActivity], tags=["Farmer Identity & Multi-Farm"])
def generate_stage_tasks(
    cycle_id: str,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Generate agronomic task recommendations grounded in current crop growth stage."""
    _, profile = auth
    cycle = crop_cycle_service.get_crop_cycle(cycle_id=cycle_id, farmer_id=profile.farmer_id)
    if not cycle:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crop cycle not found or unauthorized.")
    return crop_calendar_service.generate_grounded_stage_activities(cycle=cycle, farmer_id=profile.farmer_id)


# ==============================================================================
# UNIFIED FARM CONTEXT RESOLUTION ENDPOINT
# ==============================================================================

@app.get("/api/v1/farm/context", response_model=FarmContext, tags=["Farmer Identity & Multi-Farm"])
def get_resolved_farm_context(
    farm_id: Optional[str] = None,
    plot_id: Optional[str] = None,
    auth: Tuple[UserRecord, FarmerProfile] = Depends(get_current_farmer_auth),
):
    """Resolve active farm, plot, crop stage, soil, and due tasks for authenticated farmer."""
    _, profile = auth
    return farm_context_service.resolve_context(
        farmer_id=profile.farmer_id,
        farm_id_override=farm_id,
        plot_id_override=plot_id,
    )


# ==============================================================================
# PHASE 7: ROLE-BASED DASHBOARDS & AGRICULTURAL ANALYTICS ENDPOINTS
# ==============================================================================

@app.get("/api/v1/dashboard/metrics/catalog", tags=["Role-Based Dashboards & Analytics"])
def get_metrics_catalog():
    """Retrieve complete catalog of defined dashboard metrics with formulas and data sources."""
    return METRIC_CATALOG


@app.get("/api/v1/dashboard/farmer/overview", response_model=FarmerDashboardOverview, tags=["Role-Based Dashboards & Analytics"])
def get_farmer_dashboard_overview(
    farm_id: Optional[str] = None,
    force_refresh: bool = False,
    user: UserRecord = Depends(require_roles([UserRole.FARMER, UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Retrieve personalized agricultural dashboard overview for authenticated farmer."""
    farmer_profile = auth_service.repo.get_farmer_profile_by_user_id(user.user_id)
    if not farmer_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User '{user.username}' has no linked farmer profile to view farmer dashboard.",
        )
    try:
        return system.analytics_orchestrator.get_farmer_dashboard(
            farmer_id=farmer_profile.farmer_id,
            farm_id=farm_id,
            force_refresh=force_refresh,
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Farmer dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/dashboard/officer/overview", response_model=FieldOfficerDashboardOverview, tags=["Role-Based Dashboards & Analytics"])
def get_field_officer_dashboard_overview(
    crop: Optional[str] = None,
    risk: Optional[str] = None,
    status_filter: Optional[str] = None,
    date_range_days: int = 30,
    force_refresh: bool = False,
    user: UserRecord = Depends(require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Retrieve regional agricultural intelligence dashboard for authorized field officers."""
    try:
        return system.analytics_orchestrator.get_field_officer_dashboard(
            officer_id=user.user_id,
            officer_name=user.username,
            filter_crop=crop,
            filter_risk=risk,
            filter_status=status_filter,
            date_range_days=date_range_days,
            force_refresh=force_refresh,
        )
    except Exception as e:
        logger.error(f"Field officer dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/dashboard/officer/cases", response_model=List[AdvisoryCase], tags=["Role-Based Dashboards & Analytics"])
def get_field_officer_cases(
    crop: Optional[str] = None,
    risk: Optional[str] = None,
    status_filter: Optional[str] = None,
    date_range_days: int = 30,
    user: UserRecord = Depends(require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Query reported cases within assigned officer jurisdiction."""
    territories = system.analytics_orchestrator.get_officer_territory(user.user_id)
    if not territories and user.role != UserRole.ADMINISTRATOR:
        return []

    districts = [t.district for t in territories] if territories else None
    date_from = None
    if date_range_days > 0:
        date_from = (datetime.now() - timedelta(days=date_range_days)).isoformat()

    return system.analytics_orchestrator.db.list_advisory_cases_for_territory(
        districts=districts,
        crop=crop,
        risk=risk,
        status=status_filter,
        date_from=date_from,
        limit=100,
    )


@app.post("/api/v1/dashboard/officer/cases/{case_id}/follow-up", response_model=CaseFollowUpRecord, status_code=status.HTTP_201_CREATED, tags=["Role-Based Dashboards & Analytics"])
def record_case_follow_up(
    case_id: str,
    req: CaseFollowUpCreate,
    user: UserRecord = Depends(require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Record an agronomist interaction, field visit note, or status update."""
    try:
        return system.analytics_orchestrator.record_case_follow_up(
            case_id=case_id,
            officer_id=user.user_id,
            officer_name=user.username,
            notes=req.notes,
            action_taken=req.action_taken,
            new_status=req.new_status,
            is_admin=(user.role == UserRole.ADMINISTRATOR),
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Follow up error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/v1/dashboard/officer/cases/{case_id}/follow-ups", response_model=List[CaseFollowUpRecord], tags=["Role-Based Dashboards & Analytics"])
def list_case_follow_ups(
    case_id: str,
    user: UserRecord = Depends(require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Retrieve full chronological follow-up audit trail for a case."""
    try:
        return system.analytics_orchestrator.list_case_follow_ups(
            case_id=case_id,
            officer_id=user.user_id,
            is_admin=(user.role == UserRole.ADMINISTRATOR),
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get("/api/v1/dashboard/admin/overview", response_model=AdminDashboardOverview, tags=["Role-Based Dashboards & Analytics"])
def get_admin_dashboard_overview(
    force_refresh: bool = False,
    user: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """Retrieve platform observability, integration health, and user metrics for administrators."""
    try:
        return system.analytics_orchestrator.get_platform_admin_dashboard(
            admin_id=user.user_id,
            admin_name=user.username,
            force_refresh=force_refresh,
        )
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/dashboard/admin/territory/assign", response_model=OfficerTerritoryScope, status_code=status.HTTP_201_CREATED, tags=["Role-Based Dashboards & Analytics"])
def assign_officer_territory(
    req: OfficerTerritoryAssignRequest,
    user: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """Assign agricultural territory to a field officer account."""
    try:
        return system.analytics_orchestrator.assign_officer_territory(
            officer_id=req.officer_id,
            state=req.state,
            district=req.district,
            block=req.block,
        )
    except Exception as e:
        logger.error(f"Territory assign error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/dashboard/officer/territory", response_model=List[OfficerTerritoryScope], tags=["Role-Based Dashboards & Analytics"])
def get_my_officer_territory(
    user: UserRecord = Depends(require_roles([UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR])),
):
    """Retrieve geographic territory assigned to current field officer."""
    return system.analytics_orchestrator.get_officer_territory(user.user_id)


# =========================================================================
# PHASE 8: AGRICULTURAL EXPERT VERIFICATION & HUMAN-IN-THE-LOOP ENDPOINTS
# =========================================================================

class ReviewCommentCreate(BaseModel):
    content: str = Field(..., min_length=1)


class ProposalStatusUpdate(BaseModel):
    status: str = Field(..., min_length=3)


@app.get("/api/v1/experts/me", response_model=ExpertProfile, tags=["Agricultural Expert Verification"])
def get_my_expert_profile(
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Retrieve verified profile and specializations for logged-in agronomist."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expert profile not found for this account.",
        )
    return profile


@app.put("/api/v1/experts/me", response_model=ExpertProfile, tags=["Agricultural Expert Verification"])
def update_my_expert_profile(
    req: ExpertProfileUpdate,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Update profile attributes and agricultural specializations."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    
    updated = system.farm_context.repo.update_expert_profile(
        expert_id=profile.expert_id,
        updates=req.model_dump(exclude_unset=True),
    )
    return updated or profile


@app.post("/api/v1/experts/me/credentials", response_model=ExpertCredentialRecord, status_code=status.HTTP_201_CREATED, tags=["Agricultural Expert Verification"])
def add_my_credential(
    req: ExpertCredentialCreate,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Attach an academic degree, ICAR certification, or government accreditation."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")

    import uuid
    cred = ExpertCredentialRecord(
        credential_id=f"crd_{uuid.uuid4().hex[:12]}",
        expert_id=profile.expert_id,
        institution_name=req.institution_name,
        degree_or_certification=req.degree_or_certification,
        registration_number=req.registration_number,
        document_url=req.document_url,
        issued_date=req.issued_date,
        expiry_date=req.expiry_date,
        is_verified=False,
    )
    return system.farm_context.repo.add_expert_credential(cred)


@app.get("/api/v1/experts/me/credentials", response_model=List[ExpertCredentialRecord], tags=["Agricultural Expert Verification"])
def get_my_credentials(
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """List all credentials associated with the authenticated agronomist."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    return system.farm_context.repo.get_expert_credentials(profile.expert_id)


@app.post("/api/v1/admin/experts/provision", response_model=ExpertProfile, status_code=status.HTTP_201_CREATED, tags=["Admin & Expert Governance"])
def admin_provision_expert(
    req: ExpertProvisionRequest,
    admin: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """Administrative provisioning of accredited Agricultural Expert accounts."""
    import uuid
    # Provision privileged user account
    user, _ = auth_service.provision_privileged_user(
        username=req.username,
        password=req.password,
        role=UserRole.AGRONOMIST,
    )
    
    expert_id = f"exp_{uuid.uuid4().hex[:12]}"
    domains = []
    for d in req.domains:
        try:
            domains.append(ExpertSpecializationDomain(d))
        except Exception:
            pass
    if not domains:
        domains = [ExpertSpecializationDomain.GENERAL_AGRONOMY]

    specializations = ExpertSpecializationItem(
        crop_types=req.crop_types or ["Wheat", "Rice", "Tomato", "Potato"],
        domains=domains,
        states=req.states or ["Haryana", "Punjab", "Uttar Pradesh"],
        languages=req.languages,
    )

    now_str = datetime.now().isoformat()
    credentials = []
    if req.initial_credential_institution and req.initial_credential_reg_num:
        cred = ExpertCredentialRecord(
            credential_id=f"crd_{uuid.uuid4().hex[:12]}",
            expert_id=expert_id,
            institution_name=req.initial_credential_institution,
            degree_or_certification=req.initial_credential_degree or "M.Sc. Plant Pathology",
            registration_number=req.initial_credential_reg_num,
            is_verified=True,
            verified_by_admin=admin.user_id,
            verified_at=now_str,
        )
        credentials.append(cred)

    profile = ExpertProfile(
        expert_id=expert_id,
        user_id=user.user_id,
        full_name=req.full_name,
        email=req.email,
        phone_number=req.phone_number,
        institution=req.institution,
        designation=req.designation,
        experience_years=req.experience_years,
        specializations=specializations,
        credentials=credentials,
        verification_status=ExpertAccountStatus.ACTIVE_VERIFIED,
        verified_by_admin_id=admin.user_id,
        verified_at=now_str,
        created_at=now_str,
        updated_at=now_str,
    )

    created = system.farm_context.repo.create_expert_profile(profile)
    logger.info(f"Admin '{admin.username}' provisioned verified expert '{req.full_name}' ({expert_id})")
    return created


@app.get("/api/v1/admin/experts", response_model=List[ExpertProfile], tags=["Admin & Expert Governance"])
def list_experts(
    status: Optional[str] = None,
    admin: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """List all registered agricultural experts with optional status filter."""
    return system.farm_context.repo.list_expert_profiles(status=status)


@app.put("/api/v1/admin/experts/{expert_id}/verify", tags=["Admin & Expert Governance"])
def admin_verify_expert(
    expert_id: str,
    action: ExpertVerificationAction,
    admin: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """Admin action to verify, reject, or suspend an expert."""
    success = system.farm_context.repo.verify_expert(
        expert_id=expert_id,
        admin_id=admin.user_id,
        status=action.action,
        rejection_reason=action.rejection_reason,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    return {"message": f"Expert '{expert_id}' status updated to '{action.action.value}'."}


@app.get("/api/v1/experts/reviews", response_model=List[Dict[str, Any]], tags=["Agricultural Expert Verification"])
def list_review_cases(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR])),
):
    """Query review cases in the expert queue."""
    expert_id = None
    if user.role == UserRole.AGRONOMIST and status != "queued":
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if profile:
            expert_id = profile.expert_id
    return system.farm_context.repo.list_review_cases(status=status, priority=priority, expert_id=expert_id)


@app.get("/api/v1/experts/reviews/{review_id}", response_model=Dict[str, Any], tags=["Agricultural Expert Verification"])
def get_review_case_details(
    review_id: str,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR])),
):
    """Retrieve metadata and lifecycle status for a review case."""
    rev = system.farm_context.repo.get_review_case(review_id)
    if not rev:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found.")
    if user.role == UserRole.AGRONOMIST:
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert profile not found.")
        if rev.get("assigned_expert_id") and rev["assigned_expert_id"] != profile.expert_id:
            if rev.get("secondary_expert_id") != profile.expert_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is assigned to another expert.")
    return rev


@app.get("/api/v1/experts/cases/{case_id}/dossier", response_model=CaseDossier, tags=["Agricultural Expert Verification"])
def get_case_dossier(
    case_id: str,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR, UserRole.FIELD_OFFICER])),
):
    """Assemble complete agricultural case dossier for expert review."""
    advisory = system.farm_context.repo.get_advisory_case(case_id)
    if not advisory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")

    rev = system.farm_context.repo.get_review_case_by_advisory_id(case_id)
    if user.role == UserRole.AGRONOMIST:
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert profile not found.")
        if rev and rev.get("assigned_expert_id") and rev["assigned_expert_id"] != profile.expert_id:
            if rev.get("secondary_expert_id") != profile.expert_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is assigned to another expert.")
    elif user.role == UserRole.FIELD_OFFICER:
        officer = system.farm_context.repo.get_field_officer(user.user_id)
        if officer and officer.assigned_districts and advisory.district not in officer.assigned_districts:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is outside assigned territory.")

    try:
        return system.expert_review_service.build_case_dossier(case_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.post("/api/v1/experts/reviews/{review_id}/claim", tags=["Agricultural Expert Verification"])
def claim_case(
    review_id: str,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Claim an unassigned queued case for review."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    try:
        claimed = system.expert_review_service.claim_case(review_id, profile.expert_id)
        if not claimed:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not claim case (already claimed or inactive).")
        return {"message": "Case successfully claimed.", "expert_id": profile.expert_id}
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@app.post("/api/v1/experts/reviews/decision", tags=["Agricultural Expert Verification"])
def submit_expert_decision(
    req: ExpertDecisionSubmission,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Submit diagnosis assessment and versioned treatment recommendation."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    try:
        return system.expert_review_service.submit_expert_decision(req, profile.expert_id)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/api/v1/experts/reviews/secondary-review", tags=["Agricultural Expert Verification"])
def submit_secondary_review(
    req: SecondaryReviewSubmission,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST])),
):
    """Submit secondary agronomist sign-off on dual-review cases."""
    profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Expert profile not found.")
    try:
        return system.expert_review_service.submit_secondary_review(req, profile.expert_id)
    except (ValueError, PermissionError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/api/v1/experts/reviews/{review_id}/versions", tags=["Agricultural Expert Verification"])
def list_recommendation_versions(
    review_id: str,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR])),
):
    """List immutable recommendation versions for a case audit trail."""
    rev = system.farm_context.repo.get_review_case(review_id)
    if not rev:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found.")
    if user.role == UserRole.AGRONOMIST:
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert profile not found.")
        if rev.get("assigned_expert_id") and rev["assigned_expert_id"] != profile.expert_id:
            if rev.get("secondary_expert_id") != profile.expert_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is assigned to another expert.")
    return system.farm_context.repo.list_recommendation_versions(review_id)


@app.post("/api/v1/experts/reviews/{review_id}/comments", tags=["Agricultural Expert Verification"])
def add_review_comment(
    review_id: str,
    req: ReviewCommentCreate,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR, UserRole.FIELD_OFFICER])),
):
    """Add a collaboration comment to the review timeline."""
    rev = system.farm_context.repo.get_review_case(review_id)
    if not rev:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found.")
    if user.role == UserRole.AGRONOMIST:
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert profile not found.")
        if rev.get("assigned_expert_id") and rev["assigned_expert_id"] != profile.expert_id:
            if rev.get("secondary_expert_id") != profile.expert_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is assigned to another expert.")
    elif user.role == UserRole.FIELD_OFFICER:
        advisory = system.farm_context.repo.get_advisory_case(rev["case_id"])
        officer = system.farm_context.repo.get_field_officer(user.user_id)
        if officer and officer.assigned_districts and advisory and advisory.district not in officer.assigned_districts:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is outside assigned territory.")

    import uuid
    comment = ExpertReviewComment(
        comment_id=f"cmt_{uuid.uuid4().hex[:12]}",
        review_id=review_id,
        author_id=user.user_id,
        author_name=user.username,
        author_role=user.role.value,
        content=req.content,
    )
    system.farm_context.repo.add_review_comment(comment)
    return {"message": "Comment added successfully."}


@app.get("/api/v1/experts/reviews/{review_id}/comments", response_model=List[ExpertReviewComment], tags=["Agricultural Expert Verification"])
def list_review_comments(
    review_id: str,
    user: UserRecord = Depends(require_roles([UserRole.AGRONOMIST, UserRole.ADMINISTRATOR, UserRole.FIELD_OFFICER])),
):
    """List chronological collaboration comments for a case."""
    rev = system.farm_context.repo.get_review_case(review_id)
    if not rev:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found.")
    if user.role == UserRole.AGRONOMIST:
        profile = system.farm_context.repo.get_expert_profile_by_user_id(user.user_id)
        if not profile:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expert profile not found.")
        if rev.get("assigned_expert_id") and rev["assigned_expert_id"] != profile.expert_id:
            if rev.get("secondary_expert_id") != profile.expert_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is assigned to another expert.")
    elif user.role == UserRole.FIELD_OFFICER:
        advisory = system.farm_context.repo.get_advisory_case(rev["case_id"])
        officer = system.farm_context.repo.get_field_officer(user.user_id)
        if officer and officer.assigned_districts and advisory and advisory.district not in officer.assigned_districts:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Case is outside assigned territory.")

    return system.farm_context.repo.list_review_comments(review_id)


@app.get("/api/v1/farmer/cases/{case_id}/badge", response_model=FarmerVerificationBadgeInfo, tags=["Agricultural Expert Verification"])
def get_farmer_verification_badge(
    case_id: str,
    user: UserRecord = Depends(require_roles([UserRole.FARMER, UserRole.FIELD_OFFICER, UserRole.AGRONOMIST, UserRole.ADMINISTRATOR])),
):
    """Retrieve farmer verification badge, accredited provenance, and audio scripts."""
    advisory = system.farm_context.repo.get_advisory_case(case_id)
    if not advisory:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Advisory case not found.")

    if user.role == UserRole.FARMER:
        fmr_prof = system.farm_context.repo.get_farmer_profile_by_user_id(user.user_id)
        if not fmr_prof or advisory.farmer_id != fmr_prof.farmer_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Cannot access another farmer's verification badge.")

    try:
        return system.expert_review_service.get_farmer_verification_badge(case_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get("/api/v1/admin/proposals", response_model=List[KnowledgeImprovementProposal], tags=["Admin & Expert Governance"])
def list_knowledge_proposals(
    status: Optional[str] = None,
    admin: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """List knowledge improvement proposals generated from expert overrides."""
    return system.farm_context.repo.list_knowledge_proposals(status=status)


@app.put("/api/v1/admin/proposals/{proposal_id}", tags=["Admin & Expert Governance"])
def update_knowledge_proposal_status(
    proposal_id: str,
    req: ProposalStatusUpdate,
    admin: UserRecord = Depends(require_roles([UserRole.ADMINISTRATOR])),
):
    """Admin action to approve, integrate, or reject a knowledge improvement proposal."""
    success = system.farm_context.repo.update_knowledge_proposal_status(
        proposal_id=proposal_id,
        status=req.status,
        admin_id=admin.user_id,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Proposal not found.")
    return {"message": f"Proposal '{proposal_id}' status updated to '{req.status}'."}







