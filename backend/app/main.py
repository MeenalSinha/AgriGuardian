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
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector
from backend.app.services.agriculture.knowledge_service import KnowledgeService
from backend.app.services.conversation.engine import ConversationEngine

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

# In-memory sessions store (can be backed by Redis / PostgreSQL in Phase 2)
sessions_store: Dict[str, ConversationSession] = {}


# ==============================================================================
# SCHEMAS FOR API REQUESTS
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


