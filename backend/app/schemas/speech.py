"""Pydantic schemas for Speech-to-Text and Text-to-Speech interactions."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioValidationResult(BaseModel):
    """Result of audio file or buffer validation."""
    is_valid: bool
    format: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class TranscriptionRequest(BaseModel):
    """Request schema for Gnani Prisma STT."""
    audio_path: str
    language_code: str = "hi-IN"
    model: str = "prisma-v2.5"


class TranscriptionResult(BaseModel):
    """Normalized response schema from Gnani Prisma STT."""
    success: bool
    transcript: str
    language_code: str
    model: str
    request_id: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SpeechSynthesisRequest(BaseModel):
    """Request schema for Gnani Timbre TTS."""
    text: str
    language_code: str = "hi-IN"
    voice: str = "Nalini"
    model: str = "timbre-v2.5"
    speed: float = 1.0
    container: str = "wav"
    sample_rate: int = 48000


class SpeechSynthesisResult(BaseModel):
    """Normalized response schema from Gnani Timbre TTS."""
    success: bool
    audio_file_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    container: str = "wav"
    voice: str
    language_code: str
    model: str
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

