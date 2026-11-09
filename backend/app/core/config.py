"""Centralized application and integration configuration for AgriGuardian."""

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Supported Indian Language Catalog (verified against Gnani Prisma & Timbre v2.5)
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "hi-IN": {
        "code": "hi-IN",
        "short_code": "hi",
        "name": "Hindi",
        "native_name": "हिन्दी",
        "default_voice": "Nalini",
        "fallback_voice": "Deepak",
        "tts_available": True,
        "stt_available": True,
    },
    "en-IN": {
        "code": "en-IN",
        "short_code": "en",
        "name": "English (India)",
        "native_name": "English (India)",
        "default_voice": "Kaveri",
        "fallback_voice": "Arjun",
        "tts_available": True,
        "stt_available": True,
    },
    "ta-IN": {
        "code": "ta-IN",
        "short_code": "ta",
        "name": "Tamil",
        "native_name": "தமிழ்",
        "default_voice": "Kaveri",
        "fallback_voice": "Nalini",
        "tts_available": True,
        "stt_available": True,
    },
    "te-IN": {
        "code": "te-IN",
        "short_code": "te",
        "name": "Telugu",
        "native_name": "తెలుగు",
        "default_voice": "Kaveri",
        "fallback_voice": "Nalini",
        "tts_available": True,
        "stt_available": True,
    },
    "kn-IN": {
        "code": "kn-IN",
        "short_code": "kn",
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "default_voice": "Kaveri",
        "fallback_voice": "Nalini",
        "tts_available": True,
        "stt_available": True,
    },
    "ml-IN": {
        "code": "ml-IN",
        "short_code": "ml",
        "name": "Malayalam",
        "native_name": "മലയാളം",
        "default_voice": "Kaveri",
        "fallback_voice": "Nalini",
        "tts_available": True,
        "stt_available": True,
    },
    "mr-IN": {
        "code": "mr-IN",
        "short_code": "mr",
        "name": "Marathi",
        "native_name": "मराठी",
        "default_voice": "Nalini",
        "fallback_voice": "Deepak",
        "tts_available": True,
        "stt_available": True,
    },
    "pa-IN": {
        "code": "pa-IN",
        "short_code": "pa",
        "name": "Punjabi",
        "native_name": "ਪੰਜਾਬੀ",
        "default_voice": "Nalini",
        "fallback_voice": "Deepak",
        "tts_available": True,
        "stt_available": True,
    },
    "bn-IN": {
        "code": "bn-IN",
        "short_code": "bn",
        "name": "Bengali",
        "native_name": "বাংলা",
        "default_voice": "Nalini",
        "fallback_voice": "Deepak",
        "tts_available": True,
        "stt_available": True,
    },
    "gu-IN": {
        "code": "gu-IN",
        "short_code": "gu",
        "name": "Gujarati",
        "native_name": "ગુજરાતી",
        "default_voice": "Nalini",
        "fallback_voice": "Deepak",
        "tts_available": True,
        "stt_available": True,
    },
}

# Verified Timbre v2.5 Voice Catalog (Model-Specific)
TIMBRE_V25_VOICES: Dict[str, Dict[str, str]] = {
    "Nalini": {"gender": "female", "primary_language": "hi-IN", "style": "natural"},
    "Deepak": {"gender": "male", "primary_language": "hi-IN", "style": "natural"},
    "Kaveri": {"gender": "female", "primary_language": "en-IN", "style": "natural"},
    "Arjun": {"gender": "male", "primary_language": "en-IN", "style": "natural"},
}


class Settings(BaseSettings):
    """Central configuration validated through environment variables and local .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application Settings ---
    app_name: str = "AgriGuardian"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # --- AI & LLM Settings ---
    llm_provider: str = "openai"
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 600
    llm_timeout_seconds: float = 15.0
    llm_max_retries: int = 2

    # --- Gnani AI Settings ---
    gnani_api_base_url: str = "https://api.vachana.ai"
    gnani_api_key: Optional[str] = Field(default=None, description="Gnani AI X-API-Key-ID")

    # Gnani Prisma v2.5 Speech-to-Text (STT)
    gnani_prisma_model: str = "prisma-v2.5"
    gnani_prisma_endpoint: str = "/stt/v3"
    gnani_prisma_default_language: str = "hi-IN"
    gnani_prisma_timeout_seconds: float = 20.0
    gnani_prisma_max_retries: int = 2

    # Gnani Timbre v2.5 Text-to-Speech (TTS)
    gnani_timbre_model: str = "timbre-v2.5"
    gnani_timbre_endpoint: str = "/api/v1/tts/inference"
    gnani_timbre_default_voice: str = "Nalini"
    gnani_timbre_default_language: str = "hi-IN"
    gnani_timbre_speed: float = 1.0
    gnani_timbre_sample_rate: int = 48000
    gnani_timbre_container: str = "wav"
    gnani_timbre_encoding: str = "linear_pcm"
    gnani_timbre_timeout_seconds: float = 25.0
    gnani_timbre_max_retries: int = 2

    # --- Weather API Settings ---
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"
    default_latitude: float = 28.6139
    default_longitude: float = 77.2090
    default_location_name: str = "Delhi, India"
    weather_cache_ttl_seconds: int = 3600
    weather_timeout_seconds: float = 8.0

    # --- Market Intelligence Settings ---
    market_cache_ttl_seconds: int = 1800
    market_timeout_seconds: float = 5.0
    data_gov_in_api_key: Optional[str] = Field(default=None, description="OGD India data.gov.in API Key")
    enam_api_key: Optional[str] = Field(default=None, description="Institutional e-NAM APMC API Key")

    # --- Phase 6: Farmer Authentication & Security Settings ---
    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32),
        description="Cryptographic HMAC signing key for JWT session tokens (loaded from JWT_SECRET_KEY or generated dynamically)",
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    farm_db_path: str = "data/farm_guardian.db"

    # --- Storage and File System Paths ---
    data_dir: str = "data"
    audio_temp_dir: str = "audio"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    config_dir: str = "config"
    max_audio_size_bytes: int = 10 * 1024 * 1024  # 10 MB limit
    max_audio_duration_seconds: int = 60
    cleanup_temp_files: bool = True

    # --- Feature Flags ---
    feature_voice_assistant: bool = True
    feature_gradcam: bool = True
    feature_pdf_export: bool = True

    @property
    def is_gnani_configured(self) -> bool:
        """Check whether Gnani credentials are provided."""
        return bool(self.gnani_api_key and self.gnani_api_key.strip() and not self.gnani_api_key.startswith("your_"))

    @property
    def is_llm_configured(self) -> bool:
        """Check whether OpenAI credentials are provided."""
        return bool(self.openai_api_key and self.openai_api_key.strip() and not self.openai_api_key.startswith("your_"))

    def ensure_directories(self) -> None:
        """Create required runtime storage directories."""
        for dir_path in [
            self.data_dir,
            self.audio_temp_dir,
            self.reports_dir,
            self.logs_dir,
            self.config_dir,
            "temp",
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached global settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings

