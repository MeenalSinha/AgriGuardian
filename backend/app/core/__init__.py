"""Core module containing configuration, logging, and exceptions."""

from backend.app.core.config import Settings, get_settings, SUPPORTED_LANGUAGES, TIMBRE_V25_VOICES
from backend.app.core.logging import setup_logging, get_logger
from backend.app.core.exceptions import (
    AgriGuardianException,
    ConfigurationError,
    ValidationError,
    StorageError,
    AudioProcessingError,
    IntegrationError,
    WeatherAPIError,
    MarketAPIError,
    LLMAPIError,
    GnaniAPIError,
    GnaniAuthError,
    GnaniTimeoutError,
    GnaniRateLimitError,
    GnaniAudioFormatError,
    GnaniTranscriptionError,
    GnaniSpeechSynthesisError,
)

__all__ = [
    "Settings",
    "get_settings",
    "SUPPORTED_LANGUAGES",
    "TIMBRE_V25_VOICES",
    "setup_logging",
    "get_logger",
    "AgriGuardianException",
    "ConfigurationError",
    "ValidationError",
    "StorageError",
    "AudioProcessingError",
    "IntegrationError",
    "WeatherAPIError",
    "MarketAPIError",
    "LLMAPIError",
    "GnaniAPIError",
    "GnaniAuthError",
    "GnaniTimeoutError",
    "GnaniRateLimitError",
    "GnaniAudioFormatError",
    "GnaniTranscriptionError",
    "GnaniSpeechSynthesisError",
]

