"""Custom exception hierarchy for AgriGuardian."""

from typing import Optional, Any, Dict


class AgriGuardianException(Exception):
    """Base exception for all AgriGuardian errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgriGuardianException):
    """Raised when application or integration configuration is invalid or missing."""
    pass


class ValidationError(AgriGuardianException):
    """Raised when input data validation fails."""
    pass


class StorageError(AgriGuardianException):
    """Raised when file or database persistence operations fail."""
    pass


class AudioProcessingError(AgriGuardianException):
    """Raised when audio recording, conversion, or format validation fails."""
    pass


class IntegrationError(AgriGuardianException):
    """Base class for third-party service integration failures."""

    def __init__(
        self,
        message: str,
        service: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details)
        self.service = service
        self.status_code = status_code


class WeatherAPIError(IntegrationError):
    """Raised when weather API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, service="Open-Meteo", status_code=status_code, details=details)


class MarketAPIError(IntegrationError):
    """Raised when market data retrieval fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, service="MarketIntelligence", status_code=status_code, details=details)


class LLMAPIError(IntegrationError):
    """Raised when LLM reasoning service call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, service="LLMAdvisor", status_code=status_code, details=details)


class GnaniAPIError(IntegrationError):
    """Base exception for Gnani AI integration failures."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, service="GnaniAI", status_code=status_code, details=details)


class GnaniAuthError(GnaniAPIError):
    """Raised when Gnani authentication fails (invalid or missing X-API-Key-ID)."""
    pass


class GnaniTimeoutError(GnaniAPIError):
    """Raised when Gnani API request times out."""
    pass


class GnaniRateLimitError(GnaniAPIError):
    """Raised when Gnani API returns rate limiting (429)."""
    pass


class GnaniAudioFormatError(GnaniAPIError):
    """Raised when audio format or duration violates Gnani requirements."""
    pass


class GnaniTranscriptionError(GnaniAPIError):
    """Raised when Gnani Prisma transcription returns an error or empty result."""
    pass


class GnaniSpeechSynthesisError(GnaniAPIError):
    """Raised when Gnani Timbre speech synthesis fails."""
    pass

