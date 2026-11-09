"""Gnani Timbre v2.5 Text-to-Speech (TTS) Client Integration."""

import os
import time
from typing import Optional
from backend.app.core.config import Settings, get_settings, SUPPORTED_LANGUAGES, TIMBRE_V25_VOICES
from backend.app.core.exceptions import (
    GnaniAPIError,
    GnaniSpeechSynthesisError,
    ValidationError,
)
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.audio_utils import save_temporary_audio
from backend.app.integrations.gnani.client import GnaniBaseClient
from backend.app.schemas.speech import SpeechSynthesisResult

logger = get_logger("GnaniTimbre")


class GnaniTimbreClient(GnaniBaseClient):
    """Client for Gnani Timbre v2.5 Speech Synthesis."""

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.model = self.settings.gnani_timbre_model
        self.endpoint = self.settings.gnani_timbre_endpoint
        self.timeout = self.settings.gnani_timbre_timeout_seconds
        self.max_retries = self.settings.gnani_timbre_max_retries

    def get_recommended_voice(self, language_code: str) -> str:
        """Get the verified default voice for a given language code."""
        lang_info = SUPPORTED_LANGUAGES.get(language_code)
        if lang_info and "default_voice" in lang_info:
            return lang_info["default_voice"]
        return self.settings.gnani_timbre_default_voice

    def synthesize(
        self,
        text: str,
        language_code: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> SpeechSynthesisResult:
        """Synthesize agricultural response text into natural speech audio.

        Args:
            text: Text to synthesize into speech.
            language_code: BCP-47 language tag (e.g. 'hi-IN').
            voice: Name of the Timbre voice (e.g. 'Nalini', 'Kaveri').
            speed: Speech playback rate multiplier (default: 1.0).

        Returns:
            SpeechSynthesisResult with saved audio file path and metadata.

        Raises:
            ValidationError: If input text is empty.
            GnaniAuthError: If GNANI_API_KEY is not configured or rejected.
            GnaniSpeechSynthesisError: If synthesis fails.
        """
        if not text or not text.strip():
            raise ValidationError("Cannot synthesize empty text into speech.")

        clean_text = text.strip()
        lang = language_code or self.settings.gnani_timbre_default_language
        selected_voice = voice or self.get_recommended_voice(lang)
        speech_speed = speed or self.settings.gnani_timbre_speed

        logger.info(f"Synthesizing {len(clean_text)} chars in {lang} with voice '{selected_voice}' using Timbre v2.5")

        payload = {
            "text": clean_text,
            "voice": selected_voice,
            "model": self.model,
            "language": lang,
            "speed": speech_speed,
            "audio_config": {
                "sample_rate": self.settings.gnani_timbre_sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": self.settings.gnani_timbre_encoding,
                "container": self.settings.gnani_timbre_container,
            },
        }

        start_time = time.time()
        try:
            response = self._execute_request(
                method="POST",
                endpoint=self.endpoint,
                timeout=self.timeout,
                max_retries=self.max_retries,
                headers={"Content-Type": "application/json"},
                json_body=payload,
            )

            audio_bytes = response.content
            if not audio_bytes or len(audio_bytes) < 100:
                logger.error(f"Timbre returned empty audio payload ({len(audio_bytes)} bytes)")
                raise GnaniSpeechSynthesisError("Received empty or corrupt audio response from Gnani Timbre.")

            # Save temporary audio file
            ext = f".{self.settings.gnani_timbre_container}"
            saved_path = save_temporary_audio(audio_bytes, prefix=f"timbre_{selected_voice}", ext=ext)

            duration = time.time() - start_time
            logger.info(f"Timbre TTS completed in {duration:.2f}s, saved {len(audio_bytes)} bytes to {saved_path}")

            return SpeechSynthesisResult(
                success=True,
                audio_file_path=saved_path,
                audio_bytes=audio_bytes,
                container=self.settings.gnani_timbre_container,
                voice=selected_voice,
                language_code=lang,
                model=self.model,
                duration_seconds=duration,
            )

        except Exception as e:
            if isinstance(e, GnaniAPIError):
                raise
            logger.error(f"Error during Timbre speech synthesis: {e}", exc_info=True)
            raise GnaniSpeechSynthesisError(f"Timbre speech synthesis failed: {e}")

