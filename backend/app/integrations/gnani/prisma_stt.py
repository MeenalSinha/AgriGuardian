"""Gnani Prisma v2.5 Speech-to-Text (STT) Client Integration."""

import json
import os
import time
from typing import Optional
from backend.app.core.config import Settings, get_settings, SUPPORTED_LANGUAGES
from backend.app.core.exceptions import (
    GnaniAPIError,
    GnaniAudioFormatError,
    GnaniTranscriptionError,
)
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.audio_utils import validate_audio_file
from backend.app.integrations.gnani.client import GnaniBaseClient
from backend.app.schemas.speech import TranscriptionResult

logger = get_logger("GnaniPrisma")


class GnaniPrismaClient(GnaniBaseClient):
    """Client for Gnani Prisma v2.5 Automatic Speech Recognition (ASR)."""

    def __init__(self, settings: Optional[Settings] = None):
        super().__init__(settings)
        self.model = self.settings.gnani_prisma_model
        self.endpoint = self.settings.gnani_prisma_endpoint
        self.timeout = self.settings.gnani_prisma_timeout_seconds
        self.max_retries = self.settings.gnani_prisma_max_retries

    def transcribe(
        self,
        audio_file_path: str,
        language_code: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file into text using Gnani Prisma v2.5.

        Args:
            audio_file_path: Absolute or relative path to the audio file on disk.
            language_code: Target language code (e.g. 'hi-IN'). Defaults to configured default.

        Returns:
            TranscriptionResult containing recognized transcript, latency, and metadata.

        Raises:
            GnaniAuthError: If GNANI_API_KEY is not configured or rejected.
            GnaniAudioFormatError: If the audio file is missing, empty, or oversized.
            GnaniTranscriptionError: If transcription fails or returns an error.
        """
        lang = language_code or self.settings.gnani_prisma_default_language

        # Validate language code
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language code '{lang}' is not in catalog. Using '{self.settings.gnani_prisma_default_language}'")
            lang = self.settings.gnani_prisma_default_language

        # Validate audio file
        val_result = validate_audio_file(audio_file_path)
        if not val_result.is_valid:
            logger.error(f"Audio validation failed for {audio_file_path}: {val_result.error_message}")
            raise GnaniAudioFormatError(
                val_result.error_message or "Invalid audio file for transcription.",
                details={"audio_path": audio_file_path, "format": val_result.format},
            )

        start_time = time.time()
        filename = os.path.basename(audio_file_path)

        try:
            with open(audio_file_path, "rb") as f:
                file_bytes = f.read()

            files = {
                "audio_file": (filename, file_bytes, f"audio/{val_result.format}"),
            }
            data = {
                "language_code": lang,
                "model": self.model,
            }

            response = self._execute_request(
                method="POST",
                endpoint=self.endpoint,
                timeout=self.timeout,
                max_retries=self.max_retries,
                data=data,
                files=files,
            )

            latency = time.time() - start_time
            raw_json = response.json()

            # Parse transcript from Gnani Prisma response schema
            transcript = raw_json.get("transcript") or raw_json.get("text") or raw_json.get("transcription")
            if transcript is None:
                # Some API variants might return data in nested dict
                if "data" in raw_json and isinstance(raw_json["data"], dict):
                    transcript = raw_json["data"].get("transcript") or raw_json["data"].get("text")

            if not transcript or not str(transcript).strip():
                logger.warning(f"Prisma returned empty transcription for {filename}")
                return TranscriptionResult(
                    success=False,
                    transcript="",
                    language_code=raw_json.get("detected_language_code", lang),
                    model=raw_json.get("model", self.model),
                    request_id=raw_json.get("request_id"),
                    processing_time=raw_json.get("processing_time", latency),
                    raw_response=raw_json,
                    error="No speech detected in audio recording. Please speak clearly and try again.",
                )

            logger.info(
                f"Prisma transcription succeeded in {latency:.2f}s: '{str(transcript)[:50]}...' "
                f"(Request ID: {raw_json.get('request_id')})"
            )

            return TranscriptionResult(
                success=True,
                transcript=str(transcript).strip(),
                language_code=raw_json.get("detected_language_code", lang),
                model=raw_json.get("model", self.model),
                request_id=raw_json.get("request_id"),
                confidence=raw_json.get("confidence"),
                processing_time=raw_json.get("processing_time", latency),
                raw_response=raw_json,
            )

        except Exception as e:
            if isinstance(e, GnaniAPIError):
                raise
            logger.error(f"Unexpected error during Prisma transcription: {e}", exc_info=True)
            raise GnaniTranscriptionError(f"Transcription failed: {e}")

