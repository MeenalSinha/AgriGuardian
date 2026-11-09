"""Unit tests for Gnani Prisma v2.5 Speech-to-Text integration."""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    GnaniAudioFormatError,
    GnaniAuthError,
    GnaniRateLimitError,
    GnaniTimeoutError,
)
from backend.app.integrations.gnani.prisma_stt import GnaniPrismaClient


def test_prisma_missing_credentials(temp_dir, sample_wav_path):
    """Verify transcription fails with GnaniAuthError when GNANI_API_KEY is not set."""
    unconfigured_settings = Settings(
        gnani_api_key=None,
        audio_temp_dir=temp_dir,
    )
    client = GnaniPrismaClient(settings=unconfigured_settings)

    with pytest.raises(GnaniAuthError) as exc_info:
        client.transcribe(sample_wav_path)
    assert "GNANI_API_KEY" in str(exc_info.value)


def test_prisma_empty_audio_rejection(temp_dir, test_settings):
    """Verify transcription rejects empty audio before calling the network."""
    empty_file = os_join = os = __import__("os").path.join(temp_dir, "empty_voice.wav")
    with open(empty_file, "wb") as f:
        pass

    client = GnaniPrismaClient(settings=test_settings)
    with pytest.raises(GnaniAudioFormatError):
        client.transcribe(empty_file)


def test_prisma_successful_transcription(sample_wav_path, test_settings):
    """Verify parsing of valid Prisma STT API response."""
    client = GnaniPrismaClient(settings=test_settings)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "request_id": "test-req-001",
        "transcript": "मेरे टमाटर के पौधे सूख रहे हैं।",
        "model": "prisma-v2.5",
        "detected_language_code": "hi-IN",
        "processing_time": 0.42,
    }

    with patch.object(client, "_execute_request", return_value=mock_resp):
        res = client.transcribe(sample_wav_path, language_code="hi-IN")

        assert res.success is True
        assert res.transcript == "मेरे टमाटर के पौधे सूख रहे हैं।"
        assert res.language_code == "hi-IN"
        assert res.model == "prisma-v2.5"
        assert res.request_id == "test-req-001"


def test_prisma_empty_speech_response(sample_wav_path, test_settings):
    """Verify handling when audio contains no decipherable speech."""
    client = GnaniPrismaClient(settings=test_settings)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "request_id": "test-req-002",
        "transcript": "",
        "model": "prisma-v2.5",
    }

    with patch.object(client, "_execute_request", return_value=mock_resp):
        res = client.transcribe(sample_wav_path, language_code="hi-IN")
        assert res.success is False
        assert "no speech detected" in res.error.lower()


def test_prisma_auth_failure_handling(sample_wav_path, test_settings):
    """Verify HTTP 401 raises GnaniAuthError."""
    client = GnaniPrismaClient(settings=test_settings)

    with patch("httpx.Client.request") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid API key"
        mock_req.return_value = mock_resp

        with pytest.raises(GnaniAuthError):
            client.transcribe(sample_wav_path)

