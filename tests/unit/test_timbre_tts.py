"""Unit tests for Gnani Timbre v2.5 Text-to-Speech integration."""

import os
from unittest.mock import MagicMock, patch
import httpx
import pytest
from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    GnaniAuthError,
    GnaniSpeechSynthesisError,
    ValidationError,
)
from backend.app.integrations.gnani.timbre_tts import GnaniTimbreClient


def test_timbre_empty_text_rejection(test_settings):
    """Verify empty text cannot be synthesized."""
    client = GnaniTimbreClient(settings=test_settings)
    with pytest.raises(ValidationError):
        client.synthesize("")


def test_timbre_voice_recommendation(test_settings):
    """Verify voice mapping for languages."""
    client = GnaniTimbreClient(settings=test_settings)
    assert client.get_recommended_voice("hi-IN") == "Nalini"
    assert client.get_recommended_voice("en-IN") == "Kaveri"


def test_timbre_missing_credentials(temp_dir):
    """Verify synthesis fails with GnaniAuthError when key is missing."""
    unconfigured_settings = Settings(
        gnani_api_key=None,
        audio_temp_dir=temp_dir,
    )
    client = GnaniTimbreClient(settings=unconfigured_settings)
    with pytest.raises(GnaniAuthError):
        client.synthesize("नमस्कार")


def test_timbre_successful_synthesis(test_settings):
    """Verify parsing and file creation of valid Timbre TTS response."""
    client = GnaniTimbreClient(settings=test_settings)

    dummy_audio = b"RIFF" + (b"\x00" * 500)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.content = dummy_audio

    with patch.object(client, "_execute_request", return_value=mock_resp):
        res = client.synthesize(
            text="आपकी गेहूं की फसल में नमी की कमी है।",
            language_code="hi-IN",
            voice="Nalini",
        )

        assert res.success is True
        assert res.voice == "Nalini"
        assert res.language_code == "hi-IN"
        assert res.model == "timbre-v2.5"
        assert res.audio_file_path is not None
        assert os.path.exists(res.audio_file_path)


def test_timbre_empty_audio_payload_rejection(test_settings):
    """Verify error when provider returns 0 bytes audio."""
    client = GnaniTimbreClient(settings=test_settings)

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.content = b""

    with patch.object(client, "_execute_request", return_value=mock_resp):
        with pytest.raises(GnaniSpeechSynthesisError):
            client.synthesize("परीक्षण संदेश")

