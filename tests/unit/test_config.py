"""Unit tests for centralized configuration and catalogs."""

import os
import pytest
from backend.app.core.config import Settings, SUPPORTED_LANGUAGES, TIMBRE_V25_VOICES


def test_settings_defaults():
    """Verify default settings values and properties."""
    settings = Settings()
    assert settings.app_name == "AgriGuardian"
    assert settings.gnani_api_base_url == "https://api.vachana.ai"
    assert settings.gnani_prisma_model == "prisma-v2.5"
    assert settings.gnani_timbre_model == "timbre-v2.5"
    assert settings.gnani_prisma_endpoint == "/stt/v3"
    assert settings.gnani_timbre_endpoint == "/api/v1/tts/inference"


def test_supported_languages_catalog():
    """Verify catalog covers 10 Indian languages with default voices."""
    assert len(SUPPORTED_LANGUAGES) >= 10
    assert "hi-IN" in SUPPORTED_LANGUAGES
    assert "en-IN" in SUPPORTED_LANGUAGES
    assert "ta-IN" in SUPPORTED_LANGUAGES
    assert "te-IN" in SUPPORTED_LANGUAGES
    assert "kn-IN" in SUPPORTED_LANGUAGES

    # Each language must have verified metadata
    for code, info in SUPPORTED_LANGUAGES.items():
        assert "name" in info
        assert "native_name" in info
        assert "default_voice" in info
        assert info["stt_available"] is True
        assert info["tts_available"] is True


def test_timbre_voices_catalog():
    """Verify Timbre v2.5 voice catalog contains verified voices."""
    assert "Nalini" in TIMBRE_V25_VOICES
    assert "Deepak" in TIMBRE_V25_VOICES
    assert "Kaveri" in TIMBRE_V25_VOICES
    assert "Arjun" in TIMBRE_V25_VOICES


def test_ensure_directories(temp_dir):
    """Verify that settings creates all necessary directories."""
    settings = Settings(
        data_dir=os.path.join(temp_dir, "custom_data"),
        audio_temp_dir=os.path.join(temp_dir, "custom_audio"),
        reports_dir=os.path.join(temp_dir, "custom_reports"),
        logs_dir=os.path.join(temp_dir, "custom_logs"),
        config_dir=os.path.join(temp_dir, "custom_config"),
    )
    settings.ensure_directories()
    assert os.path.exists(settings.data_dir)
    assert os.path.exists(settings.audio_temp_dir)
    assert os.path.exists(settings.reports_dir)
    assert os.path.exists(settings.logs_dir)
    assert os.path.exists(settings.config_dir)


def test_is_gnani_configured():
    """Verify gnani configuration checks."""
    s1 = Settings(gnani_api_key=None)
    assert s1.is_gnani_configured is False

    s2 = Settings(gnani_api_key="your_gnani_api_key_here")
    assert s2.is_gnani_configured is False

    s3 = Settings(gnani_api_key="actual_key_12345")
    assert s3.is_gnani_configured is True

