"""Pytest shared fixtures and configurations for AgriGuardian tests."""

import os
import shutil
import tempfile
import pytest
from backend.app.core.config import Settings
from backend.app.repositories.data_manager import DataManager
from backend.app.services.agriculture.rule_engine import DynamicRuleEngine
from backend.app.services.advisory.orchestrator import AgriGuardianSystem


@pytest.fixture
def temp_dir():
    """Create and tear down a temporary test directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir):
    """Provide isolated test settings pointing to temporary folders."""
    data_dir = os.path.join(temp_dir, "data")
    audio_dir = os.path.join(temp_dir, "audio")
    reports_dir = os.path.join(temp_dir, "reports")
    logs_dir = os.path.join(temp_dir, "logs")
    config_dir = os.path.join(temp_dir, "config")

    settings = Settings(
        app_name="AgriGuardianTest",
        app_env="test",
        debug=True,
        data_dir=data_dir,
        audio_temp_dir=audio_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        config_dir=config_dir,
        gnani_api_key="test_mock_gnani_key",
        openai_api_key="test_mock_openai_key",
    )
    settings.ensure_directories()
    return settings


@pytest.fixture
def test_data_manager(test_settings):
    """Provide isolated DataManager."""
    return DataManager(settings=test_settings)


@pytest.fixture
def test_rule_engine(test_settings):
    """Provide isolated RuleEngine."""
    return DynamicRuleEngine(settings=test_settings)


@pytest.fixture
def sample_wav_path():
    """Path to verified sample WAV fixture."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_audio.wav")
    if not os.path.exists(fixture_path):
        os.makedirs(os.path.dirname(fixture_path), exist_ok=True)
        import wave
        import struct
        with wave.open(fixture_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            data = struct.pack("<" + ("h" * 16000), *([0] * 16000))
            wf.writeframes(data)
    return fixture_path

