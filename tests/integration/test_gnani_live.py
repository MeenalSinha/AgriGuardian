"""Live gated integration tests against real Gnani AI endpoints (api.vachana.ai).

NOTE: These tests execute genuine network calls only when GNANI_API_KEY is present in the environment.
If credentials are absent or invalid, the test is explicitly reported as SKIPPED/BLOCKED.
"""

import os
import pytest
from backend.app.core.config import get_settings
from backend.app.integrations.gnani.prisma_stt import GnaniPrismaClient
from backend.app.integrations.gnani.timbre_tts import GnaniTimbreClient


@pytest.mark.integration
def test_live_gnani_prisma_stt(sample_wav_path):
    """Execute live Speech-to-Text inference on Gnani Prisma v2.5."""
    settings = get_settings()

    if not settings.is_gnani_configured:
        pytest.skip(
            "GNANI_API_KEY is not configured in environment or .env. "
            "Status: BLOCKED / SKIPPED (Live integration test requires active API credentials)."
        )

    client = GnaniPrismaClient(settings=settings)
    try:
        result = client.transcribe(sample_wav_path, language_code="hi-IN")
    except Exception as e:
        if "connect" in str(e).lower() or "network" in str(e).lower() or "timeout" in str(e).lower():
            pytest.skip(f"Live Gnani endpoint unreachable in current environment (Sandbox/Offline). Status: BLOCKED: {e}")
        raise

    assert result.model == settings.gnani_prisma_model
    assert result.request_id is not None
    print(f"Live Prisma STT result: success={result.success}, transcript='{result.transcript}', id={result.request_id}")


@pytest.mark.integration
def test_live_gnani_timbre_tts():
    """Execute live Text-to-Speech synthesis on Gnani Timbre v2.5."""
    settings = get_settings()

    if not settings.is_gnani_configured:
        pytest.skip(
            "GNANI_API_KEY is not configured in environment or .env. "
            "Status: BLOCKED / SKIPPED (Live integration test requires active API credentials)."
        )

    client = GnaniTimbreClient(settings=settings)
    try:
        result = client.synthesize(
            text="नमस्ते, एग्रीगार्डियन में आपका स्वागत है।",
            language_code="hi-IN",
            voice="Nalini",
        )
    except Exception as e:
        if "connect" in str(e).lower() or "network" in str(e).lower() or "timeout" in str(e).lower():
            pytest.skip(f"Live Gnani endpoint unreachable in current environment (Sandbox/Offline). Status: BLOCKED: {e}")
        raise

    assert result.success is True
    assert result.audio_file_path is not None
    assert os.path.exists(result.audio_file_path)
    assert os.path.getsize(result.audio_file_path) > 100
    print(f"Live Timbre TTS result: voice={result.voice}, path={result.audio_file_path}, bytes={len(result.audio_bytes)}")

