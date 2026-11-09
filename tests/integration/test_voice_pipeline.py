"""Integration test for the end-to-end voice-to-voice agricultural pipeline."""

import os
from unittest.mock import MagicMock, patch
import httpx
import pytest
from backend.app.schemas.conversation import ConversationSession
from backend.app.schemas.speech import TranscriptionResult
from backend.app.services.conversation.engine import ConversationEngine


def test_end_to_end_voice_pipeline(sample_wav_path, test_settings):
    """Verify complete end-to-end voice workflow:

    Audio Recording -> Prisma STT -> Transcript Review -> Contextual Reasoning -> Timbre TTS -> Audio Playback.
    """
    engine = ConversationEngine(settings=test_settings)
    session = ConversationSession(
        farmer_name="अनिता देवी",
        crop="Tomato",
        location="Bengaluru Rural",
        language="hi-IN",
    )

    # 1. Mock Prisma STT response
    mock_stt_resp = MagicMock(spec=httpx.Response)
    mock_stt_resp.status_code = 200
    mock_stt_resp.json.return_value = {
        "success": True,
        "request_id": "req-stt-pipeline-01",
        "transcript": "मेरे टमाटर के पत्तों पर कीड़े दिख रहे हैं।",
        "model": "prisma-v2.5",
        "detected_language_code": "hi-IN",
        "processing_time": 0.45,
    }

    # 2. Mock Timbre TTS response
    mock_tts_resp = MagicMock(spec=httpx.Response)
    mock_tts_resp.status_code = 200
    mock_tts_resp.content = b"RIFF" + (b"\x00" * 400)

    with patch.object(engine.prisma_client, "_execute_request", return_value=mock_stt_resp):
        with patch.object(engine.timbre_client, "_execute_request", return_value=mock_tts_resp):
            # Step A: Farmer records and submits audio to Prisma
            stt_result = engine.transcribe_audio(sample_wav_path, language_code="hi-IN")
            assert stt_result.success is True
            assert stt_result.transcript == "मेरे टमाटर के पत्तों पर कीड़े दिख रहे हैं।"

            # Step B: Farmer reviews and confirms transcript
            confirmed_text = stt_result.transcript  # Farmer can accept or edit

            # Step C: Submit to agricultural reasoning & Timbre speech synthesis
            turn, tts_result = engine.process_turn(
                session=session,
                user_query=confirmed_text,
                input_type="voice",
                original_transcript=stt_result.transcript,
                crop="Tomato",
                location="Bengaluru Rural",
                synthesize_voice=True,
            )

            # Step D: Verify generated response & audio
            assert turn.role == "user"
            assert turn.input_type == "voice"
            assert turn.intent == "disease"
            assert turn.response_text is not None
            assert turn.audio_available is True
            assert turn.audio_file_path is not None
            assert os.path.exists(turn.audio_file_path)
            assert tts_result.success is True
            assert tts_result.voice == "Nalini"
            assert tts_result.model == "timbre-v2.5"

            # Step E: Verify session state history
            assert len(session.turns) == 1
            assert session.turns[0].confirmed_text == confirmed_text

