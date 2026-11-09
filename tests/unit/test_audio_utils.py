"""Unit tests for audio format validation, safe temporary files, and cleanup."""

import os
import time
import pytest
from backend.app.integrations.gnani.audio_utils import (
    validate_audio_file,
    detect_audio_format,
    save_temporary_audio,
    cleanup_temporary_files,
)


def test_detect_audio_format():
    """Verify format detection based on header bytes."""
    assert detect_audio_format(b"RIFF\x00\x00\x00\x00WAVE") == ".wav"
    assert detect_audio_format(b"ID3\x03\x00\x00\x00") == ".mp3"
    assert detect_audio_format(b"OggS\x00\x02\x00\x00") == ".ogg"
    assert detect_audio_format(b"fLaC\x00\x00\x00\x22") == ".flac"


def test_validate_audio_file(sample_wav_path):
    """Verify validation of genuine WAV file."""
    res = validate_audio_file(sample_wav_path)
    assert res.is_valid is True
    assert res.format == "wav"
    assert res.file_size_bytes > 0


def test_validate_missing_audio_file():
    """Verify validation fails for non-existent file."""
    res = validate_audio_file("missing_audio_file.wav")
    assert res.is_valid is False
    assert "does not exist" in res.error_message


def test_validate_empty_audio_file(temp_dir):
    """Verify validation fails for 0-byte file."""
    empty_file = os.path.join(temp_dir, "empty.wav")
    with open(empty_file, "wb") as f:
        pass

    res = validate_audio_file(empty_file)
    assert res.is_valid is False
    assert "empty" in res.error_message.lower()


def test_save_and_cleanup_temporary_audio(temp_dir, test_settings):
    """Verify saving temp audio and cleaning up expired files."""
    dummy_bytes = b"RIFF" + (b"\x00" * 100)
    saved_path = save_temporary_audio(
        dummy_bytes, prefix="test_cleanup", ext=".wav", directory=test_settings.audio_temp_dir
    )
    assert os.path.exists(saved_path)

    # Force mtime into the past
    past_time = time.time() - 7200
    os.utime(saved_path, (past_time, past_time))

    cleaned = cleanup_temporary_files(test_settings.audio_temp_dir, max_age_seconds=3600)
    assert cleaned >= 1
    assert not os.path.exists(saved_path)
