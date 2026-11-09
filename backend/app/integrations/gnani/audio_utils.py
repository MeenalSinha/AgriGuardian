"""Audio processing utilities: validation, format inspection, safe temp files, and cleanup."""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.speech import AudioValidationResult

logger = get_logger("AudioUtils")

# Supported audio extensions and their expected MIME types for Gnani Prisma
SUPPORTED_AUDIO_FORMATS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/m4a",
    ".webm": "audio/webm",
}

# Magic bytes signature detection for common audio containers
AUDIO_SIGNATURES = [
    (b"RIFF", ".wav"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
    (b"\xff\xf2", ".mp3"),
    (b"OggS", ".ogg"),
    (b"fLaC", ".flac"),
    (b"\x1a\x45\xdf\xa3", ".webm"),  # Matroska / WebM
]


def detect_audio_format(file_bytes: bytes, filename: Optional[str] = None) -> str:
    """Detect audio format from bytes signature or fallback to filename extension."""
    if file_bytes:
        prefix = file_bytes[:12]
        for signature, ext in AUDIO_SIGNATURES:
            if prefix.startswith(signature):
                return ext

        # Check MP4/M4A ftyp
        if len(prefix) >= 8 and b"ftyp" in prefix[4:8]:
            return ".m4a"

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in SUPPORTED_AUDIO_FORMATS:
            return ext

    return ".wav"


def validate_audio_file(file_path: str) -> AudioValidationResult:
    """Validate an audio file on disk before sending to Gnani Prisma."""
    settings = get_settings()

    if not os.path.exists(file_path):
        return AudioValidationResult(
            is_valid=False,
            format="unknown",
            file_size_bytes=0,
            error_message=f"Audio file does not exist: {file_path}",
        )

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return AudioValidationResult(
            is_valid=False,
            format="empty",
            file_size_bytes=0,
            error_message="Audio file is empty (0 bytes). Please record or upload valid audio.",
        )

    if file_size > settings.max_audio_size_bytes:
        max_mb = settings.max_audio_size_bytes / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return AudioValidationResult(
            is_valid=False,
            format="oversized",
            file_size_bytes=file_size,
            error_message=f"Audio file exceeds maximum size limit of {max_mb:.1f} MB (Actual: {actual_mb:.1f} MB).",
        )

    with open(file_path, "rb") as f:
        header = f.read(64)

    ext = detect_audio_format(header, file_path)

    return AudioValidationResult(
        is_valid=True,
        format=ext.lstrip("."),
        file_size_bytes=file_size,
    )


def save_temporary_audio(
    file_bytes: bytes, prefix: str = "gnani_audio", ext: str = ".wav", directory: Optional[str] = None
) -> str:
    """Save audio bytes to a secure temporary file."""
    settings = get_settings()
    target_dir = directory or settings.audio_temp_dir
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    if not ext.startswith("."):
        ext = f".{ext}"

    unique_id = uuid.uuid4().hex[:10]
    filename = f"{prefix}_{unique_id}{ext}"
    target_path = os.path.join(target_dir, filename)

    with open(target_path, "wb") as f:
        f.write(file_bytes)

    logger.debug(f"Saved temporary audio file: {target_path} ({len(file_bytes)} bytes)")
    return target_path


def cleanup_temporary_files(directory: str, max_age_seconds: int = 3600) -> int:
    """Clean up old temporary audio or report files to protect privacy and storage."""
    deleted_count = 0
    now = time.time()

    if not os.path.exists(directory):
        return 0

    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                age = now - entry.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        os.remove(entry.path)
                        deleted_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove temporary file {entry.path}: {e}")
    except Exception as e:
        logger.error(f"Error during temporary file cleanup in {directory}: {e}")

    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} temporary files from {directory}")

    return deleted_count
