"""Image validation service for plant disease diagnosis."""

import os
from typing import Optional, Tuple
from PIL import Image, ImageOps
from backend.app.core.logging import get_logger
from backend.app.schemas.disease import ImageValidationResult

logger = get_logger("ImageValidator")

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_DIMENSION_PX = 64
MAX_DIMENSION_PX = 8192

SUPPORTED_MIME_TYPES = {
    "image/jpeg": [b"\xFF\xD8\xFF"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/webp": [b"RIFF"],
}


class ImageValidator:
    """Validates uploaded or captured crop images for integrity and supported format."""

    @staticmethod
    def validate_file(image_path: str) -> Tuple[ImageValidationResult, Optional[Image.Image]]:
        """Validate an image file from disk path.

        Returns:
            Tuple of (ImageValidationResult, Optional[PIL.Image.Image])
        """
        if not os.path.exists(image_path):
            return (
                ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=0,
                    error_message=f"Image file does not exist at path: {image_path}",
                ),
                None,
            )

        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return (
                ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=0,
                    error_message="Image file is completely empty (0 bytes).",
                ),
                None,
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            return (
                ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=file_size,
                    error_message=f"Image file exceeds maximum allowable size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
                ),
                None,
            )

        # Inspect magic bytes
        detected_format = None
        try:
            with open(image_path, "rb") as f:
                header = f.read(16)
                if header.startswith(b"\xFF\xD8\xFF"):
                    detected_format = "JPEG"
                elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                    detected_format = "PNG"
                elif header.startswith(b"RIFF") and b"WEBP" in header:
                    detected_format = "WEBP"
        except Exception as e:
            logger.error(f"Failed to read image headers: {e}")
            return (
                ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=file_size,
                    error_message=f"Failed reading image header: {str(e)}",
                ),
                None,
            )

        if not detected_format:
            return (
                ImageValidationResult(
                    is_valid=False,
                    file_size_bytes=file_size,
                    error_message="Unsupported or unrecognized image header. Please provide a standard JPEG, PNG, or WEBP photograph.",
                ),
                None,
            )

        # Decode with PIL and apply EXIF orientation correction
        try:
            with Image.open(image_path) as raw_img:
                # Correct EXIF rotation if tag exists
                img = ImageOps.exif_transpose(raw_img)
                if img is None:
                    img = raw_img

                # Convert to standard RGB (drops alpha/palette)
                rgb_img = img.convert("RGB")
                width, height = rgb_img.size

                if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
                    return (
                        ImageValidationResult(
                            is_valid=False,
                            format=detected_format,
                            width=width,
                            height=height,
                            file_size_bytes=file_size,
                            error_message=f"Image resolution ({width}x{height}) is too small. Minimum dimension is {MIN_DIMENSION_PX}px.",
                        ),
                        None,
                    )

                if width > MAX_DIMENSION_PX or height > MAX_DIMENSION_PX:
                    return (
                        ImageValidationResult(
                            is_valid=False,
                            format=detected_format,
                            width=width,
                            height=height,
                            file_size_bytes=file_size,
                            error_message=f"Image resolution ({width}x{height}) exceeds maximum limit of {MAX_DIMENSION_PX}px.",
                        ),
                        None,
                    )

                validation_result = ImageValidationResult(
                    is_valid=True,
                    format=detected_format,
                    width=width,
                    height=height,
                    channels=3,
                    file_size_bytes=file_size,
                )
                return validation_result, rgb_img

        except Exception as e:
            logger.error(f"Image decompression failure: {e}", exc_info=True)
            return (
                ImageValidationResult(
                    is_valid=False,
                    format=detected_format,
                    file_size_bytes=file_size,
                    error_message=f"Image is corrupt or failed decompression: {str(e)}",
                ),
                None,
            )

