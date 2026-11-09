"""Unit tests for ImageValidator component."""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest

from backend.app.services.agriculture.image_validator import ImageValidator


@pytest.fixture
def temp_images_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def test_validator_valid_jpeg(temp_images_dir):
    """Ensure a well-formed JPEG passes validation."""
    img_path = os.path.join(temp_images_dir, "test_valid.jpg")
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
    img.save(img_path, format="JPEG")

    result, pil_img = ImageValidator.validate_file(img_path)
    assert result.is_valid is True
    assert result.format == "JPEG"
    assert result.width == 256
    assert result.height == 256
    assert result.channels == 3
    assert pil_img is not None


def test_validator_valid_png(temp_images_dir):
    """Ensure a well-formed PNG passes validation."""
    img_path = os.path.join(temp_images_dir, "test_valid.png")
    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))
    img.save(img_path, format="PNG")

    result, pil_img = ImageValidator.validate_file(img_path)
    assert result.is_valid is True
    assert result.format == "PNG"
    assert result.width == 300
    assert result.height == 300
    assert pil_img is not None


def test_validator_missing_file():
    """Ensure missing file path returns invalid result."""
    result, pil_img = ImageValidator.validate_file("nonexistent_path_xyz123.jpg")
    assert result.is_valid is False
    assert "does not exist" in result.error_message
    assert pil_img is None


def test_validator_empty_file(temp_images_dir):
    """Ensure completely empty 0-byte file returns invalid result."""
    img_path = os.path.join(temp_images_dir, "empty.jpg")
    with open(img_path, "wb") as f:
        pass

    result, pil_img = ImageValidator.validate_file(img_path)
    assert result.is_valid is False
    assert "empty" in result.error_message
    assert pil_img is None


def test_validator_corrupt_file(temp_images_dir):
    """Ensure non-image garbage bytes return invalid result."""
    img_path = os.path.join(temp_images_dir, "corrupt.jpg")
    with open(img_path, "wb") as f:
        f.write(b"NOT_A_VALID_IMAGE_HEADER_DATA_123456789")

    result, pil_img = ImageValidator.validate_file(img_path)
    assert result.is_valid is False
    assert pil_img is None


def test_validator_too_small_resolution(temp_images_dir):
    """Ensure tiny icons (< 64px) are rejected for diagnostic safety."""
    img_path = os.path.join(temp_images_dir, "tiny.jpg")
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    img.save(img_path, format="JPEG")

    result, pil_img = ImageValidator.validate_file(img_path)
    assert result.is_valid is False
    assert "too small" in result.error_message
    assert pil_img is None
