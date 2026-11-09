"""Unit tests for ImageQualityAssessor component."""

import cv2
import numpy as np
from PIL import Image
import pytest

from backend.app.services.agriculture.image_quality import ImageQualityAssessor


def test_quality_sharp_image():
    """Ensure high-contrast high-frequency image is rated sharp/good."""
    # Create checkerboard pattern with sharp edges
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[::16, :] = 255
    arr[:, ::16] = 255
    arr[64:192, 64:192] = [40, 180, 50]  # Green patch

    img = Image.fromarray(arr)
    result = ImageQualityAssessor.assess_quality(img)

    assert result.is_blurry is False
    assert result.blur_score > 100.0
    assert result.exposure_status == "Optimal"
    assert result.quality_tier in ("Good", "Acceptable")


def test_quality_severely_blurred_image():
    """Ensure Gaussian-blurred image triggers blur detection and advice."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[::32, :] = 200
    # Apply intense Gaussian blur
    blurred = cv2.GaussianBlur(arr, (31, 31), 10.0)

    img = Image.fromarray(blurred)
    result = ImageQualityAssessor.assess_quality(img)

    assert result.is_blurry is True
    assert result.blur_score < 75.0
    assert result.quality_tier == "Poor"
    assert "blurry" in " ".join(result.warnings).lower()
    assert "hold camera steady" in result.farmer_advice.lower()


def test_quality_under_exposed_dark_image():
    """Ensure nearly pitch-black photo triggers under-exposed warning."""
    # Very dark image
    arr = np.ones((256, 256, 3), dtype=np.uint8) * 15

    img = Image.fromarray(arr)
    result = ImageQualityAssessor.assess_quality(img)

    assert result.exposure_status == "Under-exposed"
    assert result.brightness_score < 40.0
    assert "too dark" in " ".join(result.warnings).lower()
    assert result.quality_tier == "Poor"


def test_quality_over_exposed_glare_image():
    """Ensure blown-out white image triggers over-exposed warning."""
    # Blown-out white image
    arr = np.ones((256, 256, 3), dtype=np.uint8) * 245

    img = Image.fromarray(arr)
    result = ImageQualityAssessor.assess_quality(img)

    assert result.exposure_status == "Over-exposed"
    assert result.brightness_score > 220.0
    assert "over-exposure" in " ".join(result.warnings).lower()
    assert result.quality_tier == "Poor"
