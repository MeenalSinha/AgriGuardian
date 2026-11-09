"""Unit tests for SeverityAssessor decoupling confidence from physical damage."""

import numpy as np
from PIL import Image
import pytest

from backend.app.services.agriculture.severity_service import SeverityAssessor


def test_severity_healthy_leaf():
    """Ensure healthy foliage yields 0.0% damage and 'Healthy' stage."""
    # Pure green image
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = [34, 139, 34]  # Forest green

    img = Image.fromarray(arr)
    result = SeverityAssessor.calculate_severity(img, disease_name="Healthy Foliage", is_healthy=True)

    assert result.severity_score_percent == 0.0
    assert result.severity_stage == "Healthy"
    assert "no visible pathological lesions" in result.clinical_summary


def test_severity_diseased_leaf_with_lesions():
    """Ensure diseased leaf with dark necrotic patches calculates measurable severity."""
    # Green leaf background with dark brown/black necrotic spots
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = [50, 160, 60]  # Green leaf tissue

    # Add dark necrotic lesion circle
    arr[60:120, 60:120] = [30, 20, 15]  # Dark brown necrotic patch

    img = Image.fromarray(arr)
    result = SeverityAssessor.calculate_severity(img, disease_name="Early Blight", is_healthy=False)

    assert result.severity_score_percent > 0.0
    assert result.severity_stage in ("Mild", "Moderate", "Severe")
    assert result.lesion_area_pixels > 0


def test_severity_with_weather_context():
    """Ensure ambient weather factors elevate environmental multiplier without falsifying visual area."""
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    arr[:, :] = [50, 160, 60]
    arr[80:110, 80:110] = [30, 20, 15]

    img = Image.fromarray(arr)
    weather_high_risk = {"rain_recent": True, "humidity_high": True}

    result = SeverityAssessor.calculate_severity(
        img, disease_name="Late Blight", is_healthy=False, weather_context=weather_high_risk
    )

    assert result.environmental_risk_multiplier > 1.0
    assert len(result.environmental_warnings) >= 2
