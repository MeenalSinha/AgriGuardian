"""Unit tests for the Plant Disease Detector and severity scoring."""

import os
import pytest
from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector


def test_disease_detector_initialization(test_settings):
    """Verify disease detector initializes properly."""
    detector = PlantDiseaseDetector(settings=test_settings)
    assert len(detector.class_names) == 10
    assert "Early Blight" in detector.disease_severity


def test_severity_calculation(test_settings):
    """Verify compound severity score calculation with weather factors."""
    detector = PlantDiseaseDetector(settings=test_settings)

    # High humidity + recent rain elevates severity score
    weather = {"rain_recent": True, "humidity_high": True, "temp_high": False, "temp_very_high": False}
    sev = detector.calculate_severity(disease="Early Blight", confidence=0.87, weather=weather)

    assert "severity" in sev
    assert "final_score" in sev
    assert "weather_factors" in sev
    assert len(sev["weather_factors"]) >= 2
    assert sev["severity"] in ("High", "Critical")


def test_prediction_missing_image(test_settings):
    """Verify prediction handles missing image file gracefully without unhandled crashes."""
    detector = PlantDiseaseDetector(settings=test_settings)
    result = detector.predict("non_existent_image_123.jpg")

    assert result["disease"] == "Unable to detect"
    assert result["confidence"] == 0.0
    assert "recommendation" in result

