"""Unit tests for PlantDiseaseDetector pipeline."""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest

from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector


@pytest.fixture
def sample_leaf_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    # Create synthetic leaf image
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :] = [45, 150, 55]  # Green leaf
    arr[100:160, 100:160] = [80, 50, 30]  # Brown spot
    img = Image.fromarray(arr)
    img.save(path, format="JPEG")

    yield path

    if os.path.exists(path):
        os.remove(path)


def test_detector_not_simulated(sample_leaf_image):
    """Ensure is_simulated is explicitly False and real PyTorch inference executed."""
    detector = PlantDiseaseDetector()
    diag = detector.diagnose(sample_leaf_image)

    assert diag.is_simulated is False
    assert diag.inference_time_ms > 0
    assert len(diag.top_predictions) == 3
    assert diag.quality_assessment is not None
    assert diag.severity_assessment is not None
    assert diag.management_guidelines is not None


def test_detector_missing_image():
    """Ensure missing image returns invalid result cleanly."""
    detector = PlantDiseaseDetector()
    diag = detector.diagnose("missing_leaf_file_123.jpg")

    assert diag.disease_id == "Invalid_Image"
    assert diag.is_confident is False
    assert diag.is_uncertain is True
    assert diag.confidence == 0.0


def test_detector_predict_backward_compatibility(sample_leaf_image):
    """Ensure legacy predict() returns expected dictionary format."""
    detector = PlantDiseaseDetector()
    res = detector.predict(sample_leaf_image)

    assert "disease" in res
    assert "confidence" in res
    assert "top_3" in res
    assert "severity" in res
    assert "recommendation" in res
    assert res["is_simulated"] is False
    assert "diagnosis_result" in res
