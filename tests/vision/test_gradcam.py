"""Unit tests for GradCAMExplainer component."""

import os
import torch
import numpy as np
from PIL import Image
import pytest

from backend.app.services.agriculture.explainability_service import GradCAMExplainer
from backend.app.services.agriculture.model_loader import ModelLoader


def test_gradcam_generation():
    """Ensure Grad-CAM generates valid heatmap overlay matching image dimensions."""
    loader = ModelLoader.get_instance()
    explainer = GradCAMExplainer(loader.model)

    # Synthetic test image
    img = Image.fromarray((np.random.rand(280, 280, 3) * 255).astype(np.uint8))
    tensor = loader.transform(img).unsqueeze(0).to(loader.device)

    overlay_rgb, file_path, base64_str = explainer.generate_heatmap(
        input_tensor=tensor,
        target_class_index=1,  # Early Blight
        original_image=img,
        alpha=0.5,
    )

    assert overlay_rgb is not None
    assert overlay_rgb.shape == (280, 280, 3)
    assert file_path is not None
    assert os.path.exists(file_path)
    assert base64_str is not None
    assert len(base64_str) > 100

    # Cleanup temporary test image
    try:
        os.remove(file_path)
    except Exception:
        pass
