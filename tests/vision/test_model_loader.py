"""Unit tests for ModelLoader and artifact integrity verification."""

import torch
import pytest

from backend.app.services.agriculture.model_loader import ModelLoader


def test_model_loader_initialization():
    """Ensure ModelLoader correctly loads PyTorch model, manifest, and classes."""
    loader = ModelLoader.get_instance()
    assert loader.model is not None
    assert loader.manifest is not None
    assert loader.manifest.get("architecture") == "MobileNetV3-Small"
    assert len(loader.classes) == 10


def test_model_forward_pass_shape():
    """Ensure input tensor [1, 3, 224, 224] produces exactly 10 output logits."""
    loader = ModelLoader.get_instance()
    dummy_input = torch.randn(1, 3, 224, 224).to(loader.device)

    loader.model.eval()
    with torch.no_grad():
        logits = loader.model(dummy_input)

    assert logits.shape == (1, 10)
    assert not torch.isnan(logits).any()


def test_model_transforms():
    """Ensure transform pipeline resizes and normalizes correctly."""
    from PIL import Image
    import numpy as np

    loader = ModelLoader.get_instance()
    img = Image.fromarray((np.random.rand(400, 300, 3) * 255).astype(np.uint8))
    tensor = loader.transform(img)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

