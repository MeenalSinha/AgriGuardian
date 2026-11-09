"""Explainable AI (XAI) service delivering genuine Grad-CAM visual heatmaps."""

import os
import base64
import tempfile
import uuid
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger("GradCAM")


class GradCAMExplainer:
    """Computes genuine Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.features[-1]
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        self._cleanup_hooks()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains gradients with respect to layer output
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h_fwd, h_bwd])

    def _cleanup_hooks(self) -> None:
        """Remove active PyTorch hooks."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_index: int,
        original_image: Image.Image,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
        """Compute Grad-CAM activation heatmap and blend with original leaf photograph.

        Args:
            input_tensor: Preprocessed tensor [1, 3, 224, 224]
            target_class_index: Class index to explain
            original_image: Original PIL RGB image
            alpha: Heatmap blend opacity (0.0 to 1.0)

        Returns:
            Tuple of (overlay_rgb_array, file_path_on_disk, base64_png_string)
        """
        self._register_hooks()
        try:
            # Enable gradient calculation for Grad-CAM backward pass
            self.model.zero_grad()
            input_tensor = input_tensor.clone().requires_grad_(True)

            logits = self.model(input_tensor)
            if logits.shape[1] <= target_class_index:
                raise ValueError(f"Target class {target_class_index} exceeds logit dimension {logits.shape[1]}")

            target_score = logits[0, target_class_index]
            target_score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture gradients or activations in hook.")

            # 1. Global Average Pooling over spatial dimensions (H, W)
            # Gradients shape: [1, C, H, W]
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

            # 2. Weighted linear combination of feature activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

            # 3. Rectification via ReLU (keep positive contributors to class score)
            cam = F.relu(cam)

            # 4. Normalize to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            cam_min, cam_max = np.min(cam), np.max(cam)
            if cam_max - cam_min > 1e-7:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            # 5. Resize CAM to match original image dimensions
            orig_w, orig_h = original_image.size
            cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # 6. Apply colormap
            cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # 7. Blend with original image
            original_rgb = np.array(original_image)
            overlay_rgb = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

            # 8. Save to disk in temp dir
            settings = get_settings()
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            filename = f"gradcam_{uuid.uuid4().hex[:10]}.png"
            output_path = os.path.join(temp_dir, filename)

            # Save as PNG
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, overlay_bgr)

            # 9. Encode to Base64 string for direct frontend embedding
            _, buffer = cv2.imencode(".png", overlay_bgr)
            base64_str = base64.b64encode(buffer).decode("utf-8")

            return overlay_rgb, output_path, base64_str

        except Exception as e:
            logger.error(f"Grad-CAM computation failed: {e}", exc_info=True)
            return np.array(original_image), None, None
        finally:
            self._cleanup_hooks()

