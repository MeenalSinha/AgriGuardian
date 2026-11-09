"""Model loader and runtime manager for AgriGuardian vision models."""

import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from backend.app.core.config import get_settings
from backend.app.core.exceptions import ConfigurationError
from backend.app.core.logging import get_logger

logger = get_logger("ModelLoader")


class ModelLoader:
    """Manages model artifact lifecycle, verification, and PyTorch inference pipeline."""

    _instance: Optional["ModelLoader"] = None

    def __init__(self, model_path: Optional[str] = None, manifest_path: Optional[str] = None):
        settings = get_settings()
        # Resolve repository root
        # __file__ is backend/app/services/agriculture/model_loader.py
        current = os.path.abspath(__file__)
        for _ in range(5):
            current = os.path.dirname(current)
        repo_root = current

        candidate_dirs = [
            os.path.join(repo_root, "models"),
            os.path.join(repo_root, "backend", "models"),
            os.path.abspath("models"),
        ]

        resolved_model_path = model_path
        resolved_manifest_path = manifest_path

        if not resolved_manifest_path:
            for c_dir in candidate_dirs:
                c_manifest = os.path.join(c_dir, "tomato_disease_manifest.json")
                if os.path.exists(c_manifest):
                    resolved_manifest_path = c_manifest
                    break
            if not resolved_manifest_path:
                resolved_manifest_path = os.path.join(candidate_dirs[0], "tomato_disease_manifest.json")

        if not resolved_model_path:
            for c_dir in candidate_dirs:
                c_model = os.path.join(c_dir, "tomato_disease_model.pth")
                if os.path.exists(c_model):
                    resolved_model_path = c_model
                    break
            if not resolved_model_path:
                resolved_model_path = os.path.join(candidate_dirs[0], "tomato_disease_model.pth")

        self.model_path = resolved_model_path
        self.manifest_path = resolved_manifest_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.manifest: Dict[str, Any] = {}
        self.classes: List[Dict[str, Any]] = []
        self.transform: Optional[transforms.Compose] = None
        self._load_artifacts()

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Load manifest, verify checksum, and instantiate PyTorch model in eval mode."""
        if not os.path.exists(self.manifest_path):
            raise ConfigurationError(f"Model manifest not found at: {self.manifest_path}")

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.classes = self.manifest.get("classes", [])
        except Exception as e:
            raise ConfigurationError(f"Failed to parse model manifest: {e}")

        if not os.path.exists(self.model_path):
            raise ConfigurationError(f"Model checkpoint weights not found at: {self.model_path}")

        # Verify integrity checksum
        expected_checksum = self.manifest.get("file_metadata", {}).get("sha256_checksum")
        if expected_checksum:
            sha256 = hashlib.sha256()
            with open(self.model_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            actual_checksum = sha256.hexdigest()
            if actual_checksum != expected_checksum:
                logger.warning(
                    f"Checksum mismatch for {self.model_path}. Expected: {expected_checksum}, got: {actual_checksum}"
                )

        # Build PyTorch architecture
        logger.info(f"Loading MobileNetV3-Small classifier on {self.device}...")
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        num_classes = len(self.classes) if self.classes else 10
        model.classifier[3] = nn.Linear(in_features, num_classes)

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Build inference transform pipeline matching manifest
        input_spec = self.manifest.get("input_specification", {})
        mean = input_spec.get("mean", [0.485, 0.456, 0.406])
        std = input_spec.get("std", [0.229, 0.224, 0.225])
        resize_dim = tuple(input_spec.get("resize_dimension", [256, 256]))
        crop_dim = tuple(input_spec.get("crop_dimension", [224, 224]))

        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.CenterCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.model = model
        logger.info(f"Model successfully loaded. Version: {self.manifest.get('version')}, Device: {self.device}")

    def get_classes(self) -> List[Dict[str, Any]]:
        """Return registered disease classes."""
        return self.classes

    def get_manifest(self) -> Dict[str, Any]:
        """Return model metadata manifest."""
        return self.manifest
