"""Pydantic schemas for crop disease detection, explainability, and diagnostics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageValidationResult(BaseModel):
    """Schema for raw image validation checks."""
    is_valid: bool = Field(..., description="Whether image passed format and integrity validation")
    format: Optional[str] = Field(None, description="Image format (JPEG, PNG, WEBP)")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    channels: Optional[int] = Field(None, description="Color channels (usually 3 for RGB)")
    file_size_bytes: int = Field(..., description="Raw image file size in bytes")
    error_message: Optional[str] = Field(None, description="Validation failure details if invalid")


class QualityAssessmentResult(BaseModel):
    """Schema for leaf photograph quality assessment."""
    quality_tier: str = Field(..., description="Quality tier: 'Good', 'Acceptable', or 'Poor'")
    is_acceptable: bool = Field(..., description="Whether quality is sufficient for reliable diagnosis")
    blur_score: float = Field(..., description="Laplacian variance sharpness metric")
    is_blurry: bool = Field(..., description="Whether blur exceeded tolerance threshold")
    brightness_score: float = Field(..., description="Mean pixel luminance (0-255)")
    exposure_status: str = Field(..., description="Exposure state: 'Under-exposed', 'Optimal', 'Over-exposed'")
    contrast_score: float = Field(..., description="Luminance standard deviation")
    warnings: List[str] = Field(default_factory=list, description="Actionable image capture warnings")
    farmer_advice: Optional[str] = Field(None, description="Actionable camera guidance for the farmer")


class ClassPrediction(BaseModel):
    """Schema for individual disease class prediction candidate."""
    class_index: int = Field(..., description="Zero-based class index")
    class_id: str = Field(..., description="Standardized benchmark class label")
    common_name: str = Field(..., description="Farmer-friendly common disease name")
    scientific_name: Optional[str] = Field(None, description="Causal pathogen scientific name")
    confidence: float = Field(..., description="Calibrated softmax probability (0.0 to 1.0)")
    confidence_percentage: str = Field(..., description="Formatted percentage string (e.g. '87.4%')")


class SeverityAssessmentResult(BaseModel):
    """Schema for evidence-based disease severity and leaf area damage."""
    severity_score_percent: float = Field(..., description="Percentage of visible leaf surface exhibiting lesions")
    severity_stage: str = Field(..., description="Clinical staging: 'Healthy', 'Mild', 'Moderate', or 'Severe'")
    leaf_area_pixels: int = Field(..., description="Segmented green/foliar surface area in pixels")
    lesion_area_pixels: int = Field(..., description="Segmented necrotic/chlorotic surface area in pixels")
    environmental_risk_multiplier: float = Field(default=1.0, description="Compound multiplier from ambient weather")
    environmental_warnings: List[str] = Field(default_factory=list, description="Weather-triggered pathogen risk factors")
    clinical_summary: str = Field(..., description="Agronomic explanation of observable foliage damage")


class ManagementGuidelines(BaseModel):
    """Schema for integrated pest and disease management guidelines."""
    cultural_practices: List[str] = Field(default_factory=list, description="Foliage sanitation, spacing, watering")
    biological_controls: List[str] = Field(default_factory=list, description="Bio-agents, beneficial insects, organic sprays")
    chemical_precautions: List[str] = Field(default_factory=list, description="Targeted chemicals with cautionary safety guidance")
    kvk_advisory: str = Field(..., description="Guidance for consulting local agricultural extension / KVK officers")


class DiagnosisResult(BaseModel):
    """Comprehensive schema for end-to-end plant disease diagnosis."""
    disease_id: str = Field(..., description="Predicted primary disease identifier or 'Unknown'")
    common_name: str = Field(..., description="Primary disease common name")
    scientific_name: Optional[str] = Field(None, description="Causal pathogen scientific name")
    confidence: float = Field(..., description="Primary prediction calibrated confidence score")
    is_confident: bool = Field(..., description="Whether prediction meets minimum operational threshold")
    is_uncertain: bool = Field(..., description="Flag indicating low confidence, OOD, or ambiguous symptoms")
    top_predictions: List[ClassPrediction] = Field(default_factory=list, description="Top-k sorted prediction candidates")
    quality_assessment: QualityAssessmentResult = Field(..., description="Image quality metrics and blur warnings")
    severity_assessment: SeverityAssessmentResult = Field(..., description="Physical observable leaf damage metrics")
    management_guidelines: ManagementGuidelines = Field(..., description="ICAR-aligned IPM advisory")
    gradcam_image_path: Optional[str] = Field(None, description="Path to rendered Grad-CAM diagnostic heatmap")
    gradcam_overlay_base64: Optional[str] = Field(None, description="Base64 encoded Grad-CAM PNG for UI embedding")
    model_metadata: Dict[str, Any] = Field(default_factory=dict, description="Model architecture, version, and device")
    inference_time_ms: float = Field(..., description="Total wall-clock inference and explainability latency in ms")
    is_simulated: bool = Field(default=False, description="Explicit verification that real model weights were used")

