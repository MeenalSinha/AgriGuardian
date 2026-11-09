"""Image quality assessment service for foliar disease photography."""

from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from backend.app.schemas.disease import QualityAssessmentResult

BLUR_THRESHOLD_STRICT = 120.0
BLUR_THRESHOLD_MINIMAL = 75.0
BRIGHTNESS_MIN_OPTIMAL = 40.0
BRIGHTNESS_MAX_OPTIMAL = 220.0
CONTRAST_MIN_OPTIMAL = 25.0


class ImageQualityAssessor:
    """Evaluates sharpness, exposure, and contrast of leaf photographs."""

    @staticmethod
    def assess_quality(pil_image: Image.Image) -> QualityAssessmentResult:
        """Analyze PIL RGB image and compute optical quality metrics."""
        # Convert PIL to OpenCV grayscale and BGR numpy arrays
        rgb_array = np.array(pil_image)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        # 1. Sharpness via Laplacian Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        is_blurry = blur_score < BLUR_THRESHOLD_MINIMAL

        # 2. Brightness via Mean Luminance
        brightness_score = float(np.mean(gray))
        if brightness_score < BRIGHTNESS_MIN_OPTIMAL:
            exposure_status = "Under-exposed"
        elif brightness_score > BRIGHTNESS_MAX_OPTIMAL:
            exposure_status = "Over-exposed"
        else:
            exposure_status = "Optimal"

        # 3. Contrast via Luminance Standard Deviation (RMS contrast)
        contrast_score = float(np.std(gray))

        # 4. Warnings and Farmer Guidance
        warnings: List[str] = []
        guidance: List[str] = []

        if is_blurry:
            warnings.append(f"Image appears blurry (sharpness metric {blur_score:.1f} < {BLUR_THRESHOLD_MINIMAL:.0f}).")
            guidance.append("Hold camera steady and ensure the leaf surface is sharply in focus.")
        elif blur_score < BLUR_THRESHOLD_STRICT:
            warnings.append("Slight camera softness detected, though diagnosis remains possible.")

        if exposure_status == "Under-exposed":
            warnings.append(f"Photo is too dark (average brightness {brightness_score:.1f}/255).")
            guidance.append("Move to open sunlight or illuminate the leaf with natural daylight.")
        elif exposure_status == "Over-exposed":
            warnings.append(f"Photo has harsh glare or over-exposure (average brightness {brightness_score:.1f}/255).")
            guidance.append("Avoid direct flash glare; photograph under diffused shade or indirect light.")

        if contrast_score < CONTRAST_MIN_OPTIMAL:
            warnings.append("Low contrast detected between foliage and background.")
            guidance.append("Place leaf against a contrasting background or isolate an individual leaflet.")

        # Determine overall Quality Tier
        if is_blurry or exposure_status in ("Under-exposed", "Over-exposed"):
            quality_tier = "Poor"
            is_acceptable = False
        elif blur_score < BLUR_THRESHOLD_STRICT or contrast_score < CONTRAST_MIN_OPTIMAL:
            quality_tier = "Acceptable"
            is_acceptable = True
        else:
            quality_tier = "Good"
            is_acceptable = True

        farmer_advice = " ".join(guidance) if guidance else "Image quality is optimal for reliable diagnostic assessment."

        return QualityAssessmentResult(
            quality_tier=quality_tier,
            is_acceptable=is_acceptable,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            brightness_score=round(brightness_score, 2),
            exposure_status=exposure_status,
            contrast_score=round(contrast_score, 2),
            warnings=warnings,
            farmer_advice=farmer_advice,
        )

