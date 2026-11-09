"""Evidence-based disease severity assessment service decoupling confidence from physical damage."""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
from backend.app.schemas.disease import SeverityAssessmentResult


class SeverityAssessor:
    """Calculates physical leaf damage percentage and clinical disease staging."""

    @staticmethod
    def calculate_severity(
        pil_image: Image.Image,
        disease_name: str,
        is_healthy: bool = False,
        weather_context: Optional[Dict[str, Any]] = None,
    ) -> SeverityAssessmentResult:
        """Estimate the proportion of visible leaf surface affected by necrotic or chlorotic lesions.

        Args:
            pil_image: Original PIL RGB photograph of leaf
            disease_name: Common name of predicted disease
            is_healthy: True if top prediction is Healthy foliage
            weather_context: Optional dictionary with ambient meteorological indicators

        Returns:
            SeverityAssessmentResult
        """
        if is_healthy or disease_name.lower() in ("healthy", "healthy foliage"):
            return SeverityAssessmentResult(
                severity_score_percent=0.0,
                severity_stage="Healthy",
                leaf_area_pixels=0,
                lesion_area_pixels=0,
                environmental_risk_multiplier=1.0,
                environmental_warnings=[],
                clinical_summary="Foliage exhibits vigorous, uniform green tissue with no visible pathological lesions.",
            )

        # Convert to HSV color space
        rgb_array = np.array(pil_image)
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)

        # 1. Segment the Leaf Surface (Vegetation Mask)
        # Broad spectrum capturing healthy green, chlorotic yellow, and diseased leaf regions
        lower_leaf = np.array([15, 25, 25])
        upper_leaf = np.array([105, 255, 255])
        leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)

        # Refine leaf mask using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_pixel_count = int(np.count_nonzero(leaf_mask))

        # Fallback if background segmentation is non-standard
        total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
        if leaf_pixel_count < 100:
            leaf_pixel_count = total_pixels
            leaf_mask = np.ones((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8) * 255

        # 2. Segment Pathological Lesions Within the Leaf Mask
        # Necrotic brown/dark lesions: low brightness or brown hues
        lower_brown = np.array([5, 30, 20])
        upper_brown = np.array([25, 255, 170])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Severe chlorotic yellow halos (disease symptom)
        lower_yellow = np.array([20, 60, 130])
        upper_yellow = np.array([36, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Very dark necrotic spot centers
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        dark_mask = np.where((gray < 60) & (leaf_mask > 0), 255, 0).astype(np.uint8)

        # Combine symptomatic masks and constrain strictly to leaf area
        symptom_mask = cv2.bitwise_or(brown_mask, yellow_mask)
        symptom_mask = cv2.bitwise_or(symptom_mask, dark_mask)
        symptom_mask = cv2.bitwise_and(symptom_mask, leaf_mask)

        lesion_pixel_count = int(np.count_nonzero(symptom_mask))

        # 3. Calculate Physical Surface Damage Percentage
        damage_percentage = float((lesion_pixel_count / max(leaf_pixel_count, 1)) * 100.0)
        damage_percentage = min(max(damage_percentage, 0.5), 98.0)  # Bound reasonably

        # 4. Clinical Staging
        if damage_percentage <= 10.0:
            stage = "Mild"
            summary_desc = (
                f"Early-stage infection with localized lesions covering {damage_percentage:.1f}% of visible leaf surface. "
                "Early intervention can prevent systemic spread."
            )
        elif damage_percentage <= 30.0:
            stage = "Moderate"
            summary_desc = (
                f"Established infection with expanding lesions covering {damage_percentage:.1f}% of visible leaf surface. "
                "Active management and targeted containment are recommended."
            )
        else:
            stage = "Severe"
            summary_desc = (
                f"Advanced high-density infection covering {damage_percentage:.1f}% of visible leaf tissue. "
                "High risk of defoliation, secondary spread, and severe yield loss."
            )

        # 5. Factored Environmental Risk Context
        weather = weather_context or {}
        env_multiplier = 1.0
        env_warnings: List[str] = []

        if weather.get("rain_recent") or weather.get("rainfall_mm", 0) > 5.0:
            env_multiplier *= 1.3
            env_warnings.append("Recent rainfall significantly accelerates fungal sporulation and bacterial splash.")

        if weather.get("humidity_high") or weather.get("humidity_pct", 0) > 75.0:
            env_multiplier *= 1.2
            env_warnings.append("High relative humidity creates an ideal microclimate for foliar pathogens.")

        if weather.get("temp_very_high") or weather.get("temp_celsius", 25) > 36.0:
            env_multiplier *= 1.15
            env_warnings.append("Extreme heat compounds transpiration stress on diseased foliage.")

        return SeverityAssessmentResult(
            severity_score_percent=round(damage_percentage, 2),
            severity_stage=stage,
            leaf_area_pixels=leaf_pixel_count,
            lesion_area_pixels=lesion_pixel_count,
            environmental_risk_multiplier=round(env_multiplier, 2),
            environmental_warnings=env_warnings,
            clinical_summary=summary_desc,
        )

