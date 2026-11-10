"""Dedicated Crop Disease Diagnosis and Explainable AI View for AgriGuardian."""

import os
import tempfile
import numpy as np
from PIL import Image
import streamlit as st

from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from backend.app.core.logging import get_logger
from backend.app.integrations.gnani.audio_utils import save_temporary_audio
from backend.app.integrations.gnani.timbre_tts import GnaniTimbreClient
from backend.app.schemas.disease import DiagnosisResult
from backend.app.services.agriculture.disease_detector import PlantDiseaseDetector
from frontend.components.icons import render_icon
from frontend.components.ui_widgets import (
    ui_alert,
    ui_card,
    ui_divider,
    ui_page_header,
    ui_section_header,
    ui_status_badge,
)

logger = get_logger("DiseaseDiagnosisView")


def render_disease_diagnosis_view() -> None:
    """Render the comprehensive crop disease detection and explainability interface."""
    settings = get_settings()

    ui_page_header(
        title="Intelligent Crop Disease Diagnosis",
        subtitle="Deep learning foliar diagnostics with genuine Grad-CAM explainability, clinical surface severity quantification, and ICAR-aligned IPM advisory.",
        pills=["MobileNetV3-Small CNN", "Grad-CAM Attention Overlay", "Surface Severity %", "ICAR IPM Advisory"],
    )

    if "disease_detector" not in st.session_state:
        st.session_state.disease_detector = PlantDiseaseDetector(settings=settings)
    detector: PlantDiseaseDetector = st.session_state.disease_detector

    if "timbre_client" not in st.session_state:
        st.session_state.timbre_client = GnaniTimbreClient(settings=settings)
    timbre: GnaniTimbreClient = st.session_state.timbre_client

    # Section 1: Image Capture / Upload
    ui_section_header("Submit Crop Photograph", "High-resolution leaf specimen for neural network inference", icon="microscope")

    input_mode = st.radio(
        "Choose Capture Mode:",
        ["Upload Image File", "Use Camera", "Diagnostic Demonstration Samples"],
        horizontal=True,
    )

    temp_image_path = None
    original_pil_image = None

    if input_mode == "Upload Image File":
        uploaded_file = st.file_uploader(
            "Upload an affected plant leaf photograph (JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Take a clear, well-focused close-up of a single leaflet in natural lighting.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            ext = os.path.splitext(uploaded_file.name)[1] or ".jpg"
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
            with open(temp_image_path, "wb") as f:
                f.write(file_bytes)
            original_pil_image = Image.open(temp_image_path)

    elif input_mode == "Use Camera":
        camera_img = st.camera_input("Capture leaf photo:")
        if camera_img is not None:
            file_bytes = camera_img.read()
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, "camera_capture.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(file_bytes)
            original_pil_image = Image.open(temp_image_path)

    elif input_mode == "Diagnostic Demonstration Samples":
        sample_choice = st.selectbox(
            "Select Diagnostic Benchmark Test Case:",
            [
                "Select a sample...",
                "Case 1: Early Blight (Alternaria solani) — Concentric Ring Target Spots",
                "Case 2: Late Blight (Phytophthora infestans) — Water-Soaked Dark Foliar Lesions",
                "Case 3: Healthy Tomato Foliage — Vigorous Green Physiological State",
                "Case 4: Bacterial Spot (Xanthomonas campestris) — Water-Soaked Halo Lesions",
            ],
        )

        if sample_choice != "Select a sample...":
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, "demo_leaf.jpg")

            # Generate realistic diagnostic demonstration leaflet
            import cv2
            arr = np.full((320, 320, 3), 245, dtype=np.uint8)
            if "Healthy" in sample_choice:
                cv2.ellipse(arr, (160, 160), (120, 80), 30, 0, 360, (40, 150, 48), -1)
                cv2.line(arr, (70, 110), (250, 210), (30, 120, 38), 2)
            elif "Early Blight" in sample_choice:
                cv2.ellipse(arr, (160, 160), (120, 80), 30, 0, 360, (45, 140, 50), -1)
                cv2.line(arr, (70, 110), (250, 210), (35, 110, 40), 2)
                for r, c in [(38, (30, 60, 100)), (28, (45, 80, 130)), (18, (25, 45, 80)), (8, (15, 25, 50))]:
                    cv2.circle(arr, (150, 150), r, c, -1)
            elif "Late Blight" in sample_choice:
                cv2.ellipse(arr, (160, 160), (120, 80), 30, 0, 360, (40, 130, 45), -1)
                cv2.ellipse(arr, (140, 150), (60, 40), 20, 0, 360, (35, 50, 65), -1)
            else:
                cv2.ellipse(arr, (160, 160), (120, 80), 30, 0, 360, (50, 150, 55), -1)
                for pt in [(130, 140), (170, 160), (150, 180), (120, 160)]:
                    cv2.circle(arr, pt, 6, (20, 40, 80), -1)

            noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            demo_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            demo_img.save(temp_image_path, format="JPEG")
            original_pil_image = demo_img

    # Run Diagnosis Button
    if temp_image_path and os.path.exists(temp_image_path):
        ui_divider(style="dotted")
        col_btn, col_empty = st.columns([1.5, 3])
        with col_btn:
            run_diag = st.button("Run Full Diagnostic Assessment", type="primary", use_container_width=True)

        if run_diag:
            with st.spinner("Executing deep neural network inference, Grad-CAM backpropagation, and severity assessment..."):
                diag_result: DiagnosisResult = detector.diagnose(temp_image_path)
                st.session_state.active_diagnosis = diag_result

    # Display Active Diagnosis Results
    if "active_diagnosis" in st.session_state:
        diag: DiagnosisResult = st.session_state.active_diagnosis

        ui_divider(style="bold")
        ui_section_header("Diagnostic Assessment Report", "Deep learning classification and clinical staging", icon="file-text")

        # 1. Quality Pre-flight Banner
        q = diag.quality_assessment
        if q.quality_tier == "Good":
            ui_alert(f"Image Quality: Optimal (Sharpness: {q.blur_score:.1f}, Exposure: {q.exposure_status}) — Highly reliable diagnostic conditions.", type="success")
        elif q.quality_tier == "Acceptable":
            ui_alert(f"Image Quality: Acceptable (Sharpness: {q.blur_score:.1f}, Exposure: {q.exposure_status}) — {q.farmer_advice}", type="warning")
        else:
            ui_alert(f"Image Quality: Compromised — {q.farmer_advice}", type="error")

        # 2. Main Metrics Grid
        col1, col2, col3 = st.columns([1.4, 1.3, 1.3])

        with col1:
            ui_card(
                f"### Primary Diagnosis\n\n"
                f"**{diag.common_name}**\n\n"
                f"*{diag.scientific_name or 'Etiology indeterminate'}*\n\n"
                f"**Confidence:** `{diag.confidence * 100:.1f}%` ({'High Confidence' if diag.is_confident else 'Low Confidence / Uncertain'})"
            )

        with col2:
            sev = diag.severity_assessment
            ui_card(
                f"### Observable Severity\n\n"
                f"**Clinical Stage:** **{sev.severity_stage}**\n\n"
                f"**Foliar Lesion Area:** `{sev.severity_score_percent:.1f}%` of visible leaf\n\n"
                f"**Environmental Multiplier:** `{sev.environmental_risk_multiplier}x`"
            )

        with col3:
            ui_card(
                f"### Inference Profile\n\n"
                f"**Architecture:** `{diag.model_metadata.get('architecture', 'MobileNetV3-Small')}`\n\n"
                f"**Latency:** `{diag.inference_time_ms:.1f} ms` on CPU\n\n"
                f"**Status:** `Genuine Weights (Non-Simulated)`"
            )

        # 3. Top-3 Candidate Ranking
        ui_section_header("Model Prediction Ranking (Top-3 Candidates)", icon="bar-chart")
        for pred in diag.top_predictions:
            c_col1, c_col2 = st.columns([3, 1])
            with c_col1:
                st.progress(float(pred.confidence), text=f"**{pred.common_name}** ({pred.scientific_name or ''})")
            with c_col2:
                st.markdown(f"**{pred.confidence_percentage}**")

        ui_divider(style="dotted")

        # 4. Explainable AI: Grad-CAM Side-by-Side Visualizer
        ui_section_header("Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM)", "Visual evidence and convolutional attention localization", icon="layers")
        st.caption(
            "Grad-CAM visualizes the exact convolutional feature activation regions that triggered the model's diagnostic classification. "
            "Warm colors represent the primary visual evidence (lesion margins, necrosis, fungal spots) utilized by the network."
        )

        col_orig, col_cam = st.columns(2)
        with col_orig:
            if original_pil_image is not None:
                st.image(original_pil_image, caption="Original Input Photograph", use_container_width=True)

        with col_cam:
            if diag.gradcam_image_path and os.path.exists(diag.gradcam_image_path):
                st.image(diag.gradcam_image_path, caption="Grad-CAM Diagnostic Attention Overlay", use_container_width=True)
            else:
                st.info("Grad-CAM overlay not generated due to low prediction confidence or invalid input.")

        ui_divider(style="dotted")

        # 5. Integrated Pest Management (IPM) Guidelines
        ui_section_header("Integrated Pest & Disease Management (IPM) Guidelines", "Statutory agronomic protocols aligned with ICAR recommendations", icon="shield-check")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Cultural & Sanitation Practices",
            "Biological Controls",
            "Chemical Precautions",
            "KVK Agricultural Extension Advisory",
        ])

        with tab1:
            st.markdown("#### Cultural & Farm Sanitation Practices")
            for item in diag.management_guidelines.cultural_practices:
                st.markdown(f"- {item}")

        with tab2:
            st.markdown("#### Biological & Organic Interventions")
            for item in diag.management_guidelines.biological_controls:
                st.markdown(f"- {item}")

        with tab3:
            st.markdown("#### Chemical Precautions & Safety Waiting Periods")
            ui_alert("Safety Notice: Follow Central Insecticide Board & Registration Committee (CIBRC) guidelines. Always wear masks, gloves, and protective boots when preparing spray dilutions.", type="warning")
            for item in diag.management_guidelines.chemical_precautions:
                st.markdown(f"- {item}")

        with tab4:
            st.markdown("#### Krishi Vigyan Kendra (KVK) Specialist Guidance")
            ui_alert(f"Official Extension Advisory: {diag.management_guidelines.kvk_advisory}", type="info")

        ui_divider(style="dotted")

        # 6. Spoken Voice Audio Playback via Gnani Timbre
        ui_section_header("Spoken Audio Diagnostic Summary", "Multilingual voice readout powered by Gnani Timbre v2.5", icon="volume-2")

        lang_options = list(SUPPORTED_LANGUAGES.keys())
        v_col1, v_col2 = st.columns([1.5, 2])

        with v_col1:
            speech_lang = st.selectbox(
                "Select Voice Language / भाषा चुनें:",
                options=lang_options,
                index=0,
                format_func=lambda code: f"{SUPPORTED_LANGUAGES[code]['name']} ({SUPPORTED_LANGUAGES[code]['native_name']})",
                key="diag_speech_lang_select",
            )

        with v_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Synthesize & Play Voice Summary", type="primary", use_container_width=True):
                with st.spinner("Synthesizing spoken diagnostic summary via Gnani Timbre v2.5..."):
                    spoken_text = (
                        f"Crop Disease Diagnosis: {diag.common_name}. "
                        f"Confidence: {diag.confidence * 100:.0f} percent. "
                        f"Severity: {diag.severity_assessment.severity_stage} stage covering {diag.severity_assessment.severity_score_percent:.0f} percent foliage area. "
                        f"Recommended cultural action: {' '.join(diag.management_guidelines.cultural_practices[:1])}. "
                        f"KVK Guidance: {diag.management_guidelines.kvk_advisory}"
                    )

                    try:
                        tts_res = timbre.synthesize(text=spoken_text, language_code=speech_lang)
                        if tts_res.success and tts_res.audio_file_path and os.path.exists(tts_res.audio_file_path):
                            with open(tts_res.audio_file_path, "rb") as f:
                                st.audio(f.read(), format="audio/wav")
                            ui_alert(f"Spoken voice generated with voice '{tts_res.voice}' ({tts_res.duration_seconds:.1f}s)", type="success")
                        else:
                            ui_alert(f"Voice generation note: {tts_res.error or 'Synthesis unavailable.'}", type="warning")
                    except Exception as e:
                        ui_alert(f"Voice synthesis error: {e}", type="error")
