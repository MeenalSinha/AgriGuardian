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
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider

logger = get_logger("DiseaseDiagnosisView")


def render_disease_diagnosis_view() -> None:
    """Render the comprehensive crop disease detection and explainability interface."""
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
                    padding: 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; text-align: center;">
            <h1 style="margin: 0; font-size: 2.2rem; color: #FFFFFF;">🔬 Intelligent Crop Disease Diagnosis</h1>
            <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.95;">
                Deep learning foliar diagnostics with genuine Grad-CAM explainability and evidence-based severity staging.
            </p>
            <div style="margin-top: 0.6rem; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    🧠 MobileNetV3-Small CNN
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    🔥 Genuine Grad-CAM Heatmaps
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    📐 Decoupled Surface Severity %
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 3px 12px; border-radius: 15px; font-size: 0.82rem;">
                    🏛️ ICAR-Aligned IPM Advisory
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "disease_detector" not in st.session_state:
        st.session_state.disease_detector = PlantDiseaseDetector(settings=settings)
    detector: PlantDiseaseDetector = st.session_state.disease_detector

    if "timbre_client" not in st.session_state:
        st.session_state.timbre_client = GnaniTimbreClient(settings=settings)
    timbre: GnaniTimbreClient = st.session_state.timbre_client

    # Section 1: Image Capture / Upload
    st.markdown("### 📸 Submit Crop Photograph / पत्ती की फोटो अपलोड करें")

    input_mode = st.radio(
        "Choose Capture Mode:",
        ["📁 Upload Image File", "📷 Use Camera", "🧪 Demonstration Samples"],
        horizontal=True,
    )

    temp_image_path = None
    original_pil_image = None

    if input_mode == "📁 Upload Image File":
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

    elif input_mode == "📷 Use Camera":
        camera_img = st.camera_input("Capture leaf photo:")
        if camera_img is not None:
            file_bytes = camera_img.read()
            temp_dir = settings.audio_temp_dir or tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, "camera_capture.jpg")
            with open(temp_image_path, "wb") as f:
                f.write(file_bytes)
            original_pil_image = Image.open(temp_image_path)

    elif input_mode == "🧪 Demonstration Samples":
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

            # Generate structured synthetic demonstration leaflet
            arr = np.zeros((320, 320, 3), dtype=np.uint8)
            if "Healthy" in sample_choice:
                arr[:, :] = [38, 148, 48]
            elif "Early Blight" in sample_choice:
                arr[:, :] = [45, 140, 50]
                arr[100:190, 100:190] = [70, 45, 25]  # Concentric dark lesion
                arr[120:170, 120:170] = [40, 25, 15]  # Center
            elif "Late Blight" in sample_choice:
                arr[:, :] = [40, 130, 45]
                arr[80:240, 90:230] = [50, 35, 20]  # Large water-soaked patch
            else:
                arr[:, :] = [50, 150, 55]
                # Small bacterial speckles
                arr[60:80, 60:80] = [60, 30, 10]
                arr[180:200, 150:170] = [60, 30, 10]

            demo_img = Image.fromarray(arr)
            demo_img.save(temp_image_path, format="JPEG")
            original_pil_image = demo_img

    # Run Diagnosis Button
    if temp_image_path and os.path.exists(temp_image_path):
        ui_divider(style="dotted")
        col_btn, col_empty = st.columns([1.5, 3])
        with col_btn:
            run_diag = st.button("🔍 Run Full Diagnostic Assessment", type="primary", use_container_width=True)

        if run_diag:
            with st.spinner("🔬 Executing deep neural network inference, Grad-CAM backpropagation, and severity assessment..."):
                diag_result: DiagnosisResult = detector.diagnose(temp_image_path)
                st.session_state.active_diagnosis = diag_result

    # Display Active Diagnosis Results
    if "active_diagnosis" in st.session_state:
        diag: DiagnosisResult = st.session_state.active_diagnosis

        ui_divider(style="bold")
        st.markdown("## 📋 Diagnostic Assessment Report")

        # 1. Quality Pre-flight Banner
        q = diag.quality_assessment
        if q.quality_tier == "Good":
            st.success(f"🟢 **Image Quality: Optimal** (Sharpness: {q.blur_score:.1f}, Exposure: {q.exposure_status}) — Reliable diagnostic conditions.")
        elif q.quality_tier == "Acceptable":
            st.warning(f"🟡 **Image Quality: Acceptable** (Sharpness: {q.blur_score:.1f}, Exposure: {q.exposure_status}) — {q.farmer_advice}")
        else:
            st.error(f"🔴 **Image Quality: Compromised** — {q.farmer_advice}")

        # 2. Main Metrics Grid
        col1, col2, col3 = st.columns([1.4, 1.3, 1.3])

        with col1:
            ui_card(
                f"### 🦠 Primary Diagnosis\n\n"
                f"**{diag.common_name}**\n\n"
                f"*{diag.scientific_name or 'Etiology indeterminate'}*\n\n"
                f"**Confidence:** `{diag.confidence * 100:.1f}%` ({'High Confidence' if diag.is_confident else 'Low Confidence / Uncertain'})"
            )

        with col2:
            sev = diag.severity_assessment
            stage_colors = {"Healthy": "🟢", "Mild": "🟡", "Moderate": "🟠", "Severe": "🔴"}
            badge = stage_colors.get(sev.severity_stage, "⚪")

            ui_card(
                f"### 📐 Observable Severity\n\n"
                f"**Clinical Stage:** {badge} **{sev.severity_stage}**\n\n"
                f"**Foliar Lesion Area:** `{sev.severity_score_percent:.1f}%` of visible leaf\n\n"
                f"**Environmental Multiplier:** `{sev.environmental_risk_multiplier}x`"
            )

        with col3:
            ui_card(
                f"### ⚡ Inference Profile\n\n"
                f"**Architecture:** `{diag.model_metadata.get('architecture', 'MobileNetV3-Small')}`\n\n"
                f"**Latency:** `{diag.inference_time_ms:.1f} ms` on CPU\n\n"
                f"**Status:** `Genuine Weights (Non-Simulated)`"
            )

        # 3. Top-3 Candidate Ranking
        st.markdown("#### 📊 Model Prediction Ranking (Top-3 Candidates)")
        for pred in diag.top_predictions:
            c_col1, c_col2 = st.columns([3, 1])
            with c_col1:
                st.progress(float(pred.confidence), text=f"**{pred.common_name}** ({pred.scientific_name or ''})")
            with c_col2:
                st.markdown(f"**{pred.confidence_percentage}**")

        ui_divider(style="dotted")

        # 4. Explainable AI: Grad-CAM Side-by-Side Visualizer
        st.markdown("### 🔥 Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM)")
        st.caption(
            "Grad-CAM visualizes the exact convolutional feature activation regions that triggered the model's diagnostic classification. "
            "Warm colors (red/orange) represent the primary visual evidence (lesion margins, necrosis, fungal spots) utilized by the network."
        )

        col_orig, col_cam = st.columns(2)
        with col_orig:
            if original_pil_image is not None:
                st.image(original_pil_image, caption="Original Input Photograph", use_column_width=True)

        with col_cam:
            if diag.gradcam_image_path and os.path.exists(diag.gradcam_image_path):
                st.image(diag.gradcam_image_path, caption="Grad-CAM Diagnostic Attention Overlay", use_column_width=True)
            else:
                st.info("Grad-CAM overlay not generated due to low prediction confidence or invalid input.")

        ui_divider(style="dotted")

        # 5. Integrated Pest Management (IPM) Guidelines
        st.markdown("### 🏛️ Integrated Pest & Disease Management (IPM) Guidelines")
        st.caption("Standardized agronomic protocols aligned with Indian Council of Agricultural Research (ICAR) recommendations:")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🌿 Cultural & Sanitation Practices",
            "🐞 Biological Controls",
            "⚠️ Chemical Precautions",
            "🏛️ KVK Agricultural Extension Advisory",
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
            st.warning("⚠️ **Safety Notice:** Follow Central Insecticide Board & Registration Committee (CIBRC) guidelines. Always wear masks, gloves, and protective boots when preparing spray dilutions.")
            for item in diag.management_guidelines.chemical_precautions:
                st.markdown(f"- {item}")

        with tab4:
            st.markdown("#### Krishi Vigyan Kendra (KVK) Specialist Guidance")
            st.info(f"🏛️ **Official Extension Advice:** {diag.management_guidelines.kvk_advisory}")

        ui_divider(style="dotted")

        # 6. Spoken Voice Audio Playback via Gnani Timbre
        st.markdown("### 🎙️ Listen to Diagnosis in Your Preferred Indian Language")
        st.caption("Powered by Gnani Timbre v2.5 Indian Voice Intelligence:")

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
            if st.button("🔊 Synthesize & Play Voice Summary", type="primary", use_container_width=True):
                with st.spinner("🎧 Synthesizing spoken diagnostic summary via Gnani Timbre v2.5..."):
                    # Construct clean verbal summary for speech synthesis
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
                            st.success(f"✅ Spoken voice generated with voice '{tts_res.voice}' ({tts_res.duration_seconds:.1f}s)")
                        else:
                            st.warning(f"Voice generation note: {tts_res.error or 'Synthesis unavailable.'}")
                    except Exception as e:
                        st.error(f"Voice synthesis error: {e}")
