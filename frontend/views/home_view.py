"""Home View — Traditional Multi-source Agricultural Advisory & Disease Detection."""

import json
import os
import streamlit as st
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.location_map import render_farmer_location_map
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider, ui_metric_card

logger = get_logger("HomeView")


def render_home_view() -> None:
    """Render the standard Farm Advisory generation view."""
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #a8e063 0%, #56ab2f 100%);
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center; color: #1B5E20;">
            <h1 style="margin: 0; font-size: 2.2rem; color: #1B5E20;">🌾 AgriGuardian — Farm Advisory System</h1>
            <p style="margin-top: 0.4rem; font-size: 1rem; color: #2E7D32;">
                Integrated crop diagnostics, real-time weather risks, market insights, and tailored recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system

    if "advisory_result" not in st.session_state:
        st.session_state.advisory_result = None

    # Step 1: Farm Location Selection
    st.markdown("### 📍 Farm Location & Agro-Climatic Zone")
    col_map, col_loc = st.columns([1.5, 1])

    with col_loc:
        lat = st.number_input("Latitude", value=settings.default_latitude, format="%.4f")
        lon = st.number_input("Longitude", value=settings.default_longitude, format="%.4f")
        location_name = st.text_input("Village / District / Mandi", value=settings.default_location_name)
        farmer_name = st.text_input("Farmer Name", value="राजेश कुमार")
        phone_number = st.text_input("Phone Number (Optional)", value="+91 98765 43210")

    with col_map:
        try:
            new_lat, new_lon = render_farmer_location_map(lat=lat, lon=lon, farmer_name=farmer_name)
        except Exception as e:
            st.info(f"Interactive map preview: ({lat:.2f}, {lon:.2f})")

    ui_divider(style="dotted")

    # Step 2: Crop Details & Symptoms
    st.markdown("### 🌱 Crop Diagnosis & Condition")
    col_crop, col_sym = st.columns([1, 1.8])

    with col_crop:
        crop = st.selectbox(
            "Select Crop",
            ["Wheat", "Rice", "Tomato", "Cotton", "Potato", "Sugarcane", "Onion", "Maize"],
            index=0,
        )
        language = st.selectbox("Advisory Language", ["hi", "en"], index=0, format_func=lambda x: "हिन्दी (Hindi)" if x == "hi" else "English")

    with col_sym:
        symptoms = st.text_area(
            "Describe Crop Symptoms or Problem",
            value="पत्तियों पर पीले और भूरे धब्बे दिखाई दे रहे हैं, हाल ही में भारी बारिश हुई है।",
            height=95,
        )

    # Step 3: Optional Crop Image Upload for Disease Detection
    st.markdown("### 📸 Crop Leaf Photo (Optional for Vision AI Diagnosis)")
    uploaded_image = st.file_uploader("Upload leaf photograph (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])
    image_temp_path = None

    if uploaded_image:
        temp_img_dir = os.path.join(settings.audio_temp_dir, "uploads")
        os.makedirs(temp_img_dir, exist_ok=True)
        image_temp_path = os.path.join(temp_img_dir, uploaded_image.name)
        with open(image_temp_path, "wb") as f:
            f.write(uploaded_image.read())
        st.image(image_temp_path, caption=f"Uploaded Crop Image: {uploaded_image.name}", width=260)

    # Generate Advisory Action
    if st.button("🚀 Generate Agricultural Advisory", type="primary", use_container_width=True):
        with st.spinner("Analyzing weather conditions, market prices, and diagnostic rules..."):
            result = system.process_request(
                farmer_name=farmer_name,
                crop=crop,
                location=location_name,
                symptoms=symptoms,
                lat=lat,
                lon=lon,
                image_path=image_temp_path,
                language=language,
                phone_number=phone_number,
            )
            st.session_state.advisory_result = result
            st.success("Advisory successfully generated!")

    # Display Advisory Results
    if st.session_state.advisory_result:
        res = st.session_state.advisory_result
        ui_divider(style="gradient")

        st.markdown(f"## 📋 Advisory Assessment for {res['farmer_name']} ({res['crop']})")

        # Risk Banner
        risk = res.get("risk", "Low")
        risk_color = "#C62828" if risk == "High" else "#F57F17" if risk == "Medium" else "#2E7D32"
        st.markdown(
            f"""
            <div style="background-color: {risk_color}15; border-left: 6px solid {risk_color}; padding: 1.2rem; border-radius: 8px; margin: 1rem 0;">
                <h3 style="color: {risk_color}; margin: 0;">Assessed Risk Level: {risk}</h3>
                <p style="margin-top: 0.4rem; color: #333;">{res.get('justification', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_adv_hi, col_adv_en = st.columns(2)
        with col_adv_hi:
            ui_card("सलाह (Hindi Guidance)", res.get("advisory_hi", ""), icon="🇮🇳", border_color="#2E7D32")
        with col_adv_en:
            ui_card("Advisory (English)", res.get("advisory_en", ""), icon="🇬🇧", border_color="#1565C0")

        # Three-step action plan
        st.markdown("### 🎯 Three-Step Action Plan")
        for i, step in enumerate(res.get("three_step_plan", []), 1):
            st.markdown(
                f"""
                <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-left: 4px solid #4CAF50;
                            padding: 0.7rem 1rem; border-radius: 6px; margin: 0.4rem 0;">
                    <b>Step {i}:</b> {step}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Environmental & Market Context
        st.markdown("### 🌦️ Agro-Climatic & Market Conditions")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            ui_metric_card("Live Weather Summary", res.get("weather_summary", "N/A"), icon="🌦️")
        with col_m2:
            ui_metric_card("Market Price Benchmark", res.get("market_summary", "N/A"), icon="💰")

        # Grad-CAM Visualization if image analysis performed
        if res.get("image_analysis") and res["image_analysis"].get("gradcam_image"):
            gradcam_path = res["image_analysis"]["gradcam_image"]
            if os.path.exists(gradcam_path):
                st.markdown("### 🔬 Explainable AI Diagnostic Map (Grad-CAM)")
                st.image(gradcam_path, caption="Saliency Attention Map indicating detected symptom areas", width=350)

        # Download Real PDF Report Button
        st.markdown("### 📄 Reports & Export")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if st.button("📄 Generate & Download PDF Report", use_container_width=True):
                pdf_path = system.generate_pdf_report(res)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ Click to Download Official PDF",
                            data=pdf_file.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True,
                        )

        with col_r2:
            st.download_button(
                label="📊 Export Case JSON Data",
                data=json.dumps(res, indent=2, ensure_ascii=False),
                file_name=f"advisory_{res['farmer_name']}.json",
                mime="application/json",
                use_container_width=True,
            )

