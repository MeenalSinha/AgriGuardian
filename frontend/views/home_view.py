"""Home View — Traditional Multi-source Agricultural Advisory & Disease Detection."""

import json
import os
import streamlit as st
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.icons import render_icon
from frontend.components.location_map import render_farmer_location_map
from frontend.components.ui_widgets import (
    ui_alert,
    ui_card,
    ui_divider,
    ui_metric_card,
    ui_page_header,
    ui_section_header,
    ui_status_badge,
)

logger = get_logger("HomeView")


def render_home_view() -> None:
    """Render the standard Farm Advisory generation view with enterprise styling."""
    settings = get_settings()

    ui_page_header(
        title="Farm Advisory & Decision Support",
        subtitle="Multi-source agricultural synthesis combining localized weather telemetry, mandi benchmark pricing, ICAR agronomic rules, and computer vision diagnosis.",
        pills=["Multi-Source Synthesis", "CIBRC Aligned", "ICAR Validated", "Bilingual Support"],
        icon_name="leaf",
    )

    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system

    if "advisory_result" not in st.session_state:
        st.session_state.advisory_result = None

    # Step 1: Farm Location Selection
    ui_section_header("Farm Location & Agro-Climatic Context", "Geographic coordinates and district benchmarking", icon="map-pin")
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
            st.info(f"Location coordinates: ({lat:.2f}, {lon:.2f})")

    ui_divider(style="dotted")

    # Step 2: Crop Details & Symptoms
    ui_section_header("Crop Details & Observable Symptoms", "Phenological stage and field observations", icon="sprout")
    col_crop, col_sym = st.columns([1, 1.8])

    with col_crop:
        crop = st.selectbox(
            "Select Crop",
            ["Wheat", "Rice", "Tomato", "Cotton", "Potato", "Sugarcane", "Onion", "Maize"],
            index=0,
        )
        language = st.selectbox(
            "Advisory Output Language",
            ["hi", "en"],
            index=0,
            format_func=lambda x: "हिन्दी (Hindi)" if x == "hi" else "English",
        )

    with col_sym:
        symptoms = st.text_area(
            "Describe Crop Symptoms or Field Issues",
            value="पत्तियों पर पीले और भूरे धब्बे दिखाई दे रहे हैं, हाल ही में भारी बारिश हुई है।",
            height=95,
        )

    # Step 3: Optional Crop Image Upload for Disease Detection
    ui_section_header("Foliar Diagnostic Photography (Optional)", "Computer vision disease detection and Grad-CAM explainability", icon="microscope")
    uploaded_image = st.file_uploader("Upload leaf photograph (.jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])
    image_temp_path = None

    if uploaded_image:
        temp_img_dir = os.path.join(settings.audio_temp_dir, "uploads")
        os.makedirs(temp_img_dir, exist_ok=True)
        image_temp_path = os.path.join(temp_img_dir, uploaded_image.name)
        with open(image_temp_path, "wb") as f:
            f.write(uploaded_image.read())
        st.image(image_temp_path, caption=f"Input Specimen: {uploaded_image.name}", width=260)

    # Generate Advisory Action
    if st.button("Generate Agricultural Advisory", type="primary", use_container_width=True):
        with st.spinner("Synthesizing meteorological risks, mandi benchmarks, and agronomic guidelines..."):
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
            ui_alert("Agricultural advisory successfully generated and grounded in statutory guidelines.", type="success")

    # Display Advisory Results
    if st.session_state.advisory_result:
        res = st.session_state.advisory_result
        ui_divider(style="gradient")

        ui_section_header(f"Advisory Assessment: {res['farmer_name']} ({res['crop']})", icon="file-text")

        # Risk Banner
        risk = res.get("risk", "Low")
        risk_color = "var(--ag-danger)" if risk == "High" else "var(--ag-warning)" if risk == "Medium" else "var(--ag-success)"
        risk_bg = "var(--ag-danger-bg)" if risk == "High" else "var(--ag-warning-bg)" if risk == "Medium" else "var(--ag-success-bg)"
        risk_border = "var(--ag-danger-border)" if risk == "High" else "var(--ag-warning-border)" if risk == "Medium" else "var(--ag-success-border)"
        
        st.markdown(
            f"""
            <div style="background-color: {risk_bg}; border: 1px solid {risk_border}; border-left: 5px solid {risk_color}; padding: 1.1rem 1.25rem; border-radius: 8px; margin: 0.8rem 0;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <span style="font-weight:700; color:{risk_color}; font-size:1.05rem;">Assessed Risk Level: {risk}</span>
                    {ui_status_badge(risk)}
                </div>
                <p style="margin-top: 0.4rem; color: #334155; font-size:0.92rem; line-height:1.45;">{res.get('justification', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_adv_hi, col_adv_en = st.columns(2)
        with col_adv_hi:
            ui_card("Hindi Guidance (सलाह)", res.get("advisory_hi", ""), icon="file-text", border_color="var(--ag-brand-primary)")
        with col_adv_en:
            ui_card("English Advisory", res.get("advisory_en", ""), icon="file-text", border_color="var(--ag-info)")

        # Three-step action plan
        ui_section_header("Three-Step Action Plan", "Prioritized interventions for immediate field implementation", icon="check-circle")
        for i, step in enumerate(res.get("three_step_plan", []), 1):
            st.markdown(
                f"""
                <div style="background-color: #FFFFFF; border: 1px solid var(--ag-border); border-left: 4px solid var(--ag-brand-primary);
                            padding: 0.8rem 1.1rem; border-radius: 8px; margin: 0.45rem 0; box-shadow: var(--ag-shadow-xs);">
                    <span style="font-weight:700; color:var(--ag-brand-primary); margin-right:8px;">Step {i}:</span>
                    <span style="color:var(--ag-text-primary); font-size:0.92rem;">{step}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Evidence Grounding & Provenance
        rec_data = res.get("recommendation")
        if rec_data:
            if rec_data.get("expert_referral_required"):
                ui_alert(
                    f"KVK Agronomist Consultation Advised: {rec_data.get('expert_referral_details', 'Elevated damage threshold detected. On-site verification recommended.')}",
                    type="warning",
                )

            precautions = rec_data.get("precautions", [])
            if precautions:
                st.markdown("#### Key Safety Precautions")
                for prec in precautions:
                    st.markdown(f"- **Precaution:** {prec}")

            ev_list = rec_data.get("supporting_evidence", []) or res.get("supporting_evidence", [])
            if ev_list:
                with st.expander(f"Grounded Evidence Provenance ({len(ev_list)} Authoritative Citations)", expanded=False):
                    conf = rec_data.get("confidence_assessment", {})
                    if conf:
                        st.caption(f"**Evidence Confidence:** {conf.get('level', 'Moderate')} — *{conf.get('rationale', '')}*")
                    for ev in ev_list:
                        org = ev.get("organization", "Agricultural Authority")
                        src_id = ev.get("source_id", "SOURCE")
                        text = ev.get("excerpt_text", "")
                        st.markdown(f"- **[{src_id}] {org}**: *\"{text}\"*")

        # Environmental & Market Context
        ui_section_header("Agro-Climatic & Market Conditions", "Live conditions influencing field decision support", icon="cloud-rain")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            ui_metric_card("Live Weather Summary", res.get("weather_summary", "N/A"), icon="cloud-rain")
        with col_m2:
            ui_metric_card("Market Price Benchmark", res.get("market_summary", "N/A"), icon="trending-up")

        # Grad-CAM Visualization if image analysis performed
        if res.get("image_analysis") and res["image_analysis"].get("gradcam_image"):
            gradcam_path = res["image_analysis"]["gradcam_image"]
            if os.path.exists(gradcam_path):
                ui_section_header("Explainable AI Diagnostic Attention Map (Grad-CAM)", icon="layers")
                st.image(gradcam_path, caption="Saliency Attention Map indicating detected symptom areas", width=350)

        # Download Real PDF Report Button
        ui_section_header("Reports & Data Export", "Export audit-ready documentation and records", icon="download")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if st.button("Generate Official PDF Report", use_container_width=True):
                pdf_path = system.generate_pdf_report(res)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Document",
                            data=pdf_file.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True,
                        )

        with col_r2:
            st.download_button(
                label="Export Case JSON Record",
                data=json.dumps(res, indent=2, ensure_ascii=False),
                file_name=f"advisory_{res['farmer_name']}.json",
                mime="application/json",
                use_container_width=True,
            )
