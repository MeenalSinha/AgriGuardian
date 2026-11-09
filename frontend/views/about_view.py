"""About View — Architecture, Technology Stack, and Mission."""

import streamlit as st
from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from frontend.components.ui_widgets import ui_card, ui_divider


def render_about_view() -> None:
    """Render the about and architecture documentation page."""
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #2E7D32 0%, #388E3C 100%);
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.2rem; color: white;">ℹ️ About AgriGuardian</h1>
            <p style="margin-top: 0.4rem; font-size: 1rem; opacity: 0.95;">
                Intelligent Agricultural Companion for Indian Farmers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ## 🌾 Mission
        AgriGuardian bridges the critical gap between cutting-edge AI and everyday farming realities in India.
        By combining **multilingual voice intelligence**, **micro-climate weather awareness**, **dynamic agricultural rules**,
        and **crop health diagnostics**, AgriGuardian empowers farmers to make informed, data-driven decisions in their own language.

        ---

        ## 🧠 Core System Capabilities

        ### 1. 🎙️ Multilingual Voice Assistant (Phase 1)
        - **Speech-to-Text (STT):** Powered by **Gnani Prisma v2.5** (`prisma-v2.5`), converting spoken Indian regional languages directly into actionable queries.
        - **Text-to-Speech (TTS):** Powered by **Gnani Timbre v2.5** (`timbre-v2.5`), synthesizing compassionate, natural spoken guidance across regional languages and voices (such as *Nalini*, *Deepak*, *Kaveri*).
        - **10 Indian Languages Supported:** Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Punjabi, Bengali, and Gujarati.
        - **Transcript Verification:** Allows farmers to review and edit recognized speech before advisory processing.

        ### 2. 🌦️ Context-Aware Agricultural Reasoning
        - **Live Open-Meteo Integration:** Evaluates real-time temperatures, humidity, and rainfall sums for the farmer's exact GPS coordinates.
        - **Dynamic Agricultural Rule Engine:** Evaluates environmental triggers (e.g., high humidity + yellowing = fungal risk).
        - **Agricultural Safety Guardrails:** Prevents dangerous hallucinated chemical dosages, prioritizing integrated pest management (IPM) and authorized Krishi Vigyan Kendra (KVK) extension referrals.

        ### 3. 🔬 Vision AI & Explainability
        - **Diagnostic Heuristics & Grad-CAM Attention:** Visualizes high-salience leaf regions to aid agronomists in identifying symptom patterns.
        - **Compound Severity Scoring:** Combines disease risk weight with ambient weather vectors.

        ### 4. 📊 Field Officer Dashboard & PDF Reports
        - **Executive Analytics:** Aggregates regional risk distributions, peak query times, and crop issue trends.
        - **Authentic PDF Reporting:** Automatically compiles full diagnostic records and action plans using ReportLab for field archival.

        ---

        ## 🧩 Technical Architecture

        | Layer | Technology / Provider | Purpose |
        |---|---|---|
        | **Frontend UI** | Streamlit | Responsive, mobile-friendly interface |
        | **Voice STT** | Gnani Prisma v2.5 | Automatic Speech Recognition for Indic languages |
        | **Voice TTS** | Gnani Timbre v2.5 | Natural Speech Synthesis for Indic languages |
        | **Core Framework** | Python 3.14, Pydantic v2 | Typed configuration, validation, modular backend |
        | **Weather API** | Open-Meteo REST API | 3-day meteorological forecasts |
        | **Market Intelligence** | Agmarknet Indicative Model | Transparent mandi benchmark prices |
        | **Document Engine** | ReportLab | PDF generation for field advisory exports |
        | **Mapping** | Folium, Streamlit-Folium | Interactive geospatial farm location picker |
        """
    )

    ui_divider(style="gradient")

    st.markdown("### 🌐 Verified Language Support Catalog")
    lang_table = [
        {"Language": info["name"], "Script": info["native_name"], "Code": code, "Default Voice": info["default_voice"]}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]
    st.table(lang_table)

