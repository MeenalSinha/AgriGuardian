"""About View — Architecture, Technology Stack, and Mission."""

import streamlit as st
from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from frontend.components.ui_widgets import ui_page_header, ui_section_header, ui_divider
from frontend.components.icons import render_icon


def render_about_view() -> None:
    """Render the about and architecture documentation page."""
    settings = get_settings()

    ui_page_header(
        title="System Architecture & Platform Overview",
        subtitle="Intelligent Agricultural Companion for Indian Farmers — Technical Architecture, Capabilities & Research Standards.",
        icon_name="info",
    )

    st.markdown(
        """
        ### Mission
        AgriGuardian bridges the critical gap between cutting-edge AI and everyday farming realities in India.
        By combining **multilingual voice intelligence**, **micro-climate weather awareness**, **dynamic agricultural rules**,
        and **crop health diagnostics**, AgriGuardian empowers farmers to make informed, data-driven decisions in their own language.

        ---

        ### Core System Capabilities

        #### 1. Multilingual Voice Assistant (Phase 1)
        - **Speech-to-Text (STT):** Powered by **Gnani Prisma v2.5** (`prisma-v2.5`), converting spoken Indian regional languages directly into actionable queries.
        - **Text-to-Speech (TTS):** Powered by **Gnani Timbre v2.5** (`timbre-v2.5`), synthesizing compassionate, natural spoken guidance across regional languages and voices (such as *Nalini*, *Deepak*, *Kaveri*).
        - **10 Indian Languages Supported:** Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Punjabi, Bengali, and Gujarati.
        - **Transcript Verification:** Allows farmers to review and edit recognized speech before advisory processing.

        #### 2. Context-Aware Agricultural Reasoning
        - **Live Open-Meteo Integration:** Evaluates real-time temperatures, humidity, and rainfall sums for the farmer's exact GPS coordinates.
        - **Dynamic Agricultural Rule Engine:** Evaluates environmental triggers (e.g., high humidity + yellowing = fungal risk).
        - **Agricultural Safety Guardrails:** Prevents dangerous chemical dosages, prioritizing integrated pest management (IPM) and authorized Krishi Vigyan Kendra (KVK) extension referrals.

        #### 3. Vision AI & Explainability
        - **Diagnostic Heuristics & Grad-CAM Attention:** Visualizes high-salience leaf regions to aid agronomists in identifying symptom patterns.
        - **Compound Severity Scoring:** Combines disease risk weight with ambient weather vectors.

        #### 4. Field Officer Dashboard & PDF Reports
        - **Executive Analytics:** Aggregates regional risk distributions, peak query times, and crop issue trends.
        - **Authentic PDF Reporting:** Automatically compiles full diagnostic records and action plans using ReportLab for field archival.

        ---

        ### Technical Architecture

        | Layer | Technology / Provider | Purpose |
        |---|---|---|
        | **Frontend UI** | Streamlit + Custom SaaS Design System | Clean, responsive, mobile-friendly interface |
        | **Voice STT** | Gnani Prisma v2.5 | Automatic Speech Recognition for Indic languages |
        | **Voice TTS** | Gnani Timbre v2.5 | Natural Speech Synthesis for Indic languages |
        | **Core Framework** | Python 3.11+, Pydantic v2 | Typed configuration, validation, modular backend |
        | **Weather API** | Open-Meteo REST API | Multi-day meteorological forecasts & ERA5 baseline |
        | **Market Intelligence** | Agmarknet & e-NAM Catalog | APMC mandi rates & statutory MSP benchmarks |
        | **Document Engine** | ReportLab | PDF generation for field advisory exports |
        | **Mapping** | Folium, Streamlit-Folium | Interactive geospatial farm location picker |
        """
    )

    ui_divider(style="dotted")

    ui_section_header("Verified Language Support Catalog", "Supported Indian languages, native scripts, and standard voice profiles.", "bot")
    lang_table = [
        {"Language": info["name"], "Script": info["native_name"], "Code": code, "Default Voice": info["default_voice"]}
        for code, info in SUPPORTED_LANGUAGES.items()
    ]
    st.table(lang_table)
