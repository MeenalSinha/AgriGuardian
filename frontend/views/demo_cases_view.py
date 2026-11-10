"""Demo Cases View — Pre-configured Realistic Field Scenarios for Validation."""

import streamlit as st
from backend.app.core.config import get_settings
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.ui_widgets import ui_page_header, ui_alert, ui_divider
from frontend.components.icons import render_icon

DEMO_SCENARIOS = [
    {
        "farmer_name": "राजेश कुमार (Rajesh Kumar)",
        "crop": "Wheat",
        "location": "Meerut, Uttar Pradesh",
        "symptoms": "Yellowing leaves with small brown spots, recent heavy rainfall over past 48 hours",
        "lat": 29.0,
        "lon": 77.7,
        "language": "hi",
    },
    {
        "farmer_name": "सुरेश पटेल (Suresh Patel)",
        "crop": "Cotton",
        "location": "Nagpur, Maharashtra",
        "symptoms": "Wilting plants, drooping leaves, intense dry heat exceeding 40°C",
        "lat": 21.1,
        "lon": 79.1,
        "language": "hi",
    },
    {
        "farmer_name": "अनिता देवी (Anita Devi)",
        "crop": "Tomato",
        "location": "Bengaluru Rural, Karnataka",
        "symptoms": "Small pinhole punctures in foliage, visible tiny insect clusters on leaf undersides",
        "lat": 12.9,
        "lon": 77.6,
        "language": "en",
    },
]


def render_demo_cases_view() -> None:
    """Render pre-configured demo cases for rapid evaluation."""
    settings = get_settings()

    ui_page_header(
        title="Reference Benchmark Cases",
        subtitle="Pre-configured agricultural field scenarios reflecting typical Indian smallholder cultivation challenges.",
        icon_name="file-text",
    )

    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system

    st.markdown("Select a reference case below to execute the end-to-end reasoning and risk advisory pipeline:")

    for idx, case in enumerate(DEMO_SCENARIOS, 1):
        with st.expander(f"Case {idx}: {case['farmer_name']} — {case['crop']} ({case['location']})", expanded=(idx == 1)):
            col1, col2 = st.columns([2.5, 1])

            with col1:
                st.write(f"**Farmer:** {case['farmer_name']}")
                st.write(f"**Crop:** {case['crop']}")
                st.write(f"**Location:** {case['location']} ({case['lat']}°N, {case['lon']}°E)")
                st.write(f"**Reported Condition:** {case['symptoms']}")

            with col2:
                if st.button(f"Execute Case {idx}", key=f"run_demo_{idx}", type="primary", use_container_width=True):
                    with st.spinner("Processing scenario through weather, rules, and reasoning engine..."):
                        result = system.process_request(
                            farmer_name=case["farmer_name"].split(" ")[0],
                            crop=case["crop"],
                            location=case["location"],
                            symptoms=case["symptoms"],
                            lat=case["lat"],
                            lon=case["lon"],
                            language=case["language"],
                        )
                        st.session_state.advisory_result = result
                        ui_alert(f"Case {idx} completed with Risk Assessment: **{result['risk']}**", level="success")
                        st.info("You can view complete advisory details in the 'Farm Advisory' module or check 'Analytics Dashboard'.")
