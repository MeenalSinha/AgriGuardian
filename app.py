"""AgriGuardian — Intelligent Multilingual Agricultural Companion.

Main Streamlit Application Entrypoint.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Add workspace root to sys.path to ensure modular package imports resolve cleanly
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, setup_logging
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.ui_widgets import ui_divider
from frontend.views.about_view import render_about_view
from frontend.views.dashboard_view import render_dashboard_view
from frontend.views.demo_cases_view import render_demo_cases_view
from frontend.views.disease_diagnosis_view import render_disease_diagnosis_view
from frontend.views.expert_view import render_expert_view
from frontend.views.farm_management_view import render_farm_management_view
from frontend.views.home_view import render_home_view
from frontend.views.knowledge_admin_view import render_knowledge_admin_view
from frontend.views.market_intelligence_view import render_market_intelligence_view
from frontend.views.notifications_view import render_notifications_view
from frontend.views.voice_assistant_view import render_voice_assistant_view
from frontend.views.weather_dashboard_view import render_weather_dashboard_view

# Initialize configuration and logging
settings = get_settings()
setup_logging(log_level=settings.log_level, logs_dir=settings.logs_dir)
logger = get_logger("App")

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="AgriGuardian - Intelligent Farm Companion",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .main-header {
            font-size: 2.3rem;
            color: #1B5E20;
            text-align: center;
            padding: 1.2rem;
            background: linear-gradient(120deg, #a8e063 0%, #56ab2f 100%);
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize global system state
if "system" not in st.session_state:
    st.session_state.system = AgriGuardianSystem(settings=settings)

# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==============================================================================

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f33e.png",
        width=70,
    )
    st.title("AgriGuardian")
    st.caption("🌾 *Your Intelligent Agricultural Companion*")
    ui_divider(style="bold")

    page = st.radio(
        "Navigation",
        [
            "🌾 My Farms & Crops",
            "🔔 Alerts & Reminders",
            "🎙️ Voice Assistant",
            "🔬 Crop Disease Diagnosis",
            "🌦️ Weather Intelligence",
            "📈 Market Intelligence",
            "🏠 Farm Advisory",
            "👨‍⚕️ Expert Portal",
            "📚 Knowledge Base",
            "📊 Dashboard",
            "👨‍🌾 Demo Cases",
            "ℹ️ About",
        ],
        index=0,
    )

    ui_divider(style="dotted")

    # System Integration Status
    st.markdown("### 🔌 System Services")
    gnani_status = "🟢 Active" if settings.is_gnani_configured else "🟡 Config Missing"
    llm_status = "🟢 Active" if settings.is_llm_configured else "🟡 Fallback Mode"

    st.markdown(f"**Gnani AI (STT/TTS):** {gnani_status}")
    st.markdown("**Crop Vision (PyTorch):** 🟢 Active (MobileNetV3)")
    st.markdown(f"**LLM Reasoning:** {llm_status}")
    st.markdown("**Weather (Open-Meteo):** 🟢 Active")
    st.markdown("**Mandi Intelligence:** 🟢 Active")
    st.markdown("**Proactive Outbox:** 🟢 Active (Zero Data Loss)")

    ui_divider(style="dotted")

    # Recent Advisories History
    st.markdown("### 🕘 Recent Advisories")
    df_recent = st.session_state.system.data_manager.get_advisories()

    if not df_recent.empty:
        for i, row in df_recent.tail(4).iloc[::-1].iterrows():
            risk = row.get("risk", "Low")
            icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "⚪")
            farmer_label = str(row.get("farmer_name", "Farmer"))[:14]
            crop_label = str(row.get("crop", "N/A"))

            with st.expander(f"{icon} {farmer_label} ({crop_label})", expanded=False):
                st.caption(f"**Risk:** {risk}")
                st.caption(f"**Time:** {str(row.get('timestamp', ''))[:16]}")
                st.caption(f"**Advice:** {str(row.get('advisory_en', ''))[:60]}...")
    else:
        st.caption("No advisories recorded yet.")

# ==============================================================================
# ROUTE TO VIEW
# ==============================================================================

if page == "🌾 My Farms & Crops":
    render_farm_management_view()
elif page == "🔔 Alerts & Reminders":
    render_notifications_view()
elif page == "🎙️ Voice Assistant":
    render_voice_assistant_view()
elif page == "🔬 Crop Disease Diagnosis":
    render_disease_diagnosis_view()
elif page == "🌦️ Weather Intelligence":
    render_weather_dashboard_view()
elif page == "📈 Market Intelligence":
    render_market_intelligence_view()
elif page == "🏠 Farm Advisory":
    render_home_view()
elif page == "👨‍⚕️ Expert Portal":
    render_expert_view()
elif page == "📚 Knowledge Base":
    render_knowledge_admin_view()
elif page == "📊 Dashboard":
    render_dashboard_view()
elif page == "👨‍🌾 Demo Cases":
    render_demo_cases_view()
elif page == "ℹ️ About":
    render_about_view()

# Footer
ui_divider(style="gradient")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 1rem 0;'>
        🌾 <b>AgriGuardian</b> — Empowering Indian Farmers with Multilingual Voice AI and Agricultural Intelligence<br>
        <span style='font-size: 0.75rem; color: #888;'>
            Powered by Gnani Prisma v2.5 • Gnani Timbre v2.5 • Open-Meteo Weather • Dynamic Rules • ReportLab PDF
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
