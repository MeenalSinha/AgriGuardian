"""AgriGuardian — Intelligent Multilingual Agricultural Companion.

Main Streamlit Application Entrypoint.
Redesigned with enterprise SaaS aesthetic and strict NO-EMOJI compliance.
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
from frontend.components.icons import render_icon
from frontend.components.ui_widgets import load_custom_css, ui_divider, ui_status_badge
from frontend.views.about_view import render_about_view
from frontend.views.dashboard_view import render_dashboard_view
from frontend.views.demo_cases_view import render_demo_cases_view
from frontend.views.disease_diagnosis_view import render_disease_diagnosis_view
from frontend.views.expert_view import render_expert_view
from frontend.views.farm_economics_view import render_farm_economics_view
from frontend.views.farm_management_view import render_farm_management_view
from frontend.views.home_view import render_home_view
from frontend.views.knowledge_admin_view import render_knowledge_admin_view
from frontend.views.market_intelligence_view import render_market_intelligence_view
from frontend.views.notifications_view import render_notifications_view
from frontend.views.offline_sync_view import render_offline_sync_view
from frontend.views.voice_assistant_view import render_voice_assistant_view
from frontend.views.weather_dashboard_view import render_weather_dashboard_view

# Initialize configuration and logging
settings = get_settings()
setup_logging(log_level=settings.log_level, logs_dir=settings.logs_dir)
logger = get_logger("App")

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CENTRALIZED DESIGN SYSTEM
# ==============================================================================

st.set_page_config(
    page_title="AgriGuardian | Enterprise Agricultural Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject centralized enterprise stylesheet
load_custom_css()

# Initialize global system state
if "system" not in st.session_state:
    st.session_state.system = AgriGuardianSystem(settings=settings)

# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ==============================================================================

with st.sidebar:
    # Enterprise Brand Header
    st.markdown(
        f"""
        <div class="ag-sidebar-brand">
            <div class="ag-sidebar-logo-icon">
                {render_icon("sprout", size=22, color="#FFFFFF")}
            </div>
            <div class="ag-sidebar-brand-text">
                <span class="ag-sidebar-brand-title">AgriGuardian</span>
                <span class="ag-sidebar-brand-subtitle">Agricultural Intelligence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Directory with clear semantic groupings
    st.markdown('<div class="ag-nav-category">Navigation Modules</div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "My Farms & Crops",
            "Farm Economics & Profitability",
            "AI Voice Assistant",
            "Crop Disease Diagnosis",
            "Weather Intelligence",
            "Market Intelligence",
            "Farm Advisory",
            "Alerts & Reminders",
            "Offline & Sync Center",
            "Expert Review Portal",
            "Agricultural Knowledge Base",
            "Analytics Dashboard",
            "Reference Benchmark Cases",
            "System Architecture & About",
        ],
        index=0,
        label_visibility="collapsed",
    )

    ui_divider(style="default")

    # System Integration Telemetry
    st.markdown('<div class="ag-nav-category">System Infrastructure</div>', unsafe_allow_html=True)
    
    gnani_active = bool(settings.is_gnani_configured)
    gnani_status = "Operational" if gnani_active else "Standby Mode"
    gnani_color = "var(--ag-success)" if gnani_active else "var(--ag-warning)"
    gnani_dot = "ag-dot-active" if gnani_active else "ag-dot-warning"

    llm_active = bool(settings.is_llm_configured)
    llm_status = "Operational" if llm_active else "Rule Engine Fallback"
    llm_color = "var(--ag-success)" if llm_active else "var(--ag-warning)"
    llm_dot = "ag-dot-active" if llm_active else "ag-dot-warning"

    st.markdown(
        f"""
        <div class="ag-service-card">
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("mic", size=13, color="#64748B")} Gnani Voice AI</span>
                <span class="ag-service-status" style="color:{gnani_color};">
                    <span class="ag-dot {gnani_dot}"></span> {gnani_status}
                </span>
            </div>
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("microscope", size=13, color="#64748B")} Vision AI (PyTorch)</span>
                <span class="ag-service-status" style="color:var(--ag-success);">
                    <span class="ag-dot ag-dot-active"></span> MobileNetV3
                </span>
            </div>
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("bot", size=13, color="#64748B")} Advisory Engine</span>
                <span class="ag-service-status" style="color:{llm_color};">
                    <span class="ag-dot {llm_dot}"></span> {llm_status}
                </span>
            </div>
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("cloud-rain", size=13, color="#64748B")} Weather Telemetry</span>
                <span class="ag-service-status" style="color:var(--ag-success);">
                    <span class="ag-dot ag-dot-active"></span> Open-Meteo
                </span>
            </div>
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("trending-up", size=13, color="#64748B")} Mandi Markets</span>
                <span class="ag-service-status" style="color:var(--ag-success);">
                    <span class="ag-dot ag-dot-active"></span> AGMARKNET
                </span>
            </div>
            <div class="ag-service-row">
                <span class="ag-service-name">{render_icon("shield-check", size=13, color="#64748B")} Outbox Queue</span>
                <span class="ag-service-status" style="color:var(--ag-success);">
                    <span class="ag-dot ag-dot-active"></span> Zero Data Loss
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ui_divider(style="default")

    # Recent Advisories History
    st.markdown('<div class="ag-nav-category">Recent Advisories</div>', unsafe_allow_html=True)
    df_recent = st.session_state.system.data_manager.get_advisories()

    if not df_recent.empty:
        for i, row in df_recent.tail(4).iloc[::-1].iterrows():
            risk = row.get("risk", "Low")
            farmer_label = str(row.get("farmer_name", "Farmer"))[:14]
            crop_label = str(row.get("crop", "Crop"))
            status_tag = f"[{risk}]"

            with st.expander(f"{status_tag} {farmer_label} ({crop_label})", expanded=False):
                st.caption(f"Risk Assessment: {risk}")
                st.caption(f"Timestamp: {str(row.get('timestamp', ''))[:16]}")
                st.caption(f"Guidance Excerpt: {str(row.get('advisory_en', ''))[:60]}...")
    else:
        st.caption("No advisories logged in current session.")

# ==============================================================================
# ROUTE TO VIEW
# ==============================================================================

if page == "My Farms & Crops":
    render_farm_management_view()
elif page == "Farm Economics & Profitability":
    render_farm_economics_view()
elif page == "AI Voice Assistant":
    render_voice_assistant_view()
elif page == "Crop Disease Diagnosis":
    render_disease_diagnosis_view()
elif page == "Weather Intelligence":
    render_weather_dashboard_view()
elif page == "Market Intelligence":
    render_market_intelligence_view()
elif page == "Farm Advisory":
    render_home_view()
elif page == "Alerts & Reminders":
    render_notifications_view()
elif page == "Offline & Sync Center":
    render_offline_sync_view()
elif page == "Expert Review Portal":
    render_expert_view()
elif page == "Agricultural Knowledge Base":
    render_knowledge_admin_view()
elif page == "Analytics Dashboard":
    render_dashboard_view()
elif page == "Reference Benchmark Cases":
    render_demo_cases_view()
elif page == "System Architecture & About":
    render_about_view()

# Global Footer
st.markdown(
    f"""
    <div class="ag-footer">
        <span class="ag-footer-brand">AgriGuardian</span> — Intelligent Agricultural Decision Support & Farm Management Platform<br>
        <span style="font-size:0.75rem; color:#94A3B8;">
            Powered by Gnani Prisma & Timbre • Open-Meteo Meteorology • Dynamic Rule Engine • CIBRC & ICAR Certified Protocols
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
