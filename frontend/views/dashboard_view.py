"""Dashboard View — Field Officer & Agronomist Analytics."""

import streamlit as st
from backend.app.core.config import get_settings
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.ui_widgets import ui_alert, ui_divider, ui_metric_card


def render_dashboard_view() -> None:
    """Render the analytics dashboard for field officers."""
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #1E88E5 0%, #1565C0 100%);
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.2rem; color: white;">📊 Field Officer & Analytics Dashboard</h1>
            <p style="margin-top: 0.4rem; font-size: 1rem; opacity: 0.9;">
                Monitor regional farmer queries, track disease outbreaks, and review pending field visit cases.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system

    df = system.data_manager.get_advisories()

    if df.empty:
        st.info("🔭 No advisories registered yet. Generate a case from the Home page or Voice Assistant to see live analytics!")
        return

    # Top KPI Metrics
    total = len(df)
    high_risk = len(df[df["risk"] == "High"]) if "risk" in df.columns else 0
    medium_risk = len(df[df["risk"] == "Medium"]) if "risk" in df.columns else 0
    low_risk = len(df[df["risk"] == "Low"]) if "risk" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui_metric_card("Total Advisories", str(total), icon="📊", color="#1565C0")
    with col2:
        ui_metric_card("High Risk Cases", str(high_risk), icon="🔴", color="#C62828")
    with col3:
        ui_metric_card("Medium Risk Cases", str(medium_risk), icon="🟡", color="#F57F17")
    with col4:
        ui_metric_card("Low Risk / Healthy", str(low_risk), icon="🟢", color="#2E7D32")

    ui_divider(style="gradient")

    # Crop and Risk Distribution
    st.markdown("### 🌾 Crop Breakdown & Outbreak Analytics")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### Crop Distribution")
        if "crop" in df.columns:
            crop_counts = df["crop"].value_counts()
            st.bar_chart(crop_counts)

    with col_c2:
        st.markdown("#### Risk Distribution")
        risk_dist = system.analytics.get_risk_distribution()
        st.bar_chart(risk_dist)

    ui_divider(style="dotted")

    # Table of Records with Status Updates
    st.markdown("### 📋 Case Register & Review")
    display_cols = ["timestamp", "farmer_name", "crop", "location", "risk", "field_visit_required", "status"]
    available_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(df[available_cols], use_container_width=True)

    # Status update tool
    st.markdown("#### ✏️ Update Case Review Status")
    col_u1, col_u2, col_u3 = st.columns([1, 1.5, 1])

    with col_u1:
        row_idx = st.number_input("Advisory Index", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1)
    with col_u2:
        new_status = st.selectbox("New Status", ["pending", "reviewed", "in_progress", "completed"])
    with col_u3:
        feedback = st.text_input("Agronomist Notes", value="")

    if st.button("Update Status"):
        if system.data_manager.update_status(row_idx, new_status, feedback):
            st.success(f"Updated advisory #{row_idx} status to '{new_status}'")
            st.rerun()
        else:
            st.error("Failed to update advisory status.")

