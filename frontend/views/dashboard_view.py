"""Dashboard View — Unified Role-Based Agricultural Intelligence and Monitoring.

Supports three role-tailored operational surfaces:
1. Personalized Farmer Dashboard (farm context, phenology, due activities, farm weather/market).
2. Field Officer Dashboard (assigned territory, case registry, small-cohort privacy, follow-up audit logs).
3. Platform Administrator Dashboard (service health, model telemetry, data quality, territory assignment).
"""

from datetime import datetime, date
import streamlit as st

from backend.app.core.config import get_settings
from backend.app.schemas.analytics import (
    AdminDashboardOverview,
    CaseFollowUpCreate,
    FarmerDashboardOverview,
    FieldOfficerDashboardOverview,
    MetricStatus,
)
from backend.app.schemas.farm import UserRole
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider, ui_metric_card


def render_dashboard_view() -> None:
    """Render unified role-based agricultural dashboard."""
    settings = get_settings()

    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system
    orchestrator = system.analytics_orchestrator

    # Determine user role from session
    current_user = st.session_state.get("current_user")
    current_farmer = st.session_state.get("current_farmer")

    # Header Banner
    st.markdown(
        """
        <div style="background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 60%, #1565C0 100%);
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.2rem; color: white;">📊 Unified Agricultural Intelligence & Monitoring</h1>
            <p style="margin-top: 0.4rem; font-size: 1.05rem; opacity: 0.95;">
                Role-grounded analytics for Farmers, Regional Agronomists & Platform Administrators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Role Selection / Switcher with Strict Server-Side Role Enforcement
    if not current_user:
        st.warning("🔒 Authentication Required: Please log in or register via '🌾 My Farms & Crops' to access your role-based dashboard.")
        st.info("For privacy and platform security, metrics and cases are scoped strictly to authenticated farmer, field officer, expert, or administrator accounts.")
        return

    # Determine permitted dashboards based on user role
    role_options = []
    if current_user.role == UserRole.FARMER:
        role_options = ["🌾 Farmer Dashboard (Personalized)"]
    elif current_user.role == UserRole.FIELD_OFFICER:
        role_options = ["👮 Field Officer Dashboard (Regional Intelligence)"]
        if current_farmer:
            role_options.append("🌾 Farmer Dashboard (My Personal Farm)")
    elif current_user.role == UserRole.AGRONOMIST:
        role_options = ["🔬 Agricultural Expert Portal (Case Verification)"]
        if current_farmer:
            role_options.append("🌾 Farmer Dashboard (My Personal Farm)")
    elif current_user.role == UserRole.ADMINISTRATOR:
        role_options = [
            "⚙️ Administrator Dashboard (Platform Telemetry)",
            "👮 Field Officer Dashboard (Regional Intelligence)",
            "🔬 Agricultural Expert Portal (Case Verification)",
            "🌾 Farmer Dashboard (Personalized)",
        ]

    col_role, col_refresh = st.columns([3, 1])
    with col_role:
        selected_dashboard = st.selectbox(
            "Select Authorized Dashboard Surface",
            options=role_options,
            index=0,
            help="Displays only dashboards authorized for your authenticated user role.",
        )
    with col_refresh:
        st.write("")
        st.write("")
        force_refresh = st.button("🔄 Force Refresh Cache", use_container_width=True)

    ui_divider(style="gradient")

    if "Farmer" in selected_dashboard:
        _render_farmer_dashboard(orchestrator, current_farmer, force_refresh)
    elif "Field Officer" in selected_dashboard:
        _render_field_officer_dashboard(orchestrator, current_user, force_refresh)
    elif "Administrator" in selected_dashboard:
        _render_administrator_dashboard(orchestrator, current_user, force_refresh)
    else:
        # Phase 8 Expert Portal placeholder (rendered via expert view module)
        st.info("🔬 Agricultural Expert Portal is available in the dedicated Expert Review console.")


# ==============================================================================
# 1. FARMER DASHBOARD SURFACE
# ==============================================================================

def _render_farmer_dashboard(orchestrator, current_farmer, force_refresh: bool) -> None:
    """Render personalized farmer dashboard."""
    if not current_farmer:
        st.warning("🔒 No Farmer Profile Found: Please log in with a registered farmer account via '🌾 My Farms & Crops'.")
        st.info("Dashboard land holdings, phenology, and tasks are isolated strictly to your authenticated farmer profile.")
        return

    farmer_id = current_farmer.farmer_id

    try:
        # Check if farmer has multiple farms
        farms = orchestrator.db.list_farms_by_farmer(farmer_id)
        selected_farm_id = None

        if farms and len(farms) > 1:
            farm_names = {f"{f.farm_name} ({f.district}, {f.state})": f.id for f in farms}
            chosen_name = st.selectbox("Select Active Farm", list(farm_names.keys()))
            selected_farm_id = farm_names[chosen_name]
        elif farms:
            selected_farm_id = farms[0].id

        overview: FarmerDashboardOverview = orchestrator.get_farmer_dashboard(
            farmer_id=farmer_id,
            farm_id=selected_farm_id,
            force_refresh=force_refresh,
        )
    except Exception as e:
        st.error(f"Failed to load farmer dashboard: {e}")
        return

    # Farmer Identity Banner
    st.markdown(
        f"""
        <div style="background-color: #F1F8E9; border-left: 5px solid #4CAF50; padding: 1rem; border-radius: 6px; margin-bottom: 1.2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #2E7D32;">🌱 Welcome, {overview.farmer.full_name}</h3>
                    <p style="margin: 0.2rem 0 0 0; color: #558B2F; font-size: 0.95rem;">
                        <strong>Farm:</strong> {overview.active_farm.farm_name if overview.active_farm else 'Primary Holding'} | 
                        <strong>Location:</strong> {overview.active_farm.district if overview.active_farm else 'Territory'}, {overview.active_farm.state if overview.active_farm else 'India'} | 
                        <strong>Profile Completion:</strong> {overview.farmer.profile_completion_pct:.0f}%
                    </p>
                </div>
                <div style="background-color: #C8E6C9; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: bold; color: #1B5E20; font-size: 0.85rem;">
                    {overview.data_freshness}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Agricultural Metric Cards
    m = overview.metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui_metric_card(
            m["farmer_active_farms"].name,
            m["farmer_active_farms"].display_value,
            icon="🏡",
            color="#2E7D32",
        )
    with c2:
        ui_metric_card(
            m["farmer_total_acres"].name,
            m["farmer_total_acres"].display_value,
            icon="📐",
            color="#1565C0",
        )
    with c3:
        ui_metric_card(
            m["farmer_active_crops"].name,
            m["farmer_active_crops"].display_value,
            icon="🌾",
            color="#F57F17",
        )
    with c4:
        ui_metric_card(
            m["farmer_due_tasks"].name,
            m["farmer_due_tasks"].display_value,
            icon="📅",
            color="#D84315" if m["farmer_due_tasks"].status == MetricStatus.WARNING else "#2E7D32",
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        ui_metric_card(
            m["farmer_crop_health_cases"].name,
            m["farmer_crop_health_cases"].display_value,
            icon="🩺",
            color="#6A1B9A",
        )
    with c6:
        ui_metric_card(
            m["farmer_weather_risk_index"].name,
            m["farmer_weather_risk_index"].display_value,
            icon="🌦️",
            color="#00838F",
        )
    with c7:
        ui_metric_card(
            m["farmer_market_price_bench"].name,
            m["farmer_market_price_bench"].display_value,
            icon="💰",
            color="#2E7D32",
        )

    ui_divider(style="dotted")

    # Phenology and Crop Lifecycle Progress
    col_p1, col_p2 = st.columns([1.5, 1])
    with col_p1:
        st.markdown("### 🌿 Active Crop Phenological Stages & Progress")
        if overview.active_crop_stages:
            for s in overview.active_crop_stages:
                st.markdown(
                    f"""
                    <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 0.9rem; margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.1rem; font-weight: bold; color: #2E7D32;">{s['crop_name']} ({s.get('variety') or 'Standard'})</span>
                                <span style="margin-left: 0.5rem; background: #E8F5E9; color: #1B5E20; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                    {s['season']}
                                </span>
                            </div>
                            <span style="font-weight: bold; color: #1565C0; font-size: 0.9rem;">
                                {s['days_since_sowing']} Days Since Sowing
                            </span>
                        </div>
                        <p style="margin: 0.4rem 0 0.2rem 0; font-size: 0.9rem; color: #555;">
                            <strong>Current Stage:</strong> <span style="color: #E65100; font-weight: 600;">{s['current_stage'].replace('_', ' ').title()}</span> | 
                            <strong>Target Yield:</strong> {s.get('target_yield') or 'N/A'} q/acre
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No active crop cycles currently sown. Add one under '🌾 My Farms & Crops'.")

    with col_p2:
        st.markdown("### 📊 Crop Area Distribution")
        if overview.crop_distribution:
            chart_data = {cp.label: cp.value for cp in overview.crop_distribution}
            st.bar_chart(chart_data)
        else:
            st.info("Crop area distribution will appear once crops are sown.")

    # Grounded Agronomic Insights
    if overview.insights:
        st.markdown("### 💡 Grounded Agronomic & Climate Insights")
        for ins in overview.insights:
            sev_color = "#D32F2F" if ins.severity == "critical" else (
                "#F57F17" if ins.severity == "warning" else (
                    "#1976D2" if ins.severity == "recommendation" else "#388E3C"
                )
            )
            st.markdown(
                f"""
                <div style="background-color: white; border-left: 4px solid {sev_color}; border-radius: 6px; padding: 0.9rem; margin-bottom: 0.6rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: {sev_color}; font-size: 1rem;">{ins.title}</strong>
                        <span style="font-size: 0.8rem; color: #777;">{ins.source}</span>
                    </div>
                    <p style="margin: 0.3rem 0; font-size: 0.92rem; color: #333;">{ins.description}</p>
                    {f'<div style="background-color: #F5F5F5; padding: 0.4rem 0.6rem; border-radius: 4px; font-size: 0.88rem; color: #1B5E20;"><strong>Action:</strong> {ins.suggested_action}</div>' if ins.suggested_action else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Due Farm Activities Checklist
    ui_divider(style="dotted")
    col_a1, col_a2 = st.columns([1, 1])

    with col_a1:
        st.markdown("### 📋 Upcoming & Due Farm Activities")
        if overview.due_activities:
            for act in overview.due_activities:
                status_color = "#D32F2F" if act.status.value == "overdue" else "#1976D2"
                st.markdown(
                    f"""
                    <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{act.title}</strong>
                            <span style="color: {status_color}; font-weight: bold; font-size: 0.8rem;">{act.status.value.upper()}</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem; color: #555;">
                            <strong>Type:</strong> {act.activity_type.value.title()} | 
                            <strong>Due:</strong> {act.due_date.isoformat() if act.due_date else 'Immediate'}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("🎉 All scheduled farm tasks are completed! No pending activities.")

    with col_a2:
        st.markdown("### 🩺 Recent Crop Health & Advisory Cases")
        if overview.recent_health_cases:
            for case in overview.recent_health_cases:
                risk_color = "#D32F2F" if case.risk == "High" else ("#F57F17" if case.risk == "Medium" else "#388E3C")
                st.markdown(
                    f"""
                    <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>Case {case.case_id} — {case.crop}</strong>
                            <span style="color: {risk_color}; font-weight: bold; font-size: 0.8rem;">{case.risk.upper()} RISK</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem; color: #555;">
                            <strong>Symptoms:</strong> {case.symptoms[:50]}... | <strong>Status:</strong> {case.status.title()}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No crop health cases reported yet. Use the Voice Assistant or Diagnosis tool to run checks.")


# ==============================================================================
# 2. FIELD OFFICER DASHBOARD SURFACE
# ==============================================================================

def _render_field_officer_dashboard(orchestrator, current_user, force_refresh: bool) -> None:
    """Render regional monitoring, case register, and agronomist follow-up interface."""
    if not current_user or current_user.role not in (UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR):
        st.error("⛔ Access Forbidden: Field Officer or Administrator credentials required.")
        return

    officer_id = current_user.user_id
    officer_name = current_user.username

    # Filter Bar
    st.markdown("### 🔍 Regional Telemetry Filters")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        crop_filter = st.selectbox("Crop Filter", ["All Crops", "Wheat", "Rice", "Tomato", "Potato", "Cotton", "Maize"])
        crop_param = None if crop_filter == "All Crops" else crop_filter
    with col_f2:
        risk_filter = st.selectbox("Risk Filter", ["All Risk Levels", "High", "Medium", "Low"])
        risk_param = None if risk_filter == "All Risk Levels" else risk_filter
    with col_f3:
        status_filter = st.selectbox("Status Filter", ["All Statuses", "pending", "in_progress", "reviewed", "resolved"])
        status_param = None if status_filter == "All Statuses" else status_filter
    with col_f4:
        date_range = st.selectbox("Date Window", ["Last 30 Days", "Last 7 Days", "Last 90 Days", "All Time"])
        days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90, "All Time": 0}
        days_param = days_map.get(date_range, 30)

    # Check officer assigned territory without automatic demo assignment
    territories = orchestrator.get_officer_territory(officer_id)
    if not territories and current_user.role != UserRole.ADMINISTRATOR:
        st.warning("⚠️ No Geographic Territory Assigned: You have not yet been assigned an administrative territory. Please contact a platform administrator.")

    overview: FieldOfficerDashboardOverview = orchestrator.get_field_officer_dashboard(
        officer_id=officer_id,
        officer_name=officer_name,
        filter_crop=crop_param,
        filter_risk=risk_param,
        filter_status=status_param,
        date_range_days=days_param,
        force_refresh=force_refresh,
    )

    # Assigned Jurisdiction Banner
    dist_names = [t.district for t in overview.assigned_territory]
    st.markdown(
        f"""
        <div style="background-color: #E3F2FD; border-left: 5px solid #1976D2; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #0D47A1;">👮 Agronomist Portal: {overview.officer_name}</h3>
                    <p style="margin: 0.2rem 0 0 0; color: #1565C0; font-size: 0.95rem;">
                        <strong>Assigned Jurisdiction:</strong> {', '.join(dist_names) if dist_names else 'No districts assigned'} | 
                        <strong>Scope:</strong> State Agricultural Extension & Advisory
                    </p>
                </div>
                <div style="background-color: #BBDEFB; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: bold; color: #0D47A1; font-size: 0.85rem;">
                    {overview.data_freshness}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Cards
    m = overview.metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        ui_metric_card(
            m["officer_assigned_districts"].name,
            m["officer_assigned_districts"].display_value,
            icon="🗺️",
            color="#1565C0",
        )
    with c2:
        ui_metric_card(
            m["officer_registered_cases"].name,
            m["officer_registered_cases"].display_value,
            icon="📋",
            color="#2E7D32",
        )
    with c3:
        ui_metric_card(
            m["officer_high_risk_cases"].name,
            m["officer_high_risk_cases"].display_value,
            icon="🔴",
            color="#C62828",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        ui_metric_card(
            m["officer_field_visits_due"].name,
            m["officer_field_visits_due"].display_value,
            icon="🚗",
            color="#D84315",
        )
    with c5:
        ui_metric_card(
            m["officer_resolution_rate"].name,
            m["officer_resolution_rate"].display_value,
            icon="✅",
            color="#2E7D32",
        )
    with c6:
        ui_metric_card(
            m["officer_registered_farmers"].name,
            m["officer_registered_farmers"].display_value,
            icon="👨‍🌾",
            color="#6A1B9A",
        )

    ui_divider(style="dotted")

    # Risk and Crop Distribution Charts
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("### ⚠️ Outbreak & Risk Level Distribution")
        st.bar_chart(overview.risk_distribution)
    with col_ch2:
        st.markdown("### 🌾 Crop Incident Breakdown")
        if overview.crop_distribution:
            crop_dict = {cp.label: cp.value for cp in overview.crop_distribution}
            st.bar_chart(crop_dict)
        else:
            st.info("No crop incident records in this date window.")

    # Location Clusters with Small Cohort Privacy Masking
    if overview.village_clusters:
        st.markdown("### 📍 Regional Incident Clusters & Privacy Shield")
        st.caption("🔒 Privacy Rule: Clusters with fewer than 3 reported cases are anonymized to protect individual farmer identity.")
        cluster_rows = []
        for vc in overview.village_clusters:
            cluster_rows.append({
                "Territory / Location": vc["location"],
                "Total Cases": vc["case_count"],
                "High Risk Cases": vc["high_risk_count"],
                "Primary Crops": ", ".join(vc["primary_crops"]),
                "Privacy Status": "🔒 Anonymized (<3 cases)" if vc["privacy_masked"] else "🟢 Visible Cohort (≥3)",
                "Sample Farmers": ", ".join(vc["sample_farmers"]),
            })
        st.dataframe(cluster_rows, use_container_width=True)

    ui_divider(style="dotted")

    # Interactive Case Register & Agronomist Review Tool
    st.markdown("### 📋 Case Register & Field Follow-Up Management")
    if not overview.case_register:
        st.info("No advisory cases match the current filters.")
        return

    # Table of Cases
    case_summary_rows = []
    for c in overview.case_register:
        case_summary_rows.append({
            "Case ID": c.case_id,
            "Date": c.created_at[:10],
            "Farmer": c.farmer_name,
            "Crop": c.crop,
            "Location": c.location,
            "Risk": c.risk,
            "Visit Req.": "Yes" if c.field_visit_required else "No",
            "Status": c.status.title(),
        })
    st.dataframe(case_summary_rows, use_container_width=True)

    # Case Review and Follow-up Section
    st.markdown("#### ✍️ Agronomist Case Inspection & Follow-Up Audit Log")
    case_ids = [c.case_id for c in overview.case_register]
    selected_case_id = st.selectbox("Select Case to Audit / Follow Up", case_ids)

    selected_case = next((c for c in overview.case_register if c.case_id == selected_case_id), None)
    if selected_case:
        with st.expander("🔍 View Full Case Dossier", expanded=True):
            st.markdown(
                f"""
                - **Farmer Name:** {selected_case.farmer_name} | **Location:** {selected_case.location}
                - **Crop:** {selected_case.crop} | **Risk Level:** {selected_case.risk}
                - **Reported Symptoms:** {selected_case.symptoms or 'Standard checkup'}
                - **Advisory (Hindi):** {selected_case.advisory_hi or 'N/A'}
                - **Recommended 3-Step Action:** {', '.join(selected_case.steps) if selected_case.steps else 'N/A'}
                - **Weather Summary:** {selected_case.weather_summary or 'Normal'}
                - **Market Rate Summary:** {selected_case.market_summary or 'Normal'}
                """
            )

        # Audit Log Form
        with st.form(f"follow_up_form_{selected_case_id}"):
            col_fu1, col_fu2 = st.columns([2, 1])
            with col_fu1:
                notes = st.text_area("Field Visit Observations", placeholder="e.g. Visited plot in Karnal. Observed foliar lesions on lower canopy. Collected sample for local Krishi Vigyan Kendra.")
                action = st.text_input("Reported Action / Immediate Cultural Advisory", placeholder="e.g. Advised farmer to improve drainage and await formal expert diagnostic review.")
                st.caption("ℹ️ Note: Agronomic diagnosis verification and chemical treatment approvals are reserved for verified Agricultural Experts under Phase 8.")
            with col_fu2:
                new_st = st.selectbox("Update Case Status", ["pending", "in_progress", "reviewed", "resolved"], index=1)
                submit_fu = st.form_submit_button("📝 Record Field Visit & Observations")

            if submit_fu:
                if not notes.strip():
                    st.error("Please enter observation notes before submitting.")
                else:
                    try:
                        is_admin_actor = (current_user.role == UserRole.ADMINISTRATOR)
                        record = orchestrator.record_case_follow_up(
                            case_id=selected_case_id,
                            officer_id=officer_id,
                            officer_name=officer_name,
                            notes=notes.strip(),
                            action_taken=action.strip() if action else None,
                            new_status=new_st,
                            is_admin=is_admin_actor,
                        )
                        st.success(f"✅ Follow-up recorded! Status updated to '{new_st}'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record follow up: {e}")

        # Historical Follow-up Audit Trail
        is_admin_actor = (current_user.role == UserRole.ADMINISTRATOR)
        history = orchestrator.list_case_follow_ups(selected_case_id, officer_id=officer_id, is_admin=is_admin_actor)
        if history:
            st.markdown("##### 📜 Chronological Follow-Up Audit Log")
            for h in history:
                st.markdown(
                    f"""
                    <div style="background-color: #F5F5F5; border-left: 3px solid #1976D2; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #555;">
                            <strong>Officer: {h.officer_name}</strong>
                            <span>{h.created_at[:19]}</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.9rem;">{h.notes}</p>
                        {f'<span style="font-size: 0.8rem; color: #1B5E20;"><strong>Action:</strong> {h.action_taken}</span> | ' if h.action_taken else ''}
                        {f'<span style="font-size: 0.8rem; color: #E65100;"><strong>Transition:</strong> {h.status_transition}</span>' if h.status_transition else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==============================================================================
# 3. ADMINISTRATOR DASHBOARD SURFACE
# ==============================================================================

def _render_administrator_dashboard(orchestrator, current_user, force_refresh: bool) -> None:
    """Render platform health, integration telemetry, and administrative governance interface."""
    if not current_user or current_user.role != UserRole.ADMINISTRATOR:
        st.error("⛔ Access Forbidden: Platform Administrator credentials required.")
        return

    admin_id = current_user.user_id
    admin_name = current_user.username

    overview: AdminDashboardOverview = orchestrator.get_platform_admin_dashboard(
        admin_id=admin_id,
        admin_name=admin_name,
        force_refresh=force_refresh,
    )

    # Admin Header Banner
    st.markdown(
        f"""
        <div style="background-color: #212121; border-left: 5px solid #FF9800; padding: 1rem; border-radius: 6px; margin: 1rem 0; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #FFA726;">⚙️ Platform Observability & Governance: {overview.admin_name}</h3>
                    <p style="margin: 0.2rem 0 0 0; color: #B0BEC5; font-size: 0.95rem;">
                        Real-time infrastructure health, AI model telemetry, and multi-tenant data governance.
                    </p>
                </div>
                <div style="background-color: #37474F; padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: bold; color: #81C784; font-size: 0.85rem;">
                    {overview.data_freshness}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core System KPIs
    m = overview.metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui_metric_card(
            m["admin_total_users"].name,
            m["admin_total_users"].display_value,
            icon="👥",
            color="#1565C0",
        )
    with c2:
        ui_metric_card(
            m["admin_active_users_30d"].name,
            m["admin_active_users_30d"].display_value,
            icon="⚡",
            color="#2E7D32",
        )
    with c3:
        ui_metric_card(
            m["admin_platform_farms"].name,
            m["admin_platform_farms"].display_value,
            icon="🏡",
            color="#F57F17",
        )
    with c4:
        ui_metric_card(
            m["admin_total_advisories"].name,
            m["admin_total_advisories"].display_value,
            icon="📊",
            color="#6A1B9A",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        ui_metric_card(
            m["admin_system_uptime"].name,
            m["admin_system_uptime"].display_value,
            icon="⏱️",
            color="#2E7D32",
        )
    with c6:
        ui_metric_card(
            m["admin_api_error_rate"].name,
            m["admin_api_error_rate"].display_value,
            icon="🛡️",
            color="#2E7D32",
        )
    with c7:
        ui_metric_card(
            m["admin_data_quality_score"].name,
            m["admin_data_quality_score"].display_value,
            icon="💎",
            color="#00838F",
        )
    with c8:
        ui_metric_card(
            m["admin_platform_crops"].name,
            m["admin_platform_crops"].display_value,
            icon="🌱",
            color="#4CAF50",
        )

    ui_divider(style="dotted")

    # Integration Health Audit Table
    st.markdown("### 🔌 Live Service Integration Health Checks")
    health_rows = []
    for h in overview.integration_health:
        health_rows.append({
            "Service / Subsystem": h.service_name,
            "Health Status": "🟢 Active" if h.status == "active" else ("🟡 Mock/Fallback" if h.status == "mock_mode" else "🔴 Degraded"),
            "Latency (ms)": f"{h.latency_ms:.1f} ms",
            "Error Rate": f"{h.error_rate_pct:.1f}%",
            "Operational Audit Details": h.details,
            "Last Checked": h.last_checked[:19],
        })
    st.dataframe(health_rows, use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### 🧠 PyTorch Computer Vision Telemetry")
        mp = overview.model_performance
        st.markdown(
            f"""
            <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1565C0;">{mp.model_name} (v{mp.version})</h4>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Execution Device:</strong> {mp.device}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Average Inference Latency:</strong> {mp.avg_latency_ms:.1f} ms</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Inferences Executed:</strong> {mp.total_inferences}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Low Confidence Rate:</strong> {mp.low_confidence_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown("### 🛡️ Data Quality & Source Governance")
        dq = overview.data_quality
        st.markdown(
            f"""
            <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2E7D32;">Data Integrity Index: {dq.overall_quality_pct:.1f}%</h4>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Missing Market Records:</strong> {dq.missing_market_records}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Stale Weather Reports:</strong> {dq.stale_weather_records}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Quarantined Synthetic Fixtures:</strong> {dq.unverified_sources} isolated records</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem;"><strong>Incomplete Farmer Profiles:</strong> {dq.incomplete_profiles}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ui_divider(style="dotted")

    # Territory Provisioning Management Tool
    st.markdown("### 🗺️ Field Officer Territory Provisioning")
    st.caption("Assign geographic state and district jurisdiction to authorized field officers.")

    with st.form("territory_provision_form"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            t_officer = st.text_input("Officer User ID", value="usr_officer_demo")
        with col_t2:
            t_state = st.selectbox("State", ["Haryana", "Uttar Pradesh", "Punjab", "Rajasthan", "Madhya Pradesh", "Maharashtra"])
        with col_t3:
            t_district = st.text_input("District", value="Karnal")

        submit_t = st.form_submit_button("📍 Assign Geographic Jurisdiction")
        if submit_t:
            try:
                scope = orchestrator.assign_officer_territory(
                    officer_id=t_officer.strip(),
                    state=t_state,
                    district=t_district.strip(),
                )
                st.success(f"✅ Successfully assigned {scope.district}, {scope.state} to officer {scope.officer_id}!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to assign territory: {e}")
