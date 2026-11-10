"""Streamlit View for AgriGuardian Agricultural Intelligence & Monitoring Dashboards.

Phase 7: Role-Based Monitoring Dashboard (Farmer, Field Officer, Administrator).
Zero-emoji compliant with enterprise SaaS styling.
"""

from typing import Optional
import streamlit as st
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.schemas.analytics import (
    AdminDashboardOverview,
    FarmerDashboardOverview,
    FieldOfficerDashboardOverview,
    MetricStatus,
)
from backend.app.schemas.farm import UserRole
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from frontend.components.ui_widgets import (
    ui_page_header,
    ui_section_header,
    ui_divider,
    ui_card,
    ui_metric_card,
    ui_alert,
    ui_badge,
)
from frontend.components.icons import render_icon


def safe_text(val):
    """Safely format values for HTML interpolation."""
    return "" if val is None else str(val)


def render_dashboard_view():
    """Render role-scoped dashboard surfaces based on the authenticated user's role."""
    settings = get_settings()
    if "system" not in st.session_state:
        st.session_state.system = AgriGuardianSystem(settings=settings)
    system: AgriGuardianSystem = st.session_state.system
    orchestrator = system.analytics_orchestrator

    # Determine user role from session
    current_user = st.session_state.get("current_user")
    current_farmer = st.session_state.get("current_farmer")

    ui_page_header(
        title="Unified Agricultural Intelligence & Monitoring",
        subtitle="Role-grounded operational analytics for Farmers, Regional Agronomists & Platform Administrators.",
        pills=["Role-Based Telemetry", "DPDP Privacy Shield", "Phenology Distribution", "System Health Matrix"],
        icon_name="layout-dashboard",
    )

    # Role Selection / Switcher with Strict Server-Side Role Enforcement
    if not current_user:
        ui_alert("Authentication Required: Please log in or register via 'My Farms & Crops' to access your role-based dashboard.", level="warning")
        st.info("For privacy and platform security, metrics and cases are scoped strictly to authenticated farmer, field officer, expert, or administrator accounts.")
        return

    # Determine permitted dashboards based on user role
    role_options = []
    if current_user.role == UserRole.FARMER:
        role_options = ["Farmer Dashboard (Personalized)"]
    elif current_user.role == UserRole.FIELD_OFFICER:
        role_options = ["Field Officer Dashboard (Regional Intelligence)"]
        if current_farmer:
            role_options.append("Farmer Dashboard (My Personal Farm)")
    elif current_user.role == UserRole.AGRONOMIST:
        role_options = ["Agricultural Expert Portal (Case Verification)"]
        if current_farmer:
            role_options.append("Farmer Dashboard (My Personal Farm)")
    elif current_user.role == UserRole.ADMINISTRATOR:
        role_options = [
            "Administrator Dashboard (Platform Telemetry)",
            "Field Officer Dashboard (Regional Intelligence)",
            "Agricultural Expert Portal (Case Verification)",
            "Farmer Dashboard (Personalized)",
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
        force_refresh = st.button("Force Refresh Cache", use_container_width=True)

    ui_divider(style="dotted")

    if "Farmer" in selected_dashboard:
        _render_farmer_dashboard(orchestrator, current_farmer, force_refresh)
    elif "Field Officer" in selected_dashboard:
        _render_field_officer_dashboard(orchestrator, current_user, force_refresh)
    elif "Administrator" in selected_dashboard:
        _render_administrator_dashboard(orchestrator, current_user, force_refresh)
    else:
        ui_alert("Agricultural Expert Portal is available in the dedicated Expert Review console.", level="info")


# ==============================================================================
# 1. FARMER DASHBOARD SURFACE
# ==============================================================================

def _render_farmer_dashboard(orchestrator, current_farmer, force_refresh: bool) -> None:
    """Render personalized farmer dashboard."""
    if not current_farmer:
        ui_alert("No Farmer Profile Found: Please log in with a registered farmer account via 'My Farms & Crops'.", level="warning")
        st.info("Dashboard land holdings, phenology, and tasks are isolated strictly to your authenticated farmer profile.")
        return

    farmer_id = current_farmer.farmer_id

    try:
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
        ui_alert(f"Failed to load farmer dashboard: {e}", level="error")
        return

    # Farmer Identity Banner
    st.markdown(
        f"""
        <div style="background-color: var(--surface); border-left: 4px solid var(--primary); border: 1px solid var(--border); border-left-width: 4px; padding: 1.1rem; border-radius: var(--radius-md); margin-bottom: 1.2rem; box-shadow: var(--shadow-sm);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: var(--primary-dark); font-size: 1.25rem;">Welcome, {overview.farmer.full_name}</h3>
                    <p style="margin: 0.3rem 0 0 0; color: var(--text-secondary); font-size: 0.95rem;">
                        <strong>Farm:</strong> {overview.active_farm.farm_name if overview.active_farm else 'Primary Holding'} &nbsp;|&nbsp; 
                        <strong>Location:</strong> {overview.active_farm.district if overview.active_farm else 'Territory'}, {overview.active_farm.state if overview.active_farm else 'India'} &nbsp;|&nbsp; 
                        <strong>Profile Completion:</strong> {overview.farmer.profile_completion_pct:.0f}%
                    </p>
                </div>
                <div style="background-color: var(--bg-card); padding: 0.4rem 0.8rem; border: 1px solid var(--border); border-radius: var(--radius-sm); font-weight: 600; color: var(--text-primary); font-size: 0.82rem;">
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
            icon_name="leaf",
        )
    with c2:
        ui_metric_card(
            m["farmer_total_acres"].name,
            m["farmer_total_acres"].display_value,
            icon_name="sprout",
        )
    with c3:
        ui_metric_card(
            m["farmer_active_crops"].name,
            m["farmer_active_crops"].display_value,
            icon_name="tractor",
        )
    with c4:
        ui_metric_card(
            m["farmer_due_tasks"].name,
            m["farmer_due_tasks"].display_value,
            icon_name="calendar",
            delta_color="inverse" if m["farmer_due_tasks"].status == MetricStatus.WARNING else "normal",
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        ui_metric_card(
            m["farmer_crop_health_cases"].name,
            m["farmer_crop_health_cases"].display_value,
            icon_name="microscope",
        )
    with c6:
        ui_metric_card(
            m["farmer_weather_risk_index"].name,
            m["farmer_weather_risk_index"].display_value,
            icon_name="cloud-rain",
        )
    with c7:
        ui_metric_card(
            m["farmer_market_price_bench"].name,
            m["farmer_market_price_bench"].display_value,
            icon_name="dollar-sign",
        )

    ui_divider(style="dotted")

    # Phenology and Crop Lifecycle Progress
    col_p1, col_p2 = st.columns([1.5, 1])
    with col_p1:
        ui_section_header("Active Crop Phenological Stages & Progress", "Current phenological phases, days since sowing, and target yield benchmarks.", "sprout")
        if overview.active_crop_stages:
            for s in overview.active_crop_stages:
                crop_label = f"{safe_text(s.get('crop_name'))} ({safe_text(s.get('variety') or 'Standard')})"
                season_label = safe_text(s.get('season'))
                days_label = safe_text(s.get('days_since_sowing'))
                stage_label = safe_text(str(s.get('current_stage', '')).replace('_', ' ').title())
                yield_label = safe_text(s.get('target_yield') or 'N/A')
                st.markdown(
                    f"""
                    <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.9rem; margin-bottom: 0.8rem; box-shadow: var(--shadow-sm);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.05rem; font-weight: bold; color: var(--primary-dark);">{crop_label}</span>
                                <span style="margin-left: 0.5rem; background: var(--bg-card); color: var(--text-secondary); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">
                                    {season_label}
                                </span>
                            </div>
                            <span style="font-weight: bold; color: var(--primary); font-size: 0.9rem;">
                                {days_label} Days Since Sowing
                            </span>
                        </div>
                        <p style="margin: 0.4rem 0 0.2rem 0; font-size: 0.9rem; color: var(--text-secondary);">
                            <strong>Current Stage:</strong> <span style="color: #D97706; font-weight: 600;">{stage_label}</span> &nbsp;|&nbsp; 
                            <strong>Target Yield:</strong> {yield_label} q/acre
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            ui_alert("No active crop cycles currently sown. Add one under 'My Farms & Crops'.", level="info")

    with col_p2:
        ui_section_header("Crop Area Distribution", "Acreage distribution across active plots.", "bar-chart")
        if overview.crop_distribution:
            chart_data = {cp.label: cp.value for cp in overview.crop_distribution}
            st.bar_chart(chart_data)
        else:
            ui_alert("Crop area distribution will appear once crops are sown.", level="info")

    # Grounded Agronomic Insights
    if overview.insights:
        ui_section_header("Grounded Agronomic & Climate Insights", "Evidence-backed advisory notices derived from authoritative agricultural sources.", "book-open")
        for ins in overview.insights:
            ins_title = safe_text(ins.title)
            ins_source = safe_text(ins.source)
            ins_desc = safe_text(ins.description)
            level = "error" if ins.severity == "critical" else ("warning" if ins.severity == "warning" else "info")
            action_text = f" — Action: {safe_text(ins.suggested_action)}" if ins.suggested_action else ""
            ui_alert(f"**{ins_title}** ({ins_source}): {ins_desc}{action_text}", level=level)

    # Due Farm Activities Checklist
    ui_divider(style="dotted")
    col_a1, col_a2 = st.columns([1, 1])

    with col_a1:
        ui_section_header("Upcoming & Due Farm Activities", "Scheduled agronomic field operations.", "calendar")
        if overview.due_activities:
            for act in overview.due_activities:
                status_color = "#DC2626" if act.status.value == "overdue" else "#16532D"
                act_title = safe_text(act.title)
                act_status = safe_text(act.status.value.upper())
                act_type = safe_text(act.activity_type.value.title())
                act_due = safe_text(act.due_date.isoformat() if act.due_date else 'Immediate')
                st.markdown(
                    f"""
                    <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; box-shadow: var(--shadow-sm);">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>{act_title}</strong>
                            <span style="color: {status_color}; font-weight: bold; font-size: 0.8rem;">{act_status}</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>Type:</strong> {act_type} &nbsp;|&nbsp; 
                            <strong>Due:</strong> {act_due}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            ui_alert("All scheduled farm tasks are completed! No pending activities.", level="success")

    with col_a2:
        ui_section_header("Recent Crop Health & Advisory Cases", "Recently evaluated foliar symptoms and diagnostic cases.", "microscope")
        if overview.recent_health_cases:
            for case in overview.recent_health_cases:
                risk_color = "#DC2626" if case.risk == "High" else ("#D97706" if case.risk == "Medium" else "#16532D")
                c_id = safe_text(case.case_id)
                c_crop = safe_text(case.crop)
                c_risk = safe_text(case.risk.upper())
                c_symp = safe_text(case.symptoms[:50])
                c_status = safe_text(case.status.title())
                st.markdown(
                    f"""
                    <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; box-shadow: var(--shadow-sm);">
                        <div style="display: flex; justify-content: space-between;">
                            <strong>Case {c_id} — {c_crop}</strong>
                            <span style="color: {risk_color}; font-weight: bold; font-size: 0.8rem;">{c_risk} RISK</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>Symptoms:</strong> {c_symp}... &nbsp;|&nbsp; <strong>Status:</strong> {c_status}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            ui_alert("No crop health cases reported yet. Use the Voice Assistant or Diagnosis tool to run checks.", level="info")


# ==============================================================================
# 2. FIELD OFFICER DASHBOARD SURFACE
# ==============================================================================

def _render_field_officer_dashboard(orchestrator, current_user, force_refresh: bool) -> None:
    """Render regional monitoring, case register, and agronomist follow-up interface."""
    if not current_user or current_user.role not in (UserRole.FIELD_OFFICER, UserRole.ADMINISTRATOR):
        ui_alert("Access Forbidden: Field Officer or Administrator credentials required.", level="error")
        return

    officer_id = current_user.user_id
    officer_name = current_user.username

    # Filter Bar
    ui_section_header("Regional Telemetry Filters", "Filter incoming field advisory cases across crops, risk levels, and date ranges.", "search")
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
        ui_alert("No Geographic Territory Assigned: You have not yet been assigned an administrative territory. Please contact a platform administrator.", level="warning")

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
        <div style="background-color: var(--surface); border-left: 4px solid var(--primary); border: 1px solid var(--border); border-left-width: 4px; padding: 1.1rem; border-radius: var(--radius-md); margin: 1rem 0; box-shadow: var(--shadow-sm);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: var(--primary-dark); font-size: 1.25rem;">Agronomist Portal: {overview.officer_name}</h3>
                    <p style="margin: 0.3rem 0 0 0; color: var(--text-secondary); font-size: 0.95rem;">
                        <strong>Assigned Jurisdiction:</strong> {', '.join(dist_names) if dist_names else 'No districts assigned'} &nbsp;|&nbsp; 
                        <strong>Scope:</strong> State Agricultural Extension & Advisory
                    </p>
                </div>
                <div style="background-color: var(--bg-card); padding: 0.4rem 0.8rem; border: 1px solid var(--border); border-radius: var(--radius-sm); font-weight: 600; color: var(--text-primary); font-size: 0.82rem;">
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
            icon_name="map-pin",
        )
    with c2:
        ui_metric_card(
            m["officer_registered_cases"].name,
            m["officer_registered_cases"].display_value,
            icon_name="file-text",
        )
    with c3:
        ui_metric_card(
            m["officer_high_risk_cases"].name,
            m["officer_high_risk_cases"].display_value,
            icon_name="shield-alert",
            delta_color="inverse",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        ui_metric_card(
            m["officer_field_visits_due"].name,
            m["officer_field_visits_due"].display_value,
            icon_name="calendar",
        )
    with c5:
        ui_metric_card(
            m["officer_resolution_rate"].name,
            m["officer_resolution_rate"].display_value,
            icon_name="check-circle",
        )
    with c6:
        ui_metric_card(
            m["officer_registered_farmers"].name,
            m["officer_registered_farmers"].display_value,
            icon_name="user-check",
        )

    ui_divider(style="dotted")

    # Risk and Crop Distribution Charts
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        ui_section_header("Outbreak & Risk Level Distribution", "Incident severity breakdown across monitored territory.", "shield-alert")
        st.bar_chart(overview.risk_distribution)
    with col_ch2:
        ui_section_header("Crop Incident Breakdown", "Pathological cases categorized by affected crop species.", "sprout")
        if overview.crop_distribution:
            crop_dict = {cp.label: cp.value for cp in overview.crop_distribution}
            st.bar_chart(crop_dict)
        else:
            ui_alert("No crop incident records in this date window.", level="info")

    # Location Clusters with Small Cohort Privacy Masking
    if overview.village_clusters:
        ui_section_header("Regional Incident Clusters & Privacy Shield", "Clusters with fewer than 3 reported cases are anonymized to protect individual farmer identity.", "shield-check")
        cluster_rows = []
        for vc in overview.village_clusters:
            cluster_rows.append({
                "Territory / Location": vc["location"],
                "Total Cases": vc["case_count"],
                "High Risk Cases": vc["high_risk_count"],
                "Primary Crops": ", ".join(vc["primary_crops"]),
                "Privacy Status": "Anonymized (<3 cases)" if vc["privacy_masked"] else "Visible Cohort (>=3)",
                "Sample Farmers": ", ".join(vc["sample_farmers"]),
            })
        st.dataframe(cluster_rows, use_container_width=True)

    ui_divider(style="dotted")

    # Interactive Case Register & Agronomist Review Tool
    ui_section_header("Case Register & Field Follow-Up Management", "Review and audit incoming farmer inquiries from assigned jurisdiction.", "file-text")
    if not overview.case_register:
        ui_alert("No advisory cases match the current filters.", level="info")
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
    st.markdown("#### Agronomist Case Inspection & Follow-Up Audit Log")
    case_ids = [c.case_id for c in overview.case_register]
    selected_case_id = st.selectbox("Select Case to Audit / Follow Up", case_ids)

    selected_case = next((c for c in overview.case_register if c.case_id == selected_case_id), None)
    if selected_case:
        with st.expander("View Full Case Dossier", expanded=True):
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
                st.caption("Note: Agronomic diagnosis verification and chemical treatment approvals are reserved for verified Agricultural Experts under Phase 8.")
            with col_fu2:
                new_st = st.selectbox("Update Case Status", ["pending", "in_progress", "reviewed", "resolved"], index=1)
                submit_fu = st.form_submit_button("Record Field Visit & Observations")

            if submit_fu:
                if not notes.strip():
                    ui_alert("Please enter observation notes before submitting.", level="error")
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
                        ui_alert(f"Follow-up recorded! Status updated to '{new_st}'.", level="success")
                        st.rerun()
                    except Exception as e:
                        ui_alert(f"Failed to record follow up: {e}", level="error")

        # Historical Follow-up Audit Trail
        is_admin_actor = (current_user.role == UserRole.ADMINISTRATOR)
        history = orchestrator.list_case_follow_ups(selected_case_id, officer_id=officer_id, is_admin=is_admin_actor)
        if history:
            st.markdown("##### Chronological Follow-Up Audit Log")
            for h in history:
                st.markdown(
                    f"""
                    <div style="background-color: var(--surface); border-left: 3px solid var(--primary); border: 1px solid var(--border); border-left-width: 3px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>Officer: {h.officer_name}</strong>
                            <span>{h.created_at[:19]}</span>
                        </div>
                        <p style="margin: 0.2rem 0; font-size: 0.9rem; color: var(--text-primary);">{h.notes}</p>
                        {f'<span style="font-size: 0.8rem; color: var(--primary-dark);"><strong>Action:</strong> {h.action_taken}</span> | ' if h.action_taken else ''}
                        {f'<span style="font-size: 0.8rem; color: #D97706;"><strong>Transition:</strong> {h.status_transition}</span>' if h.status_transition else ''}
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
        ui_alert("Access Forbidden: Platform Administrator credentials required.", level="error")
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
        <div style="background-color: var(--primary-dark); border-left: 4px solid var(--primary-light); border-radius: var(--radius-md); padding: 1.1rem; margin: 1rem 0; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: white; font-size: 1.25rem;">Platform Observability & Governance: {overview.admin_name}</h3>
                    <p style="margin: 0.3rem 0 0 0; color: rgba(255,255,255,0.85); font-size: 0.95rem;">
                        Real-time infrastructure health, AI model telemetry, and multi-tenant data governance.
                    </p>
                </div>
                <div style="background-color: rgba(255,255,255,0.15); padding: 0.4rem 0.8rem; border-radius: var(--radius-sm); font-weight: 600; color: white; font-size: 0.82rem;">
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
            icon_name="user-check",
        )
    with c2:
        ui_metric_card(
            m["admin_active_users_30d"].name,
            m["admin_active_users_30d"].display_value,
            icon_name="activity",
        )
    with c3:
        ui_metric_card(
            m["admin_platform_farms"].name,
            m["admin_platform_farms"].display_value,
            icon_name="leaf",
        )
    with c4:
        ui_metric_card(
            m["admin_total_advisories"].name,
            m["admin_total_advisories"].display_value,
            icon_name="bar-chart",
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        ui_metric_card(
            m["admin_system_uptime"].name,
            m["admin_system_uptime"].display_value,
            icon_name="activity",
        )
    with c6:
        ui_metric_card(
            m["admin_api_error_rate"].name,
            m["admin_api_error_rate"].display_value,
            icon_name="shield-check",
        )
    with c7:
        ui_metric_card(
            m["admin_data_quality_score"].name,
            m["admin_data_quality_score"].display_value,
            icon_name="award",
        )
    with c8:
        ui_metric_card(
            m["admin_platform_crops"].name,
            m["admin_platform_crops"].display_value,
            icon_name="sprout",
        )

    ui_divider(style="dotted")

    # Integration Health Audit Table
    ui_section_header("Live Service Integration Health Checks", "Active status, latencies, and connectivity telemetry across platform microservices.", "activity")
    health_rows = []
    for h in overview.integration_health:
        health_rows.append({
            "Service / Subsystem": h.service_name,
            "Health Status": "Active" if h.status == "active" else ("Mock/Fallback" if h.status == "mock_mode" else "Degraded"),
            "Latency (ms)": f"{h.latency_ms:.1f} ms",
            "Error Rate": f"{h.error_rate_pct:.1f}%",
            "Operational Audit Details": h.details,
            "Last Checked": h.last_checked[:19],
        })
    st.dataframe(health_rows, use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        ui_section_header("PyTorch Computer Vision Telemetry", "Deep learning model runtime diagnostics and inference latencies.", "microscope")
        mp = overview.model_performance
        st.markdown(
            f"""
            <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem; box-shadow: var(--shadow-sm);">
                <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-dark);">{mp.model_name} (v{mp.version})</h4>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Execution Device:</strong> {mp.device}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Average Inference Latency:</strong> {mp.avg_latency_ms:.1f} ms</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Inferences Executed:</strong> {mp.total_inferences}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Low Confidence Rate:</strong> {mp.low_confidence_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        ui_section_header("Data Quality & Source Governance", "Integrity scoring and unverified fixture quarantine metrics.", "shield-check")
        dq = overview.data_quality
        st.markdown(
            f"""
            <div style="background-color: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem; box-shadow: var(--shadow-sm);">
                <h4 style="margin: 0 0 0.5rem 0; color: var(--primary-dark);">Data Integrity Index: {dq.overall_quality_pct:.1f}%</h4>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Missing Market Records:</strong> {dq.missing_market_records}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Stale Weather Reports:</strong> {dq.stale_weather_records}</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Quarantined Synthetic Fixtures:</strong> {dq.unverified_sources} isolated records</p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: var(--text-secondary);"><strong>Incomplete Farmer Profiles:</strong> {dq.incomplete_profiles}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ui_divider(style="dotted")

    # Territory Provisioning Management Tool
    ui_section_header("Field Officer Territory Provisioning", "Assign geographic state and district jurisdiction to authorized field officers.", "map-pin")

    with st.form("territory_provision_form"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            t_officer = st.text_input("Officer User ID", value="usr_officer_demo")
        with col_t2:
            t_state = st.selectbox("State", ["Haryana", "Uttar Pradesh", "Punjab", "Rajasthan", "Madhya Pradesh", "Maharashtra"])
        with col_t3:
            t_district = st.text_input("District", value="Karnal")

        submit_t = st.form_submit_button("Assign Geographic Jurisdiction")
        if submit_t:
            try:
                scope = orchestrator.assign_officer_territory(
                    officer_id=t_officer.strip(),
                    state=t_state,
                    district=t_district.strip(),
                )
                ui_alert(f"Successfully assigned {scope.district}, {scope.state} to officer {scope.officer_id}!", level="success")
                st.rerun()
            except Exception as e:
                ui_alert(f"Failed to assign territory: {e}", level="error")
