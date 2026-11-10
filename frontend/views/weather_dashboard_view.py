"""Agricultural Weather Intelligence & Climate Risk Assessment View."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import streamlit as st
from backend.app.schemas.location import (
    PRESET_AGRICULTURAL_DISTRICTS,
    LocationPrecision,
    LocationRecord,
)
from backend.app.schemas.weather import WeatherReport
from backend.app.schemas.weather_alert import AlertStatus, HazardType, RiskLevel, WeatherAlert
from backend.app.services.weather.weather_intelligence import WeatherIntelligenceService
from frontend.components.location_map import render_farmer_location_map
from frontend.components.ui_widgets import (
    ui_page_header,
    ui_section_header,
    ui_divider,
    ui_card,
    ui_metric_card,
    ui_alert,
    ui_badge,
    ui_status_pill,
)
from frontend.components.icons import render_icon


def render_weather_dashboard_view():
    """Render comprehensive agricultural weather dashboard."""
    ui_page_header(
        title="Agricultural Weather Intelligence & Climate Risk",
        subtitle="Precision microclimate monitoring, phenology-aware weather risk indexing, and proactive spray advisories.",
        pills=["IMD Agromet Protocols", "7-Day Hourly Forecast", "Spray Window Matrix", "Terminal Heat & Frost"],
        icon_name="cloud-rain",
    )

    system = st.session_state.system
    weather_intel: WeatherIntelligenceService = system.weather_intel
    loc_manager = weather_intel.location_manager

    # -------------------------------------------------------------------------
    # 1. Location & Crop Context Selector
    # -------------------------------------------------------------------------
    ui_section_header("Farm Location & Crop Focus", "Select administrative district or pinpoint exact field coordinates.", "map-pin")
    col_loc1, col_loc2, col_loc3, col_loc4 = st.columns([1.5, 1.5, 1.2, 1.2])

    with col_loc1:
        states = loc_manager.get_supported_states()
        default_state_idx = states.index("Uttar Pradesh") if "Uttar Pradesh" in states else 0
        selected_state = st.selectbox("State", states, index=default_state_idx)

    with col_loc2:
        districts = loc_manager.get_districts_for_state(selected_state)
        default_dist_idx = districts.index("Varanasi") if "Varanasi" in districts else 0
        selected_district = st.selectbox("District", districts, index=default_dist_idx)

    with col_loc3:
        crops = ["tomato", "wheat", "rice", "cotton", "general"]
        selected_crop = st.selectbox("Crop Focus", crops, index=0, format_func=lambda x: x.title())

    with col_loc4:
        forecast_days = st.slider("Forecast Range", min_value=3, max_value=14, value=7, format="%d Days")

    # Resolve location
    current_loc: LocationRecord = loc_manager.resolve_preset_district(selected_state, selected_district)

    # Optional Map Click / GPS refinement in expander
    with st.expander("Refine Field Coordinates (Map Pin / GPS Consent)", expanded=False):
        st.caption("Select your exact field coordinates. If GPS is enabled with your consent, precise microclimate data is fetched.")
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            clicked_lat, clicked_lon = render_farmer_location_map(
                lat=current_loc.latitude,
                lon=current_loc.longitude,
                crop=selected_crop.title(),
                farmer_name=selected_district,
                height=260,
            )
            if clicked_lat != current_loc.latitude or clicked_lon != current_loc.longitude:
                current_loc = loc_manager.resolve_coordinates(
                    latitude=clicked_lat,
                    longitude=clicked_lon,
                    source="map_click",
                )
                ui_alert(f"Field pin updated: ({clicked_lat:.4f}°N, {clicked_lon:.4f}°E)", level="success")

        with col_m2:
            st.markdown("#### Spatial Precision")
            precision_badge = {
                "exact_gps": "Exact GPS (Field Level)",
                "manual_pin": "Map Pin Selection (~5 km)",
                "village_centroid": "Village Centroid",
                "district_centroid": "District Centroid (Regional)",
            }.get(current_loc.precision, "Regional Forecast")
            st.info(f"**Current Accuracy:** {precision_badge}")
            st.caption(f"Elevation: {current_loc.elevation_m or 100:.0f} m MSL")
            st.caption(f"Coordinates: {current_loc.latitude:.3f}°N, {current_loc.longitude:.3f}°E")

            gps_consent = st.checkbox("Allow GPS Device Geolocation", value=False)
            if gps_consent:
                st.caption("Privacy: GPS coordinates are processed transiently for weather models and never persisted without consent.")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 2. Fetch Weather Intelligence
    # -------------------------------------------------------------------------
    with st.spinner("Retrieving atmospheric models and evaluating agricultural risk..."):
        intel = weather_intel.get_weather_intelligence(
            location=current_loc,
            crop=selected_crop,
            forecast_days=forecast_days,
        )

    report: WeatherReport = intel["report"]
    risk_assessment = intel["risk_assessment"]
    active_alerts: List[WeatherAlert] = intel["active_alerts"]

    # -------------------------------------------------------------------------
    # 3. Active Agricultural Weather Alerts Banner
    # -------------------------------------------------------------------------
    ui_section_header("Active Agricultural Weather Alerts & Warnings", "Automated risk warnings derived from microclimate forecasts and crop phenology models.", "shield-alert")

    if active_alerts:
        for alert in active_alerts:
            sev_colors = {
                RiskLevel.EXTREME: ("#FEF2F2", "#DC2626", "EXTREME HAZARD"),
                RiskLevel.HIGH: ("#FFF7ED", "#EA580C", "HIGH RISK"),
                RiskLevel.MODERATE: ("#FEFCE8", "#CA8A04", "MODERATE RISK"),
                RiskLevel.LOW: ("#F0FDF4", "#16A34A", "LOW RISK"),
            }
            bg_color, border_color, sev_title = sev_colors.get(alert.risk_level, ("#F8FAFC", "#475569", "ALERT"))

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; border: 1px solid #E2E8F0; border-left-width: 5px; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <h4 style="margin: 0; color: {border_color}; font-size: 1.05rem; font-weight: 700;">{alert.title}</h4>
                        <span style="font-weight: 700; font-size: 0.78rem; letter-spacing: 0.05em; color: {border_color}; background: rgba(0,0,0,0.06); padding: 3px 8px; border-radius: 4px;">{sev_title}</span>
                    </div>
                    <p style="margin: 6px 0 4px 0; font-size: 0.95rem; color: #1E293B; line-height: 1.5;">{alert.summary}</p>
                    <p style="margin: 0; font-size: 0.82rem; color: #64748B;"><b>Biological & Climatic Basis:</b> {alert.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Mitigation Actions & Evidence Details ({alert.alert_id})", expanded=False):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("**Recommended Actions:**")
                    for act in alert.recommended_actions:
                        st.markdown(f"- {act}")
                with col_a2:
                    st.markdown("**Prohibited Actions:**")
                    if alert.prohibited_actions:
                        for proh in alert.prohibited_actions:
                            st.markdown(f"- {proh}")
                    else:
                        st.caption("No specific prohibited actions.")

                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; color: #64748B; background: #F8FAFC; border: 1px solid #E2E8F0; padding: 6px 10px; border-radius: 6px; margin-top: 6px;">
                        <b>Evidence Grounding:</b> Facts: <code>{', '.join(alert.evidence_ids)}</code> | Sources: <code>{', '.join(alert.source_ids)}</code> | Status: <b>{alert.status.value.upper()}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        ui_alert("No Severe Weather Alerts in Effect. Weather conditions are currently within standard physiological tolerances for this crop.", level="success")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 4. Current Atmospheric Conditions
    # -------------------------------------------------------------------------
    ui_section_header("Current Farm Microclimate", "Real-time atmospheric telemetry and surface observation measurements.", "activity")
    curr = report.current

    if curr:
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        col_c1.metric("Temperature", f"{curr.temperature_c:.1f} °C", delta=f"Feels like {curr.apparent_temperature_c:.1f}°C" if curr.apparent_temperature_c else None)
        col_c2.metric("Relative Humidity", f"{curr.relative_humidity_pct:.0f} %", delta=f"Dew pt {curr.dew_point_c:.1f}°C" if curr.dew_point_c else None)
        col_c3.metric("Precipitation", f"{curr.precipitation_mm:.1f} mm", delta="Current Rain")
        col_c4.metric("Wind Speed", f"{curr.wind_speed_kmh:.1f} km/h", delta=f"Gusts {curr.wind_gusts_kmh:.1f} km/h" if curr.wind_gusts_kmh else None)
        col_c5.metric("Condition", curr.condition_text, delta=f"Cloud {curr.cloud_cover_pct:.0f}%" if curr.cloud_cover_pct is not None else None)

        live_text = "Live" if report.provenance.is_live else "Cached / Degraded"
        st.caption(
            f"Meteorological Provenance: Source: `{report.provenance.provider}` | "
            f"NWP Model: `{report.provenance.model_name}` | "
            f"Dataset Type: `{report.provenance.source_type}` | "
            f"Live Connection: `{live_text}`"
        )
    else:
        ui_alert("Current weather observations are temporarily unavailable from meteorological service.", level="warning")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 5. Multi-Day Forecast (Daily Timeline Cards)
    # -------------------------------------------------------------------------
    ui_section_header(f"{forecast_days}-Day Agricultural Forecast", "Projected daily diurnal temperatures, precipitation accumulation, and wind speeds.", "calendar")

    if report.daily:
        cols = st.columns(min(len(report.daily), 7))
        for idx, day in enumerate(report.daily[:7]):
            with cols[idx]:
                date_obj = datetime.strptime(day.date_str, "%Y-%m-%d")
                day_label = date_obj.strftime("%a, %d %b")
                st.markdown(
                    f"""
                    <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 8px; text-align: center; background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="font-weight: 700; font-size: 0.85rem; color: #16532D; margin-bottom: 6px;">{day_label}</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{day.max_temperature_c:.0f}° / <span style="color: #64748B; font-weight: 500;">{day.min_temperature_c:.0f}°C</span></div>
                        <div style="font-size: 0.82rem; color: #0284C7; font-weight: 600; margin-top: 6px;">Precip: {day.precipitation_sum_mm:.1f} mm</div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2px;">Wind: {day.max_wind_speed_kmh:.0f} km/h</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">{f"ET0: {day.et0_evapotranspiration_mm:.1f} mm" if day.et0_evapotranspiration_mm is not None else "ET0: N/A"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 6. Historical Climate Reanalysis Comparison (ERA5 / ERA5-Land)
    # -------------------------------------------------------------------------
    ui_section_header("Climate Anomaly Analysis (ERA5 5-Year Climatological Baseline)", "Compares current forecast against the 5-year modeled ERA5 reanalysis normal for the exact same calendar period.", "bar-chart")

    hist = report.historical
    if hist:
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)

        t_anomaly = hist.temperature_anomaly_c
        t_delta_str = f"{'+' if t_anomaly > 0 else ''}{t_anomaly:.1f} °C vs 5yr normal"
        col_h1.metric("Current Period Mean Temp", f"{hist.current_period_mean_temperature_c:.1f} °C", delta=t_delta_str)

        r_departure = hist.rainfall_anomaly_pct
        r_delta_str = f"{'+' if r_departure > 0 else ''}{r_departure:.0f}% departure"
        col_h2.metric("Cumulative Rainfall", f"{hist.current_period_rainfall_mm:.1f} mm", delta=r_delta_str)

        col_h3.metric("Consecutive Dry Days", f"{hist.consecutive_dry_days} Days", delta="Dry Spell Risk" if hist.consecutive_dry_days >= 7 else "Normal")

        col_h4.metric("Baseline Mean Temp", f"{hist.baseline_mean_temperature_c:.1f} °C", delta="5-Year Normal")

        ui_alert(f"Agronomic Climate Interpretation: {hist.notes}", level="info")
    else:
        ui_alert("Historical climate baseline reanalysis is calculating.", level="info")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 7. Practical Agricultural Advisory Decisions
    # -------------------------------------------------------------------------
    ui_section_header("Practical Farm Decision Matrix", "Actionable recommendations on irrigation scheduling and crop protection spray windows.", "sprout")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Irrigation Scheduling")
        et0_today = report.daily[0].et0_evapotranspiration_mm if (report.daily and report.daily[0].et0_evapotranspiration_mm is not None) else None
        precip_3d = sum(d.precipitation_sum_mm for d in report.daily[:3]) if report.daily else 0.0

        if precip_3d >= 15.0:
            ui_alert(f"Hold Irrigation: Significant rain expected ({precip_3d:.1f} mm in next 3 days). Postpone irrigation to avoid waterlogging and runoff losses.", level="warning")
        elif et0_today is not None and et0_today >= 5.0:
            ui_alert(f"High Water Demand: Daily ET0 is high ({et0_today:.1f} mm/day). Schedule irrigation during early morning to minimize evaporative loss.", level="warning")
        elif et0_today is not None:
            ui_alert(f"Normal Irrigation: Daily ET0 is {et0_today:.1f} mm/day. Follow standard crop stage irrigation schedule.", level="success")
        else:
            ui_alert("Daily ET0 evapotranspiration not provided for today. Check root-zone soil moisture directly before scheduling irrigation.", level="info")

    with col_d2:
        st.markdown("#### Agrochemical Spray Window")
        max_wind_today = report.daily[0].max_wind_speed_kmh if (report.daily and report.daily[0].max_wind_speed_kmh is not None) else None
        precip_today = report.daily[0].precipitation_sum_mm if (report.daily and report.daily[0].precipitation_sum_mm is not None) else 0.0

        if precip_today >= 5.0:
            ui_alert("Do Not Spray: Rain expected today. Foliar sprays will be washed off, wasting inputs and risking runoff contamination.", level="error")
        elif max_wind_today is not None and max_wind_today >= 20.0:
            ui_alert(f"High Drift Hazard: Wind gusts reaching {max_wind_today:.0f} km/h. Postpone foliar spraying until early morning when winds are calm (< 12 km/h).", level="warning")
        elif max_wind_today is not None:
            ui_alert("Optimal Spray Window: Low wind speeds and clear skies make early morning ideal for necessary protective sprays.", level="success")
        else:
            ui_alert("Wind speed forecast not provided for today. Verify calm wind conditions (< 12 km/h) before spraying.", level="info")
