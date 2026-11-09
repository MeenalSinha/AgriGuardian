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
from frontend.components.ui_widgets import ui_divider


def render_weather_dashboard_view():
    """Render comprehensive agricultural weather dashboard."""
    st.markdown(
        """
        <div class="main-header">
            <h2>🌦️ Agricultural Weather Intelligence & Climate Risk</h2>
            <p>मौसम निगरानी, फसल-विशिष्ट जलवायु जोखिम आकलन एवं पूर्व चेतावनी</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    system = st.session_state.system
    weather_intel: WeatherIntelligenceService = system.weather_intel
    loc_manager = weather_intel.location_manager

    # -------------------------------------------------------------------------
    # 1. Location & Crop Context Selector
    # -------------------------------------------------------------------------
    st.markdown("### 📍 Farm Location & Crop Focus")
    col_loc1, col_loc2, col_loc3, col_loc4 = st.columns([1.5, 1.5, 1.2, 1.2])

    with col_loc1:
        states = loc_manager.get_supported_states()
        default_state_idx = states.index("Uttar Pradesh") if "Uttar Pradesh" in states else 0
        selected_state = st.selectbox("State (राज्य)", states, index=default_state_idx)

    with col_loc2:
        districts = loc_manager.get_districts_for_state(selected_state)
        default_dist_idx = districts.index("Varanasi") if "Varanasi" in districts else 0
        selected_district = st.selectbox("District (ज़िला)", districts, index=default_dist_idx)

    with col_loc3:
        crops = ["tomato", "wheat", "rice", "cotton", "general"]
        selected_crop = st.selectbox("Crop Focus (फ़सल)", crops, index=0, format_func=lambda x: x.title())

    with col_loc4:
        forecast_days = st.slider("Forecast Days", min_value=3, max_value=14, value=7)

    # Resolve location
    current_loc: LocationRecord = loc_manager.resolve_preset_district(selected_state, selected_district)

    # Optional Map Click / GPS refinement in expander
    with st.expander("🗺️ Refine Field Coordinates (Map Pin / GPS Consent)", expanded=False):
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
                st.success(f"📍 Field pin updated: ({clicked_lat:.4f}°N, {clicked_lon:.4f}°E)")

        with col_m2:
            st.markdown("#### Spatial Precision")
            precision_badge = {
                "exact_gps": "🟢 Exact GPS (Field Level)",
                "manual_pin": "🟡 Map Pin Selection (~5 km)",
                "village_centroid": "🔵 Village Centroid",
                "district_centroid": "⚪ District Centroid (Regional)",
            }.get(current_loc.precision, "Regional Forecast")
            st.info(f"**Current Accuracy:**\n{precision_badge}")
            st.caption(f"Elevation: {current_loc.elevation_m or 100:.0f} m MSL")
            st.caption(f"Coordinates: {current_loc.latitude:.3f}°N, {current_loc.longitude:.3f}°E")

            gps_consent = st.checkbox("Allow GPS Device Geolocation", value=False)
            if gps_consent:
                st.caption("🔒 Privacy: GPS coordinates are processed transiently for weather models and never sold.")

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
    st.markdown("### 🚨 Active Agricultural Weather Alerts & Warnings")

    if active_alerts:
        for alert in active_alerts:
            # Color-coded severity banner
            sev_colors = {
                RiskLevel.EXTREME: ("#FFEBEE", "#B71C1C", "🔴 EXTREME HAZARD"),
                RiskLevel.HIGH: ("#FFF3E0", "#E65100", "🟠 HIGH RISK"),
                RiskLevel.MODERATE: ("#FFFDE7", "#F57F17", "🟡 MODERATE RISK"),
                RiskLevel.LOW: ("#E8F5E9", "#2E7D32", "🟢 LOW RISK"),
            }
            bg_color, border_color, sev_title = sev_colors.get(alert.risk_level, ("#F5F5F5", "#333", "ALERT"))

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; border-left: 6px solid {border_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: {border_color};">{alert.title}</h4>
                        <span style="font-weight: 700; color: {border_color}; background: rgba(0,0,0,0.06); padding: 3px 8px; border-radius: 4px;">{sev_title}</span>
                    </div>
                    <p style="margin: 8px 0 4px 0; font-size: 1.0rem; color: #212121;">{alert.summary}</p>
                    <p style="margin: 0; font-size: 0.85rem; color: #616161;"><b>Biological & Climatic Basis:</b> {alert.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"📋 Mitigation Actions & Evidence Details ({alert.alert_id})", expanded=False):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("**✅ Recommended Actions:**")
                    for act in alert.recommended_actions:
                        st.markdown(f"- {act}")
                with col_a2:
                    st.markdown("**🚫 Prohibited Actions:**")
                    if alert.prohibited_actions:
                        for proh in alert.prohibited_actions:
                            st.markdown(f"- ⚠️ {proh}")
                    else:
                        st.caption("No specific prohibited actions.")

                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; color: #555; background: #fafafa; padding: 6px 10px; border-radius: 6px; margin-top: 6px;">
                        <b>Evidence Grounding:</b> Facts: <code>{', '.join(alert.evidence_ids)}</code> | Sources: <code>{', '.join(alert.source_ids)}</code> | Status: <b>{alert.status.value.upper()}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("✅ **No Severe Weather Alerts in Effect.** Weather conditions are currently within standard physiological tolerances for this crop.")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 4. Current Atmospheric Conditions
    # -------------------------------------------------------------------------
    st.markdown("### 🌡️ Current Farm Microclimate")
    curr = report.current

    if curr:
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        col_c1.metric("Temperature", f"{curr.temperature_c:.1f} °C", delta=f"Feels like {curr.apparent_temperature_c:.1f}°C" if curr.apparent_temperature_c else None)
        col_c2.metric("Relative Humidity", f"{curr.relative_humidity_pct:.0f} %", delta=f"Dew pt {curr.dew_point_c:.1f}°C" if curr.dew_point_c else None)
        col_c3.metric("Precipitation", f"{curr.precipitation_mm:.1f} mm", delta="Current Rain")
        col_c4.metric("Wind Speed", f"{curr.wind_speed_kmh:.1f} km/h", delta=f"Gusts {curr.wind_gusts_kmh:.1f} km/h" if curr.wind_gusts_kmh else None)
        col_c5.metric("Condition", curr.condition_text, delta=f"Cloud {curr.cloud_cover_pct:.0f}%" if curr.cloud_cover_pct is not None else None)

        st.caption(
            f"📡 **Meteorological Provenance:** Source: `{report.provenance.provider}` | "
            f"NWP Model: `{report.provenance.model_name}` | "
            f"Dataset Type: `{report.provenance.source_type}` | "
            f"Live Connection: `{'🟢 Yes' if report.provenance.is_live else '🟡 Cached / Degraded'}`"
        )
    else:
        st.warning("Current weather observations are temporarily unavailable from meteorological service.")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 5. Multi-Day Forecast (Daily Timeline Cards)
    # -------------------------------------------------------------------------
    st.markdown(f"### 📅 {forecast_days}-Day Agricultural Forecast")

    if report.daily:
        cols = st.columns(min(len(report.daily), 7))
        for idx, day in enumerate(report.daily[:7]):
            with cols[idx]:
                date_obj = datetime.strptime(day.date_str, "%Y-%m-%d")
                day_label = date_obj.strftime("%a, %d %b")
                st.markdown(
                    f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; text-align: center; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-weight: 700; font-size: 0.85rem; color: #1B5E20;">{day_label}</div>
                        <div style="font-size: 1.4rem; margin: 4px 0;">🌤️</div>
                        <div style="font-size: 0.95rem; font-weight: 600;">{day.max_temperature_c:.0f}° / <span style="color: #666;">{day.min_temperature_c:.0f}°C</span></div>
                        <div style="font-size: 0.8rem; color: #1565C0; margin-top: 4px;">💧 {day.precipitation_sum_mm:.1f} mm</div>
                        <div style="font-size: 0.75rem; color: #555;">💨 {day.max_wind_speed_kmh:.0f} km/h</div>
                        <div style="font-size: 0.75rem; color: #888; margin-top: 2px;">ET₀: {day.et0_evapotranspiration_mm or 4.0:.1f} mm</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 6. Historical Climate Reanalysis Comparison (ERA5 / ERA5-Land)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Climate Anomaly Analysis (ERA5 5-Year Climatological Baseline)")
    st.caption("Compares current forecast against the 5-year modeled ERA5 reanalysis normal for the exact same calendar period. Note: Reanalysis data is modeled and gridded, not physical on-farm station observations.")

    hist = report.historical
    if hist:
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)

        t_anomaly = hist.temperature_anomaly_c
        t_delta_str = f"{'+' if t_anomaly > 0 else ''}{t_anomaly:.1f} °C vs 5yr normal"
        col_h1.metric("Current Period Mean Temp", f"{hist.current_period_mean_temperature_c:.1f} °C", delta=t_delta_str)

        r_departure = hist.rainfall_anomaly_pct
        r_delta_str = f"{'+' if r_departure > 0 else ''}{r_departure:.0f}% departure"
        col_h2.metric("Cumulative Rainfall", f"{hist.current_period_rainfall_mm:.1f} mm", delta=r_delta_str)

        col_h3.metric("Consecutive Dry Days", f"{hist.consecutive_dry_days} Days", delta="IMD Dry Spell Risk" if hist.consecutive_dry_days >= 7 else "Normal")

        col_h4.metric("Baseline Mean Temp", f"{hist.baseline_mean_temperature_c:.1f} °C", delta="5-Year Normal")

        st.info(f"💡 **Agronomic Climate Interpretation:** {hist.notes}")
    else:
        st.info("Historical climate baseline reanalysis is calculating.")

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # 7. Practical Agricultural Advisory Decisions
    # -------------------------------------------------------------------------
    st.markdown("### 🌾 Practical Farm Decision Matrix")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 💧 Irrigation Scheduling")
        et0_today = report.daily[0].et0_evapotranspiration_mm if report.daily and report.daily[0].et0_evapotranspiration_mm else 4.2
        precip_3d = sum(d.precipitation_sum_mm for d in report.daily[:3]) if report.daily else 0.0

        if precip_3d >= 15.0:
            st.warning(f"🌧️ **Hold Irrigation:** Significant rain expected ({precip_3d:.1f} mm in next 3 days). Postpone irrigation to avoid waterlogging and runoff losses.")
        elif et0_today >= 5.0:
            st.warning(f"☀️ **High Water Demand:** Daily ET₀ is high ({et0_today:.1f} mm/day). Schedule irrigation during early morning to minimize evaporative loss.")
        else:
            st.success(f"💧 **Normal Irrigation:** Daily ET₀ is {et0_today:.1f} mm/day. Follow standard crop stage irrigation schedule.")

    with col_d2:
        st.markdown("#### 🛡️ Agrochemical Spray Window")
        max_wind_today = report.daily[0].max_wind_speed_kmh if report.daily else 10.0
        precip_today = report.daily[0].precipitation_sum_mm if report.daily else 0.0

        if precip_today >= 5.0:
            st.error("🚫 **DO NOT Spray:** Rain expected today. Foliar sprays will be washed off, wasting inputs and risking environmental contamination.")
        elif max_wind_today >= 20.0:
            st.warning(f"💨 **High Drift Hazard:** Wind gusts reaching {max_wind_today:.0f} km/h. Postpone foliar spraying until early morning when winds are calm (< 12 km/h).")
        else:
            st.success("✅ **Optimal Spray Window:** Low wind speeds and clear skies make early morning ideal for necessary protective sprays.")
