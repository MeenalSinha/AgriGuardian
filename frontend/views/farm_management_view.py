"""Streamlit View for Farmer Identity, Multi-Farm Management, Crop Lifecycle Tracking, Soil Health & Activities."""

from datetime import datetime
import streamlit as st

from backend.app.core.config import SUPPORTED_LANGUAGES, get_settings
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycleCreate,
    CropGrowthStage,
    CropSeason,
    CropStageTransitionRequest,
    FarmActivityCreate,
    FarmCreate,
    FarmerProfileUpdate,
    IrrigationMethod,
    IrrigationProfileCreate,
    LiteracyPreference,
    OwnershipType,
    PlotCreate,
    PumpingSource,
    SoilProfileCreate,
    SoilType,
    UserLoginRequest,
    UserRegistrationRequest,
    WaterAvailability,
)
from backend.app.services.auth.auth_service import AuthService, AuthenticationError
from backend.app.services.farm.crop_calendar_service import CropCalendarService
from backend.app.services.farm.crop_cycle_service import CropCycleService
from backend.app.services.farm.farm_context_service import FarmContextService
from backend.app.services.farm.farm_service import FarmService
from backend.app.services.farm.farmer_profile_service import FarmerProfileService
from backend.app.services.farm.plot_service import PlotService
from backend.app.services.farm.soil_irrigation_service import SoilIrrigationService
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider, ui_metric_card


def get_services():
    """Retrieve or initialize singleton service bundle."""
    settings = get_settings()
    auth_srv = AuthService(settings=settings)
    profile_srv = FarmerProfileService()
    farm_srv = FarmService()
    plot_srv = PlotService()
    cycle_srv = CropCycleService()
    soil_srv = SoilIrrigationService()
    calendar_srv = CropCalendarService()
    context_srv = FarmContextService(
        profile_service=profile_srv,
        farm_service=farm_srv,
        plot_service=plot_srv,
        cycle_service=cycle_srv,
        soil_service=soil_srv,
        calendar_service=calendar_srv,
    )
    return auth_srv, profile_srv, farm_srv, plot_srv, cycle_srv, soil_srv, calendar_srv, context_srv


def render_farm_management_view():
    """Render comprehensive farmer profile, multi-farm, crop tracking, soil, and calendar dashboard."""
    st.markdown("## 🌾 Farmer Identity & Farm Management")
    st.caption("Personalized Farm Profiles • Multi-Farm Management • Growth Stage Tracking • Soil & Crop Calendar")

    auth_srv, profile_srv, farm_srv, plot_srv, cycle_srv, soil_srv, calendar_srv, context_srv = get_services()

    # Session State Initialization
    if "farmer_token" not in st.session_state:
        st.session_state.farmer_token = None
    if "current_farmer" not in st.session_state:
        st.session_state.current_farmer = None
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # Check authentication state
    if not st.session_state.farmer_token or not st.session_state.current_farmer:
        render_auth_gateway(auth_srv)
        return

    # Authenticated Farmer View
    farmer = profile_srv.get_profile(st.session_state.current_farmer.farmer_id)
    if not farmer:
        st.session_state.farmer_token = None
        st.session_state.current_farmer = None
        st.rerun()

    st.session_state.current_farmer = farmer

    # Header Card with Completion Gauge & Language
    render_farmer_header(farmer, profile_srv)

    ui_divider(style="dotted")

    # Main Tabs
    tab_farms, tab_crops, tab_soil, tab_calendar = st.tabs([
        "🏡 My Farms & Plots",
        "🌱 Active Crop Tracker",
        "🧪 Soil & Irrigation",
        "📅 Crop Calendar & Tasks",
    ])

    with tab_farms:
        render_farms_and_plots_tab(farmer, farm_srv, plot_srv)

    with tab_crops:
        render_crop_lifecycle_tab(farmer, farm_srv, plot_srv, cycle_srv)

    with tab_soil:
        render_soil_irrigation_tab(farmer, farm_srv, plot_srv, soil_srv)

    with tab_calendar:
        render_calendar_activities_tab(farmer, farm_srv, cycle_srv, calendar_srv)


def render_auth_gateway(auth_srv: AuthService):
    """Render registration and login interface when no session is active."""
    st.info("👋 Welcome! Please log in to access your personalized farms, active crop stages, soil records, and advisory.")

    auth_tab1, auth_tab2 = st.tabs(["🔑 Farmer Login", "📝 New Farmer Registration"])

    with auth_tab1:
        st.subheader("Login to AgriGuardian")
        with st.form("login_form"):
            username = st.text_input("Username or Mobile Number", placeholder="e.g. 9876543210 or ramesh_patel")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In 🚀", use_container_width=True)

            if submit:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    try:
                        resp = auth_srv.authenticate_user(UserLoginRequest(username=username, password=password))
                        st.session_state.farmer_token = resp.access_token
                        user, profile = auth_srv.get_current_farmer(resp.access_token)
                        st.session_state.current_user = user
                        st.session_state.current_farmer = profile
                        st.success(f"Welcome back, {resp.full_name}!")
                        st.rerun()
                    except AuthenticationError as e:
                        st.error(f"Authentication failed: {e}")

    with auth_tab2:
        st.subheader("Create a New Farmer Account")
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                reg_name = st.text_input("Full Name *", placeholder="e.g. Ramesh Chandra Patel")
                reg_user = st.text_input("Username / Mobile *", placeholder="e.g. 9876543210")
                reg_pwd = st.text_input("Password (min 6 chars) *", type="password")
            with col2:
                reg_phone = st.text_input("Mobile Number (Optional)", placeholder="10-digit number")
                lang_options = list(SUPPORTED_LANGUAGES.keys())
                reg_lang = st.selectbox(
                    "Preferred Language",
                    lang_options,
                    index=0,
                    format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]['native_name']} ({SUPPORTED_LANGUAGES[x]['name']})",
                )

            reg_submit = st.form_submit_button("Create My Account 🌾", use_container_width=True)
            if reg_submit:
                if not reg_name or not reg_user or not reg_pwd:
                    st.error("Please fill in all mandatory fields (*).")
                elif len(reg_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        user, profile, token = auth_srv.register_farmer(UserRegistrationRequest(
                            username=reg_user,
                            password=reg_pwd,
                            full_name=reg_name,
                            phone_number=reg_phone or None,
                            preferred_language=reg_lang,
                        ))
                        st.session_state.farmer_token = token
                        st.session_state.current_user = user
                        st.session_state.current_farmer = profile
                        st.success(f"Account created successfully for {profile.full_name}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Registration error: {e}")


def render_farmer_header(farmer, profile_srv: FarmerProfileService):
    """Render profile completion bar, basic farmer details, and language preference."""
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

    with col1:
        st.markdown(f"### 👨‍🌾 {farmer.full_name}")
        st.caption(f"**Farmer ID:** `{farmer.farmer_id}` • **Mobile:** {farmer.phone_number or 'Not provided'}")

    with col2:
        comp_pct = farmer.profile_completion_pct
        st.markdown(f"**Profile Completion: {comp_pct:.0f}%**")
        st.progress(comp_pct / 100.0)
        badge = "🌱 Getting Started" if comp_pct < 40 else ("🌿 Well Grounded" if comp_pct < 80 else "🌾 Fully Configured")
        st.caption(badge)

    with col3:
        lang_info = SUPPORTED_LANGUAGES.get(farmer.preferred_language, {})
        st.markdown(f"**Language:** {lang_info.get('native_name', farmer.preferred_language)}")
        kcc = "✅ KCC Holder" if farmer.kisan_credit_card_holder else "⚪ No KCC"
        shc = "✅ Soil Card" if farmer.soil_health_card_available else "⚪ No Soil Card"
        st.caption(f"{kcc} • {shc}")

    with col4:
        if st.button("Logout 🚪", use_container_width=True):
            st.session_state.farmer_token = None
            st.session_state.current_farmer = None
            st.session_state.current_user = None
            st.rerun()


def render_farms_and_plots_tab(farmer, farm_srv: FarmService, plot_srv: PlotService):
    """Render multi-farm management and plot creation."""
    farms = farm_srv.list_farms(farmer.farmer_id)

    col_list, col_add = st.columns([3, 2])

    with col_list:
        st.markdown("#### 📍 Registered Farms")
        if not farms:
            st.info("No farms registered yet. Add your first farm using the form on the right.")
        else:
            for f in farms:
                active_badge = "🟢 **ACTIVE CONTEXT**" if f.is_active_farm else "⚪ Standby"
                with st.expander(f"🏡 {f.farm_name} ({f.total_area} {f.area_unit.value.capitalize()} / {f.total_area_acres:.2f} Acres) — {active_badge}", expanded=f.is_active_farm):
                    st.write(f"**Location:** {f.village or ''} {f.tehsil_taluka or ''}, {f.district}, {f.state}")
                    st.write(f"**Coordinates:** {f.latitude or 'Auto-resolved'}, {f.longitude or 'Auto-resolved'}")
                    st.write(f"**Ownership:** {f.ownership_type.value.capitalize()}")
                    if f.notes:
                        st.caption(f"*Notes:* {f.notes}")

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if not f.is_active_farm:
                            if st.button(f"Set as Active Farm 🌟", key=f"act_{f.farm_id}"):
                                farm_srv.set_active_farm(f.farm_id, farmer.farmer_id)
                                st.success(f"'{f.farm_name}' is now your active farm context!")
                                st.rerun()
                    with b_col2:
                        if st.button(f"Delete Farm 🗑️", key=f"del_{f.farm_id}"):
                            farm_srv.delete_farm(f.farm_id, farmer.farmer_id)
                            st.warning(f"Deleted farm '{f.farm_name}'.")
                            st.rerun()

                    # Show plots under this farm
                    plots = plot_srv.list_plots_by_farm(f.farm_id, farmer.farmer_id)
                    st.markdown("---")
                    st.markdown(f"##### 🌿 Plots in {f.farm_name} ({len(plots)})")
                    if plots:
                        for p in plots:
                            st.write(
                                f"• **{p.plot_name}:** {p.area} {p.area_unit.value} ({p.area_acres:.2f} acres) | "
                                f"Soil: *{p.soil_type.value.replace('_', ' ').capitalize()}* | "
                                f"Irrigation: *{p.irrigation_source}*"
                            )
                    else:
                        st.caption("No plots created under this farm yet.")

    with col_add:
        st.markdown("#### ➕ Add New Farm")
        with st.form("add_farm_form"):
            fname = st.text_input("Farm Name *", placeholder="e.g. Riverfront Field / North Farm")
            fstate = st.text_input("State *", placeholder="e.g. Uttar Pradesh, Punjab, Maharashtra")
            fdist = st.text_input("District *", placeholder="e.g. Meerut, Ludhiana, Nashik")
            col_a, col_b = st.columns(2)
            with col_a:
                farea = st.number_input("Total Land Area *", min_value=0.1, value=5.0, step=0.5)
            with col_b:
                funit = st.selectbox("Area Unit", [AreaUnit.ACRE, AreaUnit.HECTARE, AreaUnit.BIGHA, AreaUnit.GUNTHA], format_func=lambda x: x.value.capitalize())
            fowner = st.selectbox("Ownership Type", [OwnershipType.OWNED, OwnershipType.LEASED, OwnershipType.SHARECROPPING], format_func=lambda x: x.value.capitalize())
            fnotes = st.text_area("Notes (Optional)", placeholder="Soil remarks, landmarks, access roads...")
            
            submit_farm = st.form_submit_button("Save Farm 🚜", use_container_width=True)
            if submit_farm:
                if not fname or not fstate or not fdist or farea <= 0:
                    st.error("Please fill in Farm Name, State, District, and Area.")
                else:
                    new_farm = farm_srv.create_farm(farmer.farmer_id, FarmCreate(
                        farm_name=fname,
                        state=fstate,
                        district=fdist,
                        total_area=farea,
                        area_unit=funit,
                        ownership_type=fowner,
                        notes=fnotes or None,
                    ))
                    st.success(f"Added farm '{new_farm.farm_name}'!")
                    st.rerun()

        if farms:
            st.markdown("#### ➕ Add Plot to Farm")
            with st.form("add_plot_form"):
                sel_farm = st.selectbox("Select Target Farm", farms, format_func=lambda x: f"{x.farm_name} ({x.district})")
                pname = st.text_input("Plot / Field Name *", placeholder="e.g. Tubewell Plot / West Ridge")
                parea = st.number_input("Plot Area *", min_value=0.1, value=2.0, step=0.5)
                punit = st.selectbox("Plot Unit", [AreaUnit.ACRE, AreaUnit.HECTARE, AreaUnit.BIGHA], format_func=lambda x: x.value.capitalize())
                psoil = st.selectbox("Soil Type", [SoilType.ALLUVIAL, SoilType.BLACK_COTTON, SoilType.RED_LOAM, SoilType.SANDY_LOAM, SoilType.CLAY_LOAM], format_func=lambda x: x.value.replace("_", " ").capitalize())
                pirr = st.text_input("Irrigation Source", value="Borewell")
                
                submit_plot = st.form_submit_button("Add Plot 🌾", use_container_width=True)
                if submit_plot:
                    if not pname or parea <= 0:
                        st.error("Please specify Plot Name and Area.")
                    else:
                        new_plot = plot_srv.create_plot(sel_farm.farm_id, farmer.farmer_id, PlotCreate(
                            plot_name=pname,
                            area=parea,
                            area_unit=punit,
                            soil_type=psoil,
                            irrigation_source=pirr,
                        ))
                        st.success(f"Added plot '{new_plot.plot_name}' to {sel_farm.farm_name}!")
                        st.rerun()


def render_crop_lifecycle_tab(farmer, farm_srv: FarmService, plot_srv: PlotService, cycle_srv: CropCycleService):
    """Render active crop tracking, growth stage progression, and harvest management."""
    farms = farm_srv.list_farms(farmer.farmer_id)
    if not farms:
        st.info("Please create a farm first in the 'My Farms & Plots' tab.")
        return

    # Select farm
    active_farm = farm_srv.get_active_farm(farmer.farmer_id) or farms[0]
    plots = plot_srv.list_plots_by_farm(active_farm.farm_id, farmer.farmer_id)

    st.markdown(f"### 🌾 Active Crops on **{active_farm.farm_name}**")

    active_cycles = cycle_srv.get_active_cycles_by_farm(active_farm.farm_id, farmer.farmer_id)

    if not active_cycles:
        st.warning(f"No active crop cycle registered for '{active_farm.farm_name}'. Start a new cycle below!")
    else:
        for cycle in active_cycles:
            # Find plot name
            plot_match = next((p for p in plots if p.plot_id == cycle.plot_id), None)
            plot_name = plot_match.plot_name if plot_match else "Plot"

            with st.container(border=True):
                c_header, c_badge = st.columns([3, 1])
                with c_header:
                    st.markdown(f"#### 🌱 {cycle.crop_name} {f'({cycle.variety_name})' if cycle.variety_name else ''} on **{plot_name}**")
                    st.caption(f"**Season:** {cycle.season.value.capitalize()} • **Sown:** {cycle.sowing_date} ({cycle.days_since_sowing} days ago) • **Target Harvest:** {cycle.estimated_harvest_date or 'TBD'}")
                with c_badge:
                    confirmed_tag = "✅ Confirmed by Farmer" if cycle.stage_confirmed_by_farmer else "⏱️ Estimated Stage"
                    st.info(f"**Stage:** {cycle.current_stage.value.replace('_', ' ').capitalize()}\n\n*{confirmed_tag}*")

                # Visual stage progress
                stages_sequence = [
                    CropGrowthStage.SOWING_GERMINATION,
                    CropGrowthStage.SEEDLING_EMERGENCE,
                    CropGrowthStage.VEGETATIVE_TILLERING,
                    CropGrowthStage.FLOWERING_REPRODUCTIVE,
                    CropGrowthStage.GRAIN_FILLING_MATURITY,
                    CropGrowthStage.HARVEST_READY,
                ]
                stage_idx = stages_sequence.index(cycle.current_stage) if cycle.current_stage in stages_sequence else 0
                progress_val = min(1.0, (stage_idx + 1) / len(stages_sequence))
                st.progress(progress_val)

                # Advance or Confirm Stage
                with st.expander("🔄 Advance or Confirm Growth Stage"):
                    with st.form(f"stage_form_{cycle.cycle_id}"):
                        new_stage = st.selectbox(
                            "Select Confirmed Crop Growth Stage",
                            stages_sequence,
                            index=stage_idx,
                            format_func=lambda x: x.value.replace("_", " ").capitalize(),
                        )
                        stage_note = st.text_input("Stage Observation / Note", placeholder="e.g. 50% spikelets emerged, good color")
                        trans_btn = st.form_submit_button("Update Crop Stage ✔️")
                        if trans_btn:
                            cycle_srv.transition_stage(
                                cycle.cycle_id,
                                farmer.farmer_id,
                                CropStageTransitionRequest(
                                    new_stage=new_stage,
                                    notes=stage_note or None,
                                ),
                            )
                            st.success(f"Updated stage to {new_stage.value}!")
                            st.rerun()

                # Harvest Completion Form
                with st.expander("🌾 Record Harvest & Complete Cycle"):
                    with st.form(f"harvest_form_{cycle.cycle_id}"):
                        st.caption("Marking harvest archives the current cycle into rotation history and frees the plot.")
                        hdate = st.date_input("Actual Harvest Date", value=datetime.now().date())
                        hyield = st.number_input("Actual Harvested Yield (Quintals) *", min_value=0.1, value=25.0, step=1.0)
                        hnote = st.text_area("Harvest Notes", placeholder="Grain quality, moisture, storage location...")
                        hbtn = st.form_submit_button("Complete Harvest 🏁")
                        if hbtn:
                            cycle_srv.complete_harvest(
                                cycle.cycle_id,
                                farmer.farmer_id,
                                harvest_date=str(hdate),
                                actual_yield_quintals=hyield,
                                notes=hnote or None,
                            )
                            st.success(f"Harvest recorded for {cycle.crop_name} ({hyield} qtl)! Cycle archived.")
                            st.rerun()

    ui_divider(style="dotted")

    # Start New Crop Cycle Form
    st.markdown("#### ➕ Start a New Crop Cycle")
    if not plots:
        st.info("Please create a plot on this farm first to sow a crop.")
    else:
        with st.form("start_crop_cycle_form"):
            col1, col2 = st.columns(2)
            with col1:
                sel_plot = st.selectbox("Select Plot", plots, format_func=lambda x: f"{x.plot_name} ({x.area} {x.area_unit.value})")
                crop_name = st.text_input("Crop Name *", placeholder="e.g. Wheat, Paddy, Tomato, Mustard, Cotton")
                variety = st.text_input("Crop Variety (Optional)", placeholder="e.g. HD-2967, PBW-343, Pusa Basmati")
            with col2:
                season = st.selectbox("Season", [CropSeason.RABI, CropSeason.KHARIF, CropSeason.ZAID], format_func=lambda x: x.value.capitalize())
                sow_date = st.date_input("Sowing Date *", value=datetime.now().date())
                target_yield = st.number_input("Target Expected Yield (Quintals)", min_value=0.0, value=20.0, step=1.0)

            c_notes = st.text_area("Cycle Notes (Seed source, certified seed tag, etc.)")
            start_btn = st.form_submit_button("Initiate Crop Cycle 🌱", use_container_width=True)

            if start_btn:
                if not crop_name:
                    st.error("Please provide a crop name.")
                else:
                    try:
                        new_cycle = cycle_srv.create_crop_cycle(
                            plot_id=sel_plot.plot_id,
                            farmer_id=farmer.farmer_id,
                            cycle_data=CropCycleCreate(
                                crop_name=crop_name,
                                variety_name=variety or None,
                                season=season,
                                sowing_date=str(sow_date),
                                target_yield_quintals=target_yield if target_yield > 0 else None,
                                notes=c_notes or None,
                            ),
                        )
                        st.success(f"Started crop cycle for {new_cycle.crop_name} on {sel_plot.plot_name}!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

    # Historical Crop Rotation
    completed_cycles = cycle_srv.list_crop_cycles_by_farmer(farmer.farmer_id, status="completed")
    if completed_cycles:
        st.markdown("---")
        st.markdown("#### 📜 Crop Rotation History")
        for hc in completed_cycles:
            st.write(
                f"• **{hc.crop_name}** ({hc.season.value.capitalize()}) — Sown: {hc.sowing_date} | "
                f"Harvested: {hc.actual_harvest_date or 'Yes'} | Yield: **{hc.actual_yield_quintals or 'N/A'} qtl**"
            )


def render_soil_irrigation_tab(farmer, farm_srv: FarmService, plot_srv: PlotService, soil_srv: SoilIrrigationService):
    """Render soil fertility diagnostics and irrigation specifications."""
    farms = farm_srv.list_farms(farmer.farmer_id)
    if not farms:
        st.info("Please create a farm and plots first.")
        return

    active_farm = farm_srv.get_active_farm(farmer.farmer_id) or farms[0]
    plots = plot_srv.list_plots_by_farm(active_farm.farm_id, farmer.farmer_id)
    if not plots:
        st.info(f"No plots found on '{active_farm.farm_name}'. Add a plot to track soil health.")
        return

    sel_plot = st.selectbox("Select Plot for Soil & Irrigation Inspection", plots, format_func=lambda x: f"{x.plot_name} ({x.soil_type.value.capitalize()})")

    soil = soil_srv.get_soil_profile(sel_plot.plot_id, farmer.farmer_id)
    irr = soil_srv.get_irrigation_profile(sel_plot.plot_id, farmer.farmer_id)

    col_soil, col_irr = st.columns(2)

    with col_soil:
        st.markdown("#### 🧪 Soil Health & NPK Profile")
        if soil:
            diag = soil_srv.diagnose_soil_fertility(soil)
            st.write(f"**Soil Type:** {soil.soil_type.value.replace('_', ' ').capitalize()}")
            st.write(f"**Soil pH:** {soil.ph_level or 'Not tested'} ({diag['ph_category']})")
            st.write(f"**Organic Carbon:** {soil.organic_carbon_pct or 'Not tested'}% ({diag['organic_carbon_category']})")
            st.write(f"**Nitrogen (N):** {soil.nitrogen_kg_ha or 'N/A'} kg/ha ({diag['nitrogen_category']})")
            st.write(f"**Phosphorus (P):** {soil.phosphorus_kg_ha or 'N/A'} kg/ha ({diag['phosphorus_category']})")
            st.write(f"**Potassium (K):** {soil.potassium_kg_ha or 'N/A'} kg/ha ({diag['potassium_category']})")
            
            badge_lab = "🔬 Lab Tested / SHC" if soil.is_lab_tested else "📝 Farmer Estimate"
            st.caption(f"Status: **{badge_lab}** {f'(SHC: {soil.soil_health_card_number})' if soil.soil_health_card_number else ''}")

            if diag["recommendations"]:
                st.info("💡 **ICAR Agronomic Recommendations:**\n" + "\n".join(f"• {r}" for r in diag["recommendations"]))
        else:
            st.warning("No soil record added for this plot yet.")

        with st.expander("✏️ Update Soil Record"):
            with st.form(f"soil_form_{sel_plot.plot_id}"):
                s_ph = st.number_input("Soil pH (3.0 - 11.0)", min_value=3.0, max_value=11.0, value=soil.ph_level if soil and soil.ph_level else 7.2, step=0.1)
                s_oc = st.number_input("Organic Carbon % (0.0 - 5.0)", min_value=0.0, max_value=5.0, value=soil.organic_carbon_pct if soil and soil.organic_carbon_pct else 0.55, step=0.05)
                s_n = st.number_input("Available Nitrogen (kg/ha)", min_value=0.0, value=soil.nitrogen_kg_ha if soil and soil.nitrogen_kg_ha else 260.0, step=10.0)
                s_p = st.number_input("Available Phosphorus (kg/ha)", min_value=0.0, value=soil.phosphorus_kg_ha if soil and soil.phosphorus_kg_ha else 18.0, step=2.0)
                s_k = st.number_input("Available Potassium (kg/ha)", min_value=0.0, value=soil.potassium_kg_ha if soil and soil.potassium_kg_ha else 220.0, step=10.0)
                s_lab = st.checkbox("Accredited Lab / Soil Health Card Verified", value=soil.is_lab_tested if soil else False)
                s_shc = st.text_input("Soil Health Card Number (Optional)", value=soil.soil_health_card_number or "" if soil else "")

                save_soil = st.form_submit_button("Save Soil Record 🧪")
                if save_soil:
                    soil_srv.save_soil_profile(
                        sel_plot.plot_id,
                        farmer.farmer_id,
                        SoilProfileCreate(
                            soil_type=sel_plot.soil_type,
                            ph_level=s_ph,
                            organic_carbon_pct=s_oc,
                            nitrogen_kg_ha=s_n,
                            phosphorus_kg_ha=s_p,
                            potassium_kg_ha=s_k,
                            is_lab_tested=s_lab,
                            soil_health_card_number=s_shc or None,
                        ),
                    )
                    st.success("Saved soil profile!")
                    st.rerun()

    with col_irr:
        st.markdown("#### 💧 Irrigation Infrastructure")
        if irr:
            st.write(f"**Primary Source:** {irr.primary_source}")
            st.write(f"**Method:** {irr.irrigation_method.value.capitalize()}")
            st.write(f"**Water Availability:** {irr.water_availability.value.capitalize()}")
            st.write(f"**Pumping Source:** {irr.pumping_source.value.replace('_', ' ').capitalize()}")
        else:
            st.warning("No irrigation profile configured for this plot.")

        with st.expander("✏️ Update Irrigation Profile"):
            with st.form(f"irr_form_{sel_plot.plot_id}"):
                i_src = st.text_input("Primary Water Source", value=irr.primary_source if irr else "Borewell")
                i_meth = st.selectbox(
                    "Irrigation Method",
                    [IrrigationMethod.DRIP, IrrigationMethod.SPRINKLER, IrrigationMethod.FLOOD_FURROW, IrrigationMethod.RAINFED],
                    format_func=lambda x: x.value.capitalize(),
                )
                i_avail = st.selectbox(
                    "Water Availability",
                    [WaterAvailability.ADEQUATE, WaterAvailability.ABUNDANT, WaterAvailability.SCARCE, WaterAvailability.SEASONAL],
                    format_func=lambda x: x.value.capitalize(),
                )
                i_pump = st.selectbox(
                    "Pumping Source",
                    [PumpingSource.ELECTRIC_GRID, PumpingSource.SOLAR_PUMP, PumpingSource.DIESEL_GENERATOR, PumpingSource.GRAVITY_FLOW],
                    format_func=lambda x: x.value.replace("_", " ").capitalize(),
                )
                save_irr = st.form_submit_button("Save Irrigation Profile 💧")
                if save_irr:
                    soil_srv.save_irrigation_profile(
                        sel_plot.plot_id,
                        farmer.farmer_id,
                        IrrigationProfileCreate(
                            primary_source=i_src,
                            irrigation_method=i_meth,
                            water_availability=i_avail,
                            pumping_source=i_pump,
                        ),
                    )
                    st.success("Saved irrigation profile!")
                    st.rerun()


def render_calendar_activities_tab(farmer, farm_srv: FarmService, cycle_srv: CropCycleService, calendar_srv: CropCalendarService):
    """Render farm activity calendar, overdue tasks, and stage-grounded task generation."""
    farms = farm_srv.list_farms(farmer.farmer_id)
    if not farms:
        st.info("Please create a farm first.")
        return

    active_farm = farm_srv.get_active_farm(farmer.farmer_id) or farms[0]
    active_cycles = cycle_srv.get_active_cycles_by_farm(active_farm.farm_id, farmer.farmer_id)

    col_tasks, col_sched = st.columns([3, 2])

    with col_tasks:
        st.markdown(f"### 📋 Tasks for **{active_farm.farm_name}**")

        # Stage recommendation button
        if active_cycles:
            pri_cycle = active_cycles[0]
            if st.button(f"⚡ Generate Grounded Recommendations for {pri_cycle.crop_name} ({pri_cycle.current_stage.value})"):
                created = calendar_srv.generate_grounded_stage_activities(pri_cycle, farmer.farmer_id)
                if created:
                    st.success(f"Generated {len(created)} agronomic recommendations!")
                else:
                    st.info("All recommended tasks for this stage are already on your calendar.")
                st.rerun()

        activities = calendar_srv.list_activities(farmer.farmer_id, farm_id=active_farm.farm_id)

        due_tasks = [a for a in activities if a.status == ActivityStatus.DUE]
        planned_tasks = [a for a in activities if a.status == ActivityStatus.PLANNED]
        completed_tasks = [a for a in activities if a.status == ActivityStatus.COMPLETED]

        if due_tasks:
            st.error(f"⚠️ **{len(due_tasks)} Actionable Due Task(s):**")
            for t in due_tasks:
                with st.expander(f"🔴 DUE: {t.activity_title} (Scheduled: {t.scheduled_date})", expanded=True):
                    st.write(t.description or "No description provided.")
                    if t.stage_tag:
                        st.caption(f"Stage tag: {t.stage_tag}")
                    
                    cost = st.number_input("Cost Incurred (₹)", min_value=0.0, value=0.0, step=50.0, key=f"c_{t.activity_id}")
                    cnote = st.text_input("Completion Note", key=f"n_{t.activity_id}")
                    if st.button("Mark Completed ✓", key=f"cmp_{t.activity_id}"):
                        calendar_srv.complete_activity(
                            t.activity_id,
                            farmer.farmer_id,
                            cost_incurred=cost if cost > 0 else None,
                            notes=cnote or None,
                        )
                        st.success(f"Task '{t.activity_title}' completed!")
                        st.rerun()

        if planned_tasks:
            st.markdown("##### 📅 Upcoming Planned Tasks")
            for t in planned_tasks:
                with st.expander(f"🔵 {t.activity_title} (Due: {t.scheduled_date})"):
                    st.write(t.description or "No description provided.")
                    if st.button("Mark Completed ✓", key=f"cmp_p_{t.activity_id}"):
                        calendar_srv.complete_activity(t.activity_id, farmer.farmer_id)
                        st.success(f"Task '{t.activity_title}' completed!")
                        st.rerun()

        if completed_tasks:
            st.markdown("##### ✅ Completed Tasks")
            for t in completed_tasks[:5]:
                st.write(f"• **{t.activity_title}** — Completed on {t.completed_date} {f'(Cost: ₹{t.cost_incurred:,.2f})' if t.cost_incurred else ''}")

        if not activities:
            st.info("No tasks currently scheduled. Click 'Generate Grounded Recommendations' or schedule a task.")

    with col_sched:
        st.markdown("#### ➕ Schedule Custom Farm Task")
        with st.form("custom_task_form"):
            t_title = st.text_input("Task Title *", placeholder="e.g. Apply Zinc Spray / Clean Drip Filter")
            t_type = st.selectbox("Task Category", [
                ActivityType.IRRIGATION,
                ActivityType.FERTILIZER_APPLICATION,
                ActivityType.PESTICIDE_SPRAY,
                ActivityType.WEEDING,
                ActivityType.LAND_PREPARATION,
                ActivityType.HARVESTING,
                ActivityType.SOIL_TESTING,
                ActivityType.MARKET_SALE,
            ], format_func=lambda x: x.value.replace("_", " ").capitalize())
            t_date = st.date_input("Scheduled Date *", value=datetime.now().date())
            t_desc = st.text_area("Task Description", placeholder="Dosage, instructions, specific plot...")
            
            submit_task = st.form_submit_button("Schedule Task 📅", use_container_width=True)
            if submit_task:
                if not t_title:
                    st.error("Please provide a task title.")
                else:
                    calendar_srv.create_activity(
                        farm_id=active_farm.farm_id,
                        farmer_id=farmer.farmer_id,
                        activity_data=FarmActivityCreate(
                            activity_type=t_type,
                            activity_title=t_title,
                            description=t_desc or None,
                            scheduled_date=str(t_date),
                        ),
                    )
                    st.success(f"Task '{t_title}' scheduled!")
                    st.rerun()

