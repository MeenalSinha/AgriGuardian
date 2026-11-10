"""AgriGuardian — Streamlit Agricultural Market Intelligence & Price Forecasting View."""

import pandas as pd
import streamlit as st

from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import PriceTrendDirection
from backend.app.services.market.market_intelligence import MarketIntelligenceService
from backend.app.services.market.net_realization_calculator import NetRealizationCalculator
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


def render_market_intelligence_view():
    """Render comprehensive agricultural market intelligence view."""
    ui_page_header(
        title="Agricultural Market Intelligence",
        subtitle="Real-Time APMC Mandi Rates, Nearby Spatial Comparisons, Official MSP Benchmarks & Time-Series Forecasting.",
        icon_name="trending-up",
    )
    st.caption("Official Data Grounding: AGMARKNET Daily Bulletins | e-NAM Trade Network | CACP / DA&FW Statutory MSP")

    market_intel = MarketIntelligenceService()

    # ==========================================================================
    # 1. CONTROLS: COMMODITY & LOCATION SELECTION
    # ==========================================================================
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.5, 1.0])

    all_commodities = market_catalog.get_all_commodities()
    comm_options = {
        f"{c.canonical_name} ({c.hindi_name})": c.commodity_id for c in all_commodities
    }

    with col_c1:
        selected_comm_label = st.selectbox(
            "Select Agricultural Commodity",
            list(comm_options.keys()),
            index=0,
        )
        selected_cid = comm_options[selected_comm_label]

    with col_c2:
        all_mandis = market_catalog.get_all_mandis()
        mandi_options = {"All Surrounding Mandis (Location Radius)": None}
        for m in all_mandis:
            mandi_options[f"{m.market_name} ({m.state})"] = m.market_id

        selected_mandi_label = st.selectbox(
            "Primary APMC Mandi",
            list(mandi_options.keys()),
            index=1,
        )
        selected_mid = mandi_options[selected_mandi_label]

    with col_c3:
        search_radius = st.slider(
            "Radius (km)",
            min_value=30,
            max_value=300,
            value=120,
            step=10,
        )

    # Fetch Market Intelligence Summary
    with st.spinner("Retrieving verified APMC market bulletins and price analytics..."):
        summary = market_intel.get_market_summary(
            commodity_query=selected_cid,
            market_id=selected_mid,
            history_days=90,
            forecast_horizon_days=14,
        )

    if not summary.get("is_available"):
        ui_alert(
            summary.get("error", "Market intelligence temporarily unavailable for selected commodity."),
            level="warning",
        )
        return

    latest = summary.get("latest_observation")
    msp_info = summary.get("msp_comparison")
    trend = summary.get("trend_analysis")
    nearby = summary.get("nearby_mandis", [])
    forecast = summary.get("forecast")
    alerts = summary.get("active_alerts", [])

    # ==========================================================================
    # 2. ACTIVE MARKET ALERTS
    # ==========================================================================
    if alerts:
        ui_section_header("Active Market Alerts", "Price anomalies and market notifications.", "bell")
        for alert in alerts:
            severity = alert.get("severity", "WARNING").upper()
            level = "error" if severity == "CRITICAL" else ("warning" if severity == "WARNING" else "info")
            ui_alert(f"**{alert.get('headline')}** — {alert.get('message')}", level=level)

    # ==========================================================================
    # 3. CURRENT PRICE & MSP CARDS
    # ==========================================================================
    ui_section_header("Prevailing APMC Market Rates", "Latest physical terminal transaction rates from agricultural produce markets.", "dollar-sign")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    modal_price = latest["modal_price"]
    min_price = latest["min_price"]
    max_price = latest["max_price"]

    with m_col1:
        st.metric(
            label=f"Modal Rate ({latest['market_name']})",
            value=f"₹{modal_price:,.2f} / qtl",
            delta=f"{trend['percentage_change']:+.1f}% (30d)" if trend and trend.get("percentage_change") else None,
        )

    with m_col2:
        st.metric(
            label="Daily Auction Band",
            value=f"₹{min_price:,.0f} - ₹{max_price:,.0f}",
            delta=f"Spread: ₹{max_price - min_price:,.0f}/qtl",
            delta_color="off",
        )

    with m_col3:
        if msp_info and msp_info.get("has_msp"):
            msp_val = msp_info["msp_price"]
            diff = msp_info["difference_inr"]
            st.metric(
                label=f"Statutory MSP ({msp_info.get('marketing_season', 'RMS/KMS')[:14]})",
                value=f"₹{msp_val:,.2f} / qtl",
                delta=f"₹{diff:+,.2f} vs Market",
            )
        elif msp_info and not msp_info.get("has_msp"):
            st.metric(
                label="Statutory MSP Status",
                value="Non-MSP Mandated",
                delta="Market Determined",
                delta_color="off",
            )
        else:
            st.metric(label="Statutory MSP", value="N/A")

    with m_col4:
        arrivals_val = latest.get("arrivals_tonnes")
        st.metric(
            label="Daily Mandi Arrivals",
            value=f"{arrivals_val:,.1f} Tonnes" if arrivals_val else "Reported Closed",
            delta=f"Bulletin: {latest['reporting_date']}",
            delta_color="off",
        )

    # MSP Transparency Banner
    if msp_info:
        if msp_info.get("has_msp"):
            status_color = "#F0FDF4" if msp_info["status"] == "ABOVE_MSP" else "#FEF2F2"
            border_color = "#16A34A" if msp_info["status"] == "ABOVE_MSP" else "#DC2626"
            st.markdown(
                f"""
                <div style="background-color: {status_color}; border-left: 5px solid {border_color}; border: 1px solid #E2E8F0; border-left-width: 5px; padding: 0.9rem 1.1rem; border-radius: 6px; margin: 0.8rem 0;">
                    <b style="color: {border_color}; font-size: 0.95rem;">Minimum Support Price (MSP) Analysis:</b> {msp_info.get('advisory')}<br>
                    <small style="color: #64748B;"><i>Procurement Agencies: {', '.join(msp_info.get('procurement_agencies', []))} | Quality: {msp_info.get('eligible_variety', 'FAQ')} | <b>Statutory note:</b> {msp_info.get('disclaimer')}</i></small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif msp_info.get("message"):
            ui_alert(f"Commodity Note: {msp_info.get('message')}", level="info")

    ui_divider()

    # ==========================================================================
    # 4. NEARBY MANDI SPATIAL COMPARISON TABLE
    # ==========================================================================
    ui_section_header("Nearby Mandi Price Comparison", f"Comparing markets within {search_radius} km search radius.", "map-pin")
    if nearby:
        nearby_rows = []
        for n in nearby:
            diff_from_selected = n["modal_price"] - modal_price
            nearby_rows.append({
                "Mandi": n["market_name"],
                "District": n["district"],
                "State": n["state"],
                "Distance": f"{n['distance_km']:.1f} km",
                "Modal Rate": f"₹{n['modal_price']:,.2f}",
                "Min - Max Rate": f"₹{n['min_price']:,.0f} - ₹{n['max_price']:,.0f}",
                "Price Difference": f"{diff_from_selected:+,.2f} ₹/qtl" if diff_from_selected != 0 else "Base Mandi",
                "Variety / Grade": n["variety"],
                "Reporting Date": n["reporting_date"],
                "Source": n["source"].split("(")[0].strip(),
            })

        df_nearby = pd.DataFrame(nearby_rows)
        st.dataframe(df_nearby, use_container_width=True, hide_index=True)
        st.caption("Distances represent straight-line geodesic distance calculated from verified coordinates.")
    else:
        st.caption("No additional APMC mandis trading this commodity found within the selected radius.")

    ui_divider()

    # ==========================================================================
    # 5. HISTORICAL PRICE TREND & VOLATILITY ANALYTICS
    # ==========================================================================
    ui_section_header(f"Historical Price Trajectory & Volatility ({latest['market_name']})", "Observed auction price movement and statistical dispersion over recent trading days.", "bar-chart")

    if trend and trend.get("history_points"):
        t_col1, t_col2 = st.columns([2.5, 1.2])

        with t_col1:
            hist_df = pd.DataFrame(trend["history_points"])
            hist_df["Date"] = pd.to_datetime(hist_df["reporting_date"])
            hist_df = hist_df.sort_values("Date")

            chart_data = hist_df.set_index("Date")[["modal_price", "min_price", "max_price"]]
            chart_data.columns = ["Modal Price (₹/qtl)", "Min Price (₹/qtl)", "Max Price (₹/qtl)"]
            st.line_chart(chart_data, color=["#10B981", "#059669", "#D97706"])
            st.caption("Historical wholesale transaction rates (₹/quintal) over recent trading sessions.")

        with t_col2:
            st.markdown("#### Statistical Metrics")
            st.markdown(f"**Trend Direction:** `{trend['trend_direction'].upper()}`")
            st.markdown(f"**Period Average:** ₹{trend['period_average']:,.2f}/qtl")
            st.markdown(f"**Period Median:** ₹{trend['period_median']:,.2f}/qtl")
            st.markdown(f"**Standard Dev (Volatility):** ₹{trend['volatility_std']:,.2f}/qtl")
            st.markdown(f"**Observed Price Range:** ₹{trend['price_range']:,.2f}/qtl")
            st.markdown(f"**Data Completeness:** {trend['data_completeness_pct']:.0f}%")
            ui_alert(f"Trajectory Insight: {trend['trend_explanation']}", level="info")

    ui_divider()

    # ==========================================================================
    # 6. TIME-SERIES FORECASTING & UNCERTAINTY INTERVALS
    # ==========================================================================
    ui_section_header("14-Day Price Forecast & Walk-Forward Evaluation", "Statistical projections modeled with uncertainty bands and evaluated against naive persistence benchmarks.", "trending-up")

    if forecast and forecast.get("forecast_points"):
        fc_col1, fc_col2 = st.columns([2.2, 1.2])

        with fc_col1:
            fc_points = forecast["forecast_points"]
            fc_df = pd.DataFrame(fc_points)
            fc_df["Date"] = pd.to_datetime(fc_df["forecast_date"])

            fc_chart = fc_df.set_index("Date")[["predicted_modal_price", "lower_interval_80", "upper_interval_80"]]
            fc_chart.columns = ["Predicted Rate (₹/qtl)", "Lower Bound (80% CI)", "Upper Bound (80% CI)"]
            st.line_chart(fc_chart, color=["#16532D", "#059669", "#0284C7"])
            st.caption("Forecast horizon with expanding 80% statistical prediction uncertainty interval.")

        with fc_col2:
            st.markdown("#### Model Rigor & Evaluation")
            st.markdown(f"**Model:** {forecast['model_name']}")
            st.markdown(f"**Confidence Tier:** `{forecast['confidence_tier']}`")
            st.markdown(f"**Backtest MAE:** ₹{forecast['backtest_mae']:,.2f}/qtl")
            st.markdown(f"**Backtest RMSE:** ₹{forecast['backtest_rmse']:,.2f}/qtl")
            st.markdown(f"**Naive Baseline MAE:** ₹{forecast['baseline_naive_mae']:,.2f}/qtl")

            if forecast.get("model_outperformed_baseline"):
                ui_alert("Model outperforms naive persistence benchmark in walk-forward evaluation.", level="success")
            else:
                ui_alert("High volatility observed; baseline persistence closely tracking.", level="warning")

        # Forecast Points Table
        with st.expander("View Daily Forecast Points & Confidence Intervals", expanded=False):
            fc_table = []
            for p in fc_points:
                fc_table.append({
                    "Date": p["forecast_date"],
                    "Predicted Modal Rate": f"₹{p['predicted_modal_price']:,.2f}",
                    "80% Interval Band": f"₹{p['lower_interval_80']:,.0f} - ₹{p['upper_interval_80']:,.0f}",
                    "95% Interval Band": f"₹{p['lower_interval_95']:,.0f} - ₹{p['upper_interval_95']:,.0f}",
                })
            st.dataframe(pd.DataFrame(fc_table), use_container_width=True, hide_index=True)

        st.caption(f"Statutory Economic Disclaimer: {forecast.get('statutory_disclaimer')}")
    else:
        ui_alert("Time-series forecasting requires at least 14 sequential trading observations.", level="info")

    ui_divider()

    # ==========================================================================
    # 7. NET REALIZATION CALCULATOR (Farmgate Logistics & Opportunity Analysis)
    # ==========================================================================
    ui_section_header("Farmgate Net Realization Calculator", "Evaluate whether transport and handling costs to a distant market are offset by higher auction prices.", "sliders")

    calc_c1, calc_c2, calc_c3, calc_c4 = st.columns(4)

    with calc_c1:
        produce_qtl = st.number_input(
            "Produce Quantity (Quintals)",
            min_value=1.0,
            max_value=1000.0,
            value=40.0,
            step=5.0,
        )

    with calc_c2:
        mandi_dest_names = [n["market_name"] for n in nearby] if nearby else [latest["market_name"]]
        dest_mandi_choice = st.selectbox("Target Selling Mandi", mandi_dest_names, index=0)

        target_mandi = next((n for n in nearby if n["market_name"] == dest_mandi_choice), None)
        dest_dist = target_mandi["distance_km"] if target_mandi else 0.0
        dest_rate = target_mandi["modal_price"] if target_mandi else modal_price

    with calc_c3:
        transport_rate_km = st.number_input(
            "Transport Rate (₹/qtl/km)",
            min_value=0.5,
            max_value=10.0,
            value=2.50,
            step=0.25,
            help="Indicative small commercial pickup/truck hiring rate per quintal per km.",
        )

    with calc_c4:
        labor_rate_qtl = st.number_input(
            "Handling / Hamali (₹/qtl)",
            min_value=5.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            help="Labor loading, unloading, weighing, and bagging charges at mandi yard.",
        )

    calc_res = NetRealizationCalculator.calculate(
        market_id=target_mandi["market_id"] if target_mandi else latest["market_id"],
        market_name=dest_mandi_choice,
        commodity_name=selected_comm_label.split("(")[0].strip(),
        distance_km=dest_dist,
        quantity_quintals=produce_qtl,
        reference_modal_price=dest_rate,
        transport_rate_per_km_qtl=transport_rate_km,
    )

    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    with r_c1:
        st.metric("Gross Auction Value", f"₹{calc_res.gross_value_inr:,.2f}")
    with r_c2:
        st.metric("Estimated Transport", f"₹{calc_res.transport_cost_inr:,.2f}", delta=f"{dest_dist:.1f} km", delta_color="inverse")
    with r_c3:
        st.metric("Handling & Cess", f"₹{calc_res.handling_fees_inr:,.2f}", delta_color="inverse")
    with r_c4:
        st.metric(
            "Estimated Net Realization",
            f"₹{calc_res.estimated_net_realization_inr:,.2f}",
            delta=f"₹{calc_res.net_realization_per_quintal:,.2f} / qtl",
        )

    with st.expander("View Financial Assumptions & Quality Caveat", expanded=False):
        for assumption in calc_res.assumptions:
            st.markdown(f"- {assumption}")
        st.caption(f"Disclaimer: {calc_res.disclaimer}")
