"""Streamlit View for Phase 11 Farm Economics & Profitability Intelligence.

Comprehensive agricultural cost tracking, labor allocation, multi-event harvesting,
actual produce sales, break-even intelligence, budget variance, and scenario simulation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
import streamlit as st
import pandas as pd

from backend.app.core.config import get_settings
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.economics import (
    BudgetItemCreate,
    CropComparisonItem,
    CropComparisonRequest,
    CropCycleBudgetCreate,
    CropCycleBudgetUpdate,
    ExpenseCategory,
    ExpenseCorrectionCreate,
    FarmExpenseCreate,
    FarmHarvestCreate,
    FarmSaleCreate,
    PaymentStatus,
    ScenarioSimulateRequest,
    ScenarioType,
)
from backend.app.services.economics import EconomicsService, get_economics_service
from backend.app.services.economics.unit_conversion_service import UnitConversionService
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


def get_economics_bundle():
    """Retrieve singleton economics service and farm database repository."""
    repo = get_farm_repo()
    service = get_economics_service(repo)
    return repo, service


def render_farm_economics_view():
    """Render the comprehensive Farm Economics & Profitability Intelligence dashboard."""
    repo, economics = get_economics_bundle()
    settings = get_settings()

    ui_page_header(
        title="Farm Economics & Profitability Intelligence",
        subtitle="Decimal-exact cost accounting, labor allocation, multi-event harvest tracking, actual sales realization, and break-even analytics.",
        icon_name="dollar-sign",
    )

    # -------------------------------------------------------------------------
    # Farmer Selector & Authentication Context
    # -------------------------------------------------------------------------
    current_farmer = st.session_state.get("current_farmer")
    current_user = st.session_state.get("current_user")

    # If no farmer in session, load profiles from DB for seamless selection
    with repo.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT farmer_id, full_name, phone_number 
            FROM farmer_profiles 
            ORDER BY 
                CASE 
                    WHEN full_name LIKE '%Ramesh%' THEN 0 
                    WHEN full_name LIKE '%Attacker%' OR full_name LIKE '%Test%' THEN 2 
                    ELSE 1 
                END, 
                created_at DESC 
            LIMIT 50
        """)
        farmer_rows = cursor.fetchall()

    if not farmer_rows:
        ui_alert("No registered farmer profiles found. Please register a farmer in 'My Farms & Crops' first.", level="warning")
        return

    farmer_options = {f"{r['full_name']} ({r['phone_number']} - ID: {r['farmer_id'][:8]})": r['farmer_id'] for r in farmer_rows}

    # Select active farmer
    default_idx = 0
    if current_farmer and current_farmer.farmer_id in farmer_options.values():
        default_idx = list(farmer_options.values()).index(current_farmer.farmer_id)

    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        selected_label = st.selectbox("Select Active Farmer Account:", list(farmer_options.keys()), index=default_idx)
        active_farmer_id = farmer_options[selected_label]
    with col_f2:
        st.caption("Active Session Context")
        st.code(f"ID: {active_farmer_id}", language="text")

    user_id = current_user.user_id if current_user else f"user_{active_farmer_id}"

    # Load Farmer's Farms & Active Cycles
    farms = repo.get_farms_by_farmer(active_farmer_id)
    farm_map = {f.farm_id: f for f in farms}

    all_cycles = repo.get_all_crop_cycles_by_farmer(active_farmer_id)
    cycle_map = {c.cycle_id: c for c in all_cycles}

    ui_divider(style="dotted")

    # -------------------------------------------------------------------------
    # Main Economics Navigation Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Financial Overview",
        "Expenses & Labor",
        "Harvest & Production",
        "Sales & Realized Revenue",
        "Profitability & Budgets",
        "Scenarios & Crop Planning",
    ])

    # =========================================================================
    # TAB 1: FINANCIAL OVERVIEW
    # =========================================================================
    with tab1:
        ui_section_header("Farm Financial Summary & Cash Flow", "High-level overview of actual cultivation costs, payments, realized revenues, and cycle margins.", "dollar-sign")

        overview = economics.profitability.get_farmer_financial_overview(active_farmer_id)

        # KPI Metrics Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            ui_metric_card(
                "Total Incurred Cost",
                f"₹{overview.total_cultivation_expenses:,.2f}",
                delta=f"{overview.provisional_cycles_count} Active Cycles",
            )
        with kpi2:
            ui_metric_card(
                "Paid Expenses",
                f"₹{overview.total_paid_expenses:,.2f}",
                delta="Settled",
            )
        with kpi3:
            ui_metric_card(
                "Outstanding Due",
                f"₹{overview.total_outstanding_expenses:,.2f}",
                delta="Payables Pending",
                delta_color="off",
            )
        with kpi4:
            ui_metric_card(
                "Realized Net Revenue",
                f"₹{overview.total_realized_revenue:,.2f}",
                delta="After Mandi Deductions",
            )
        with kpi5:
            delta_type = "normal" if overview.net_balance >= 0 else "inverse"
            ui_metric_card(
                "Net Realized Balance",
                f"₹{overview.net_balance:,.2f}",
                delta="Net Realized Profit" if overview.net_balance >= 0 else "Operating Deficit",
                delta_color=delta_type,
            )

        if overview.provisional_cycles_count > 0:
            ui_alert(
                f"Provisional Status: You have {overview.provisional_cycles_count} active crop cycles currently in progress. "
                "Final profit/loss metrics for those cycles will be confirmed once all harvesting and produce sales are concluded.",
                level="info",
            )

        ui_divider(style="dotted")

        # Two columns: Expenses Breakdown & Recent Transactions
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("#### Expenditure Distribution by Category")
            all_expenses = economics.expenses.list_expenses(farmer_id=active_farmer_id, is_planned=False)
            if all_expenses:
                cat_totals: Dict[str, float] = {}
                for exp in all_expenses:
                    cat_totals[exp.expense_category.value] = cat_totals.get(exp.expense_category.value, 0.0) + exp.total_amount
                df_cat = pd.DataFrame([{"Category": k.replace("_", " ").title(), "Amount (₹)": v} for k, v in cat_totals.items()])
                st.dataframe(df_cat, use_container_width=True, hide_index=True)
            else:
                st.caption("No incurred expenses recorded yet. Use the 'Expenses & Labor' tab to add your first expense.")

        with col_c2:
            st.markdown("#### Recent Financial Transactions")
            if overview.recent_expenses:
                st.markdown("**Recent Expenses:**")
                for e in overview.recent_expenses[:3]:
                    status_text = "Paid" if e.payment_status == PaymentStatus.PAID else "Unpaid"
                    st.write(f"- **₹{e.total_amount:,.2f}** ({e.expense_description}) — {e.purchase_date} [{status_text}]")
            if overview.recent_sales:
                st.markdown("**Recent Sales:**")
                for s in overview.recent_sales[:3]:
                    st.write(f"- **₹{s.net_amount_received:,.2f}** ({s.crop_name}, {s.quantity_sold} {s.quantity_unit}) — {s.sale_date}")
            if not overview.recent_expenses and not overview.recent_sales:
                st.caption("No recent transactions recorded.")

    # =========================================================================
    # TAB 2: EXPENSES & LABOR
    # =========================================================================
    with tab2:
        ui_section_header("Record & Track Agricultural Expenses", "Decimal-safe tracking of seeds, fertilizers, pesticides, labor, equipment rent, fuel, and utilities.", "bar-chart")

        with st.expander("Record New Agricultural Expense", expanded=False):
            with st.form("add_expense_form"):
                fe1, fe2, fe3 = st.columns(3)
                with fe1:
                    exp_farm_opts = {f.farm_name: f.farm_id for f in farms}
                    sel_farm_name = st.selectbox("Farm *", list(exp_farm_opts.keys()))
                    sel_farm_id = exp_farm_opts[sel_farm_name]
                with fe2:
                    farm_plots = repo.get_plots_by_farm(sel_farm_id, active_farmer_id)
                    plot_opts = {"Entire Farm / General": None}
                    plot_opts.update({f"{p.plot_name} ({p.area_acres} ac)": p.plot_id for p in farm_plots})
                    sel_plot_name = st.selectbox("Plot (Optional)", list(plot_opts.keys()))
                    sel_plot_id = plot_opts[sel_plot_name]
                with fe3:
                    cycle_opts = {"General / No Cycle": None}
                    farm_cycles = [c for c in all_cycles if c.farm_id == sel_farm_id]
                    cycle_opts.update({f"{c.crop_name} - {c.variety_name or 'Standard'} ({c.status.value})": c.cycle_id for c in farm_cycles})
                    sel_cycle_name = st.selectbox("Crop Cycle (Optional)", list(cycle_opts.keys()))
                    sel_cycle_id = cycle_opts[sel_cycle_name]

                fe4, fe5, fe6 = st.columns(3)
                with fe4:
                    cat_list = [c for c in ExpenseCategory]
                    exp_cat = st.selectbox("Expense Category *", cat_list, format_func=lambda x: x.value.replace("_", " ").title())
                with fe5:
                    exp_subcat = st.text_input("Subcategory", placeholder="e.g. Sowing Labor, Urea, Diesel")
                with fe6:
                    exp_desc = st.text_input("Description / Notes *", placeholder="e.g. 5 laborers for transplanting")

                fe7, fe8, fe9, fe10 = st.columns(4)
                with fe7:
                    exp_qty = st.number_input("Quantity", min_value=0.0, step=0.5, value=0.0)
                with fe8:
                    exp_unit = st.selectbox("Unit", ["hours", "days", "kg", "bags", "liters", "units", "acres", "quintals"])
                with fe9:
                    exp_price = st.number_input("Unit Price (₹)", min_value=0.0, step=10.0, value=0.0)
                with fe10:
                    calc_default = float(exp_qty * exp_price) if (exp_qty > 0 and exp_price > 0) else 0.0
                    exp_total = st.number_input("Total Amount (₹) *", min_value=0.01, step=50.0, value=calc_default if calc_default > 0 else 100.0)

                fe11, fe12, fe13, fe14 = st.columns(4)
                with fe11:
                    exp_date = st.date_input("Purchase / Incurred Date *", value=datetime.now())
                with fe12:
                    exp_status = st.selectbox("Payment Status", [PaymentStatus.PAID, PaymentStatus.UNPAID, PaymentStatus.PARTIAL])
                with fe13:
                    exp_planned = st.checkbox("Planned Expense (Budget only)")
                with fe14:
                    exp_vendor = st.text_input("Supplier / Vendor", placeholder="Kisan Seva Kendra")

                submit_exp = st.form_submit_button("Save Expense Entry", use_container_width=True)
                if submit_exp:
                    try:
                        final_total = exp_total
                        if exp_qty > 0 and exp_price > 0 and abs(exp_total - (exp_qty * exp_price)) > 0.05:
                            final_total = float(UnitConversionService.round_currency(Decimal(str(exp_qty)) * Decimal(str(exp_price))))

                        payload = FarmExpenseCreate(
                            farm_id=sel_farm_id,
                            plot_id=sel_plot_id,
                            crop_cycle_id=sel_cycle_id,
                            expense_category=exp_cat,
                            subcategory=exp_subcat or None,
                            expense_description=exp_desc or f"{exp_cat.value.title()} expense",
                            quantity=exp_qty if exp_qty > 0 else None,
                            quantity_unit=exp_unit if exp_qty > 0 else None,
                            unit_price=exp_price if exp_price > 0 else None,
                            total_amount=final_total,
                            currency="INR",
                            purchase_date=exp_date.strftime("%Y-%m-%d"),
                            payment_status=exp_status,
                            is_planned=exp_planned,
                            supplier_vendor=exp_vendor or None,
                        )
                        saved = economics.expenses.record_expense(farmer_id=active_farmer_id, payload=payload, user_id=user_id)
                        ui_alert(f"Recorded expense: {saved.expense_description} (₹{saved.total_amount:,.2f})", level="success")
                        st.rerun()
                    except Exception as e:
                        ui_alert(f"Failed to record expense: {e}", level="error")

        # List Existing Expenses
        st.markdown("#### Incurred Expenses Register")
        existing_expenses = economics.expenses.list_expenses(farmer_id=active_farmer_id)
        if existing_expenses:
            exp_table_data = []
            for e in existing_expenses:
                exp_table_data.append({
                    "ID": e.expense_id,
                    "Date": e.purchase_date,
                    "Category": e.expense_category.value.replace("_", " ").title(),
                    "Description": e.expense_description,
                    "Qty": f"{e.quantity} {e.quantity_unit}" if e.quantity else "-",
                    "Total (₹)": f"₹{e.total_amount:,.2f}",
                    "Status": e.payment_status.value.upper(),
                    "Planned": "Yes" if e.is_planned else "No",
                    "Vendor": e.supplier_vendor or "-",
                })
            df_exp = pd.DataFrame(exp_table_data)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)

            # Append-Only Expense Correction Tool
            with st.expander("Apply Append-Only Financial Correction (Audit Trail)"):
                st.caption("Apply an auditable adjustment to an existing expense without destroying historical records.")
                cor_c1, cor_c2, cor_c3 = st.columns([2, 1, 1])
                with cor_c1:
                    exp_choices = {f"{e.expense_description} (₹{e.total_amount:,.2f} on {e.purchase_date}) [{e.expense_id}]": e.expense_id for e in existing_expenses}
                    selected_exp_label = st.selectbox("Select Expense to Correct:", list(exp_choices.keys()))
                    target_exp_id = exp_choices[selected_exp_label]
                with cor_c2:
                    new_corrected_amt = st.number_input("New Corrected Amount (₹)", min_value=0.01, step=10.0, value=100.0)
                with cor_c3:
                    cor_reason = st.text_input("Audit Reason *", placeholder="e.g. Overcharged on invoice")

                if st.button("Apply Audited Correction"):
                    if not cor_reason:
                        ui_alert("Please provide an audit reason for the correction.", level="error")
                    else:
                        try:
                            corr_payload = ExpenseCorrectionCreate(
                                reason=cor_reason,
                                new_amount=new_corrected_amt,
                            )
                            c_rec = economics.expenses.correct_expense(
                                expense_id=target_exp_id,
                                farmer_id=active_farmer_id,
                                payload=corr_payload,
                                user_id=user_id,
                            )
                            ui_alert(f"Correction recorded! Previous: ₹{c_rec.previous_amount:,.2f} → New: ₹{c_rec.new_amount:,.2f}", level="success")
                            st.rerun()
                        except Exception as e:
                            ui_alert(f"Correction failed: {e}", level="error")
        else:
            ui_alert("No expenses registered yet.", level="info")

    # =========================================================================
    # TAB 3: HARVEST & PRODUCTION
    # =========================================================================
    with tab3:
        ui_section_header("Crop Harvest & Production Tracking", "Record harvest events, track gross yield per acre, and reconcile storage, sales, and retained seed.", "sprout")

        with st.expander("Record Crop Harvest Event", expanded=False):
            with st.form("record_harvest_form"):
                h1, h2 = st.columns(2)
                with h1:
                    h_cycle_opts = {f"{c.crop_name} - {c.variety_name or 'Std'} ({c.cycle_id})": c for c in all_cycles}
                    if not h_cycle_opts:
                        ui_alert("No crop cycles found. Please create a crop cycle in 'My Farms & Crops' first.", level="warning")
                        sel_h_cycle = None
                    else:
                        sel_h_label = st.selectbox("Select Crop Cycle *", list(h_cycle_opts.keys()))
                        sel_h_cycle = h_cycle_opts[sel_h_label]
                with h2:
                    h_date = st.date_input("Harvest Date *", value=datetime.now())

                h3, h4, h5 = st.columns(3)
                with h3:
                    h_gross_qty = st.number_input("Gross Harvested Quantity *", min_value=0.1, step=1.0, value=20.0)
                with h4:
                    h_unit = st.selectbox("Quantity Unit", ["quintal", "kg", "ton", "bag"])
                with h5:
                    h_grade = st.selectbox("Quality Grade", ["FAQ (Fair Average Quality)", "Grade A (Premium)", "Grade B (Standard)", "Substandard"])

                st.markdown("##### Post-Harvest Produce Allocation (Reconciliation)")
                ha1, ha2, ha3, ha4, ha5 = st.columns(5)
                with ha1:
                    h_sold = st.number_input("Sold Instantly", min_value=0.0, step=1.0, value=0.0)
                with ha2:
                    h_stored = st.number_input("Stored in Warehouse", min_value=0.0, step=1.0, value=float(h_gross_qty))
                with ha3:
                    h_retained = st.number_input("Retained (Seed/Self)", min_value=0.0, step=0.5, value=0.0)
                with ha4:
                    h_damaged = st.number_input("Damaged", min_value=0.0, step=0.5, value=0.0)
                with ha5:
                    h_rejected = st.number_input("Rejected", min_value=0.0, step=0.5, value=0.0)

                h_notes = st.text_input("Harvest Notes", placeholder="First picking, excellent moisture levels")

                submit_h = st.form_submit_button("Save Harvest Record", use_container_width=True)
                if submit_h and sel_h_cycle:
                    try:
                        h_payload = FarmHarvestCreate(
                            farm_id=sel_h_cycle.farm_id,
                            plot_id=sel_h_cycle.plot_id,
                            crop_cycle_id=sel_h_cycle.cycle_id,
                            harvest_date=h_date.strftime("%Y-%m-%d"),
                            crop_name=sel_h_cycle.crop_name,
                            variety_name=sel_h_cycle.variety_name,
                            harvested_quantity=h_gross_qty,
                            quantity_unit=h_unit,
                            quality_grade=h_grade,
                            damaged_quantity=h_damaged,
                            rejected_quantity=h_rejected,
                            stored_quantity=h_stored,
                            sold_quantity=h_sold,
                            retained_quantity=h_retained,
                            notes=h_notes or None,
                        )
                        h_saved = economics.harvests.record_harvest(farmer_id=active_farmer_id, payload=h_payload, user_id=user_id)
                        ui_alert(f"Harvest recorded: {h_saved.harvested_quantity} {h_saved.quantity_unit} of {h_saved.crop_name}", level="success")
                        st.rerun()
                    except Exception as e:
                        ui_alert(f"Failed to record harvest: {e}", level="error")

        # Harvest Log
        st.markdown("#### Harvest Records Register")
        existing_harvests = economics.harvests.list_harvests(farmer_id=active_farmer_id)
        if existing_harvests:
            h_data = []
            for h in existing_harvests:
                h_data.append({
                    "ID": h.harvest_id,
                    "Date": h.harvest_date,
                    "Crop": h.crop_name,
                    "Variety": h.variety_name or "-",
                    "Harvested": f"{h.harvested_quantity} {h.quantity_unit}",
                    "Grade": h.quality_grade or "FAQ",
                    "Stored": f"{h.stored_quantity} {h.quantity_unit}",
                    "Sold": f"{h.sold_quantity} {h.quantity_unit}",
                    "Retained": f"{h.retained_quantity} {h.quantity_unit}",
                    "Damaged": f"{h.damaged_quantity} {h.quantity_unit}",
                })
            df_h = pd.DataFrame(h_data)
            st.dataframe(df_h, use_container_width=True, hide_index=True)
        else:
            ui_alert("No harvest events recorded yet.", level="info")

    # =========================================================================
    # TAB 4: SALES & REALIZED REVENUE
    # =========================================================================
    with tab4:
        ui_section_header("Produce Sales & Realized Revenue", "Record actual crop sales transactions, realized prices, mandi commissions, and net receipts.", "dollar-sign")

        with st.expander("Record Actual Produce Sale", expanded=False):
            with st.form("record_sale_form"):
                s1, s2 = st.columns(2)
                with s1:
                    s_cycle_opts = {f"{c.crop_name} ({c.cycle_id})": c for c in all_cycles}
                    if not s_cycle_opts:
                        ui_alert("No crop cycles found.", level="warning")
                        sel_s_cycle = None
                    else:
                        sel_s_label = st.selectbox("Crop Cycle *", list(s_cycle_opts.keys()))
                        sel_s_cycle = s_cycle_opts[sel_s_label]
                with s2:
                    s_date = st.date_input("Sale Date *", value=datetime.now())

                s3, s4, s5 = st.columns(3)
                with s3:
                    s_qty = st.number_input("Quantity Sold *", min_value=0.1, step=1.0, value=15.0)
                with s4:
                    s_unit = st.selectbox("Unit", ["quintal", "kg", "bag", "ton"], key="sale_unit_select")
                with s5:
                    s_price = st.number_input("Realized Price per Unit (₹) *", min_value=1.0, step=50.0, value=2275.0)

                st.markdown("##### Deductions, Transport & Commission")
                sd1, sd2, sd3 = st.columns(3)
                with sd1:
                    s_comm = st.number_input("Commission Charges (Mandi Fee) (₹)", min_value=0.0, step=20.0, value=0.0)
                with sd2:
                    s_trans = st.number_input("Transport Charges (₹)", min_value=0.0, step=50.0, value=0.0)
                with sd3:
                    s_other_ded = st.number_input("Loading / Unloading / Cess (₹)", min_value=0.0, step=10.0, value=0.0)

                s6, s7 = st.columns(2)
                with s6:
                    s_buyer = st.text_input("Buyer / Mandi Trader Name", placeholder="e.g. Ramesh Trading Co, APMC Azadpur")
                with s7:
                    s_status = st.selectbox("Payment Realization Status", [PaymentStatus.PAID, PaymentStatus.PARTIAL, PaymentStatus.UNPAID])

                submit_s = st.form_submit_button("Save Produce Sale", use_container_width=True)
                if submit_s and sel_s_cycle:
                    try:
                        gross_calc = float(UnitConversionService.round_currency(Decimal(str(s_qty)) * Decimal(str(s_price))))
                        net_calc = max(0.0, gross_calc - (s_comm + s_trans + s_other_ded))
                        s_payload = FarmSaleCreate(
                            farm_id=sel_s_cycle.farm_id,
                            plot_id=sel_s_cycle.plot_id,
                            crop_cycle_id=sel_s_cycle.cycle_id,
                            crop_name=sel_s_cycle.crop_name,
                            quantity_sold=s_qty,
                            quantity_unit=s_unit,
                            realized_price_per_unit=s_price,
                            gross_amount=gross_calc,
                            deductions_charges=s_other_ded,
                            transport_charges=s_trans,
                            commission_charges=s_comm,
                            net_amount_received=net_calc,
                            sale_date=s_date.strftime("%Y-%m-%d"),
                            buyer_name=s_buyer or None,
                            payment_status=s_status,
                        )
                        s_saved = economics.sales.record_sale(farmer_id=active_farmer_id, payload=s_payload, user_id=user_id)
                        ui_alert(f"Sale recorded: {s_saved.quantity_sold} {s_saved.quantity_unit} of {s_saved.crop_name} (Gross: ₹{s_saved.gross_amount:,.2f}, Net: ₹{s_saved.net_amount_received:,.2f})", level="success")
                        st.rerun()
                    except Exception as e:
                        ui_alert(f"Failed to record sale: {e}", level="error")

        # Sales Register
        st.markdown("#### Produce Sales Register")
        existing_sales = economics.sales.list_sales(farmer_id=active_farmer_id)
        if existing_sales:
            s_data = []
            for s in existing_sales:
                s_data.append({
                    "ID": s.sale_id,
                    "Date": s.sale_date,
                    "Crop": s.crop_name,
                    "Qty Sold": f"{s.quantity_sold} {s.quantity_unit}",
                    "Realized Price": f"₹{s.realized_price_per_unit:,.2f}",
                    "Gross (₹)": f"₹{s.gross_amount:,.2f}",
                    "Deductions (₹)": f"₹{(s.deductions_charges + s.transport_charges + s.commission_charges):,.2f}",
                    "Net Received (₹)": f"₹{s.net_amount_received:,.2f}",
                    "Buyer": s.buyer_name or "-",
                    "Status": s.payment_status.value.upper(),
                })
            df_s = pd.DataFrame(s_data)
            st.dataframe(df_s, use_container_width=True, hide_index=True)
        else:
            ui_alert("No produce sales recorded yet.", level="info")

    # =========================================================================
    # TAB 5: PROFITABILITY & BUDGETS
    # =========================================================================
    with tab5:
        ui_section_header("Crop Cycle Profitability & Budget Intelligence", "Deep financial breakdown, break-even price/yield analysis, and planned vs actual budget variances.", "bar-chart")

        if not all_cycles:
            ui_alert("No crop cycles available for profitability evaluation.", level="warning")
        else:
            prof_cycle_opts = {f"{c.crop_name} ({c.variety_name or 'Std'}) - {c.cycle_id}": c.cycle_id for c in all_cycles}
            sel_prof_label = st.selectbox("Select Crop Cycle to Evaluate:", list(prof_cycle_opts.keys()))
            target_cycle_id = prof_cycle_opts[sel_prof_label]

            # 1. Profitability Metrics
            prof = economics.profitability.calculate_crop_cycle_profitability(target_cycle_id, active_farmer_id)

            if prof.is_provisional:
                ui_alert(
                    f"PROVISIONAL FINANCIAL METRIC: This crop cycle has missing data or is still ongoing ({', '.join(prof.missing_data_fields)}). "
                    "All figures represent actual costs/revenues to date and are subject to revision upon completion.",
                    level="warning",
                )

            # Metric Cards
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            with p_c1:
                ui_metric_card("Total Actual Cost", f"₹{prof.total_actual_cost:,.2f}", delta=f"{prof.cultivated_area_acres:.1f} Acres Cultivated")
            with p_c2:
                ui_metric_card("Net Realized Revenue", f"₹{prof.total_net_revenue:,.2f}", delta=f"{prof.total_sold_quintals:.1f} Qtl Sold")
            with p_c3:
                net_label = "Net Profit" if prof.is_profitable else "Net Deficit"
                p_delta = "Profitable Cycle" if prof.is_profitable else "Negative Margin"
                ui_metric_card(net_label, f"₹{prof.net_profit_or_loss:,.2f}", delta=p_delta, delta_color="normal" if prof.is_profitable else "inverse")
            with p_c4:
                roi_text = f"{prof.return_on_cultivation_cost_pct:.1f}%" if prof.return_on_cultivation_cost_pct is not None else "N/A"
                ui_metric_card("Return on Cost (ROI)", roi_text, delta="Based on Incurred Costs")

            ui_divider(style="dotted")

            # Break-Even & Unit Metrics
            st.markdown("#### Break-Even & Normalized Economics")
            be1, be2, be3, be4 = st.columns(4)
            with be1:
                c_acre = f"₹{prof.cost_per_acre:,.2f}" if prof.cost_per_acre else "N/A"
                st.metric("Cost per Acre", c_acre)
            with be2:
                c_qtl = f"₹{prof.cost_per_quintal:,.2f}" if prof.cost_per_quintal else "N/A"
                st.metric("Cost per Quintal", c_qtl)
            with be3:
                be_price = f"₹{prof.break_even_selling_price_per_quintal:,.2f}" if prof.break_even_selling_price_per_quintal else "N/A"
                st.metric("Break-Even Price", be_price, help="Minimum selling price needed to recover total cultivation expenses.")
            with be4:
                be_yield = f"{prof.break_even_production_quintals:.2f} Qtl" if prof.break_even_production_quintals else "N/A"
                st.metric("Break-Even Yield", be_yield, help="Minimum production needed at realized price to avoid financial loss.")

            # Bilingual Explanation
            with st.expander("Financial Methodology & Transparency Notes (English & Hindi)"):
                st.markdown(f"**English:**\n{prof.explanation_en}")
                st.markdown(f"**Hindi:**\n{prof.explanation_hi}")

            ui_divider(style="dotted")

            # 2. Budget vs Actual Variance
            st.markdown("#### Crop Cycle Budget vs Actual Expenditure")
            try:
                variance_rep = economics.budgets.calculate_budget_variance(target_cycle_id, active_farmer_id)
            except Exception:
                variance_rep = None

            if variance_rep:
                if variance_rep.is_exceeded:
                    ui_alert(
                        f"BUDGET EXCEEDED: Total expenses (₹{variance_rep.total_actual:,.2f}) exceed planned budget "
                        f"(₹{variance_rep.total_planned:,.2f}) by ₹{variance_rep.amount_over_budget:,.2f} (+{variance_rep.total_variance_pct:.1f}%).",
                        level="error",
                    )

                # Variance Table
                var_data = []
                for it in variance_rep.categories:
                    var_data.append({
                        "Category": it.expense_category.value.replace("_", " ").title(),
                        "Planned (₹)": f"₹{it.planned_amount:,.2f}",
                        "Actual (₹)": f"₹{it.actual_amount:,.2f}",
                        "Variance (₹)": f"{'+' if it.variance_amount > 0 else ''}₹{it.variance_amount:,.2f}",
                        "Variance (%)": f"{'+' if it.variance_percentage > 0 else ''}{it.variance_percentage:.1f}%",
                        "Alert": "Over Budget" if it.is_over_budget else "Within Budget",
                    })
                st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
            else:
                ui_alert("No planned budget registered for this crop cycle yet. You can set a seasonal budget below.", level="info")

            # Create / Edit Budget
            with st.expander("Set or Update Seasonal Budget for this Crop Cycle"):
                with st.form("set_budget_form"):
                    b_name = st.text_input("Budget Name", value=f"Budget for {prof.crop_name} Cycle")
                    st.caption("Allocate planned budget amounts across categories:")
                    b_items = []
                    b_cols = st.columns(4)
                    cat_items = list(ExpenseCategory)
                    for idx, cat in enumerate(cat_items):
                        with b_cols[idx % 4]:
                            amt = st.number_input(f"{cat.value.replace('_', ' ').title()} (₹)", min_value=0.0, step=500.0, value=5000.0 if idx < 4 else 1000.0)
                            b_items.append(BudgetItemCreate(expense_category=cat, planned_amount=amt))

                    submit_budget = st.form_submit_button("Save Cycle Budget")
                    if submit_budget:
                        try:
                            tot_planned = sum(item.planned_amount for item in b_items)
                            b_create = CropCycleBudgetCreate(
                                farm_id=prof.farm_id,
                                crop_cycle_id=target_cycle_id,
                                budget_name=b_name,
                                total_budget_amount=tot_planned,
                                items=b_items,
                            )
                            saved_b = economics.budgets.create_budget(farmer_id=active_farmer_id, payload=b_create, user_id=user_id)
                            ui_alert(f"Budget saved: {saved_b.budget_name} (Total Planned: ₹{saved_b.total_budget_amount:,.2f})", level="success")
                            st.rerun()
                        except Exception as e:
                            ui_alert(f"Failed to save budget: {e}", level="error")

    # =========================================================================
    # TAB 6: SCENARIOS & CROP PLANNING
    # =========================================================================
    with tab6:
        ui_section_header("Scenario Simulator & Multi-Crop Economic Planning", "Simulate Conservative, Base, and Optimistic crop scenarios, evaluate sensitivity, and compare prospective crops.", "trending-up")

        sub_sim, sub_comp = st.tabs(["Scenario Simulator", "Prospective Crop Comparison"])

        # SUB-TAB 1: SIMULATOR
        with sub_sim:
            st.markdown("#### Deterministic Farm Economics Simulator")
            sim1, sim2, sim3 = st.columns(3)
            with sim1:
                sim_crop = st.text_input("Crop Name", value="Wheat (गेहूं)")
                sim_name = st.text_input("Scenario Name", value="Rabi 2026 Wheat Plan")
                sim_type = st.selectbox("Scenario Assumption", [ScenarioType.BASE, ScenarioType.CONSERVATIVE, ScenarioType.OPTIMISTIC])
            with sim2:
                sim_area = st.number_input("Planned Area (Acres) *", min_value=0.1, step=0.5, value=2.5)
                sim_cost_acre = st.number_input("Expected Cost per Acre (₹) *", min_value=100.0, step=500.0, value=18500.0)
                sim_loss_pct = st.number_input("Expected Post-Harvest Loss (%)", min_value=0.0, max_value=50.0, step=1.0, value=4.0)
            with sim3:
                sim_yield_acre = st.number_input("Expected Yield per Acre (Quintals) *", min_value=0.5, step=1.0, value=19.5)
                sim_price_qtl = st.number_input("Expected Mandi Price per Quintal (₹) *", min_value=100.0, step=50.0, value=2275.0)
                sim_trans_qtl = st.number_input("Transport & Mandi Cost per Quintal (₹)", min_value=0.0, step=5.0, value=40.0)

            req = ScenarioSimulateRequest(
                crop_name=sim_crop,
                scenario_name=sim_name,
                scenario_type=sim_type,
                planned_area_acres=sim_area,
                expected_cost_per_acre=sim_cost_acre,
                expected_yield_quintals_per_acre=sim_yield_acre,
                expected_price_per_quintal=sim_price_qtl,
                expected_harvest_loss_pct=sim_loss_pct,
                transport_cost_per_quintal=sim_trans_qtl,
            )

            res = economics.scenarios.simulate_scenario(req)

            # Results
            ui_divider(style="dotted")
            st.markdown(f"##### Projected Financial Outcome ({res.scenario_type.value.title()} Case)")
            sr1, sr2, sr3, sr4 = st.columns(4)
            with sr1:
                ui_metric_card("Total Expected Cost", f"₹{res.total_expected_cost:,.2f}", delta=f"{res.planned_area_acres} Acres")
            with sr2:
                ui_metric_card("Saleable Yield", f"{res.saleable_production_quintals:.2f} Qtl", delta=f"Gross: {res.gross_production_quintals:.2f} Qtl")
            with sr3:
                ui_metric_card("Net Expected Revenue", f"₹{res.expected_net_revenue:,.2f}", delta="After Transport Deductions")
            with sr4:
                ui_metric_card("Expected Net Profit", f"₹{res.expected_net_return:,.2f}", delta=f"ROI: {res.return_on_cost_pct:.1f}%")

            # Break-even
            sb1, sb2 = st.columns(2)
            with sb1:
                ui_alert(f"Break-Even Price: ₹{res.break_even_price_per_quintal:,.2f} per quintal (Selling below this causes loss)", level="info")
            with sb2:
                ui_alert(f"Break-Even Yield: {res.break_even_yield_quintals_per_acre:.2f} quintals/acre (Yield below this causes loss)", level="info")

            # Sensitivity Analysis Table
            st.markdown("##### Sensitivity Analysis (Price vs Yield Variations)")
            sens = res.sensitivity_analysis
            st.markdown(
                f"- **Downside Case (-10% Mandi Price):** Net Return drops to **₹{sens.get('price_down_10pct', {}).get('net_return', 0):,.2f}**\n"
                f"- **Upside Case (+10% Mandi Price):** Net Return increases to **₹{sens.get('price_up_10pct', {}).get('net_return', 0):,.2f}**\n"
                f"- **Yield Stress Case (-10% Yield):** Net Return drops to **₹{sens.get('yield_down_10pct', {}).get('net_return', 0):,.2f}**"
            )

            if st.button("Save Simulation Scenario for Future Reference"):
                try:
                    saved_sc = economics.scenarios.save_scenario(active_farmer_id, req, res, user_id)
                    ui_alert(f"Saved scenario '{saved_sc.scenario_name}' (ID: {saved_sc.scenario_id})", level="success")
                except Exception as e:
                    ui_alert(f"Failed to save scenario: {e}", level="error")

        # SUB-TAB 2: MULTI-CROP COMPARISON
        with sub_comp:
            st.markdown("#### Prospective Crop Comparison on Target Plot")
            st.caption("Compare economic returns, investment requirements, and break-even points between competing crop options.")

            plot_area = st.number_input("Target Plot Size (Acres)", min_value=0.1, step=0.5, value=2.0)
            comp_season = st.selectbox("Season", ["Rabi", "Kharif", "Zaid"])

            # 3 Standard Comparison Presets
            st.markdown("##### Crop Options to Compare:")
            c_items = []

            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                st.markdown("**Option 1: Wheat (गेहूं)**")
                w_cost = st.number_input("Wheat Cost/Acre (₹)", value=18500.0, step=500.0)
                w_yield = st.number_input("Wheat Yield/Acre (Qtl)", value=19.0, step=0.5)
                w_price = st.number_input("Wheat Price/Qtl (₹)", value=2275.0, step=25.0)
                c_items.append(CropComparisonItem(
                    crop_name="Wheat",
                    estimated_cost_per_acre=w_cost,
                    expected_yield_per_acre=w_yield,
                    expected_price_per_quintal=w_price,
                    gross_revenue_per_acre=w_yield * w_price,
                    net_return_per_acre=(w_yield * w_price) - w_cost,
                    break_even_price=w_cost / w_yield if w_yield > 0 else 0.0,
                    roi_pct=(((w_yield * w_price) - w_cost) / w_cost * 100.0) if w_cost > 0 else 0.0,
                    data_source="assumptions",
                ))

            with c_col2:
                st.markdown("**Option 2: Mustard (सरसों)**")
                m_cost = st.number_input("Mustard Cost/Acre (₹)", value=13500.0, step=500.0)
                m_yield = st.number_input("Mustard Yield/Acre (Qtl)", value=8.5, step=0.5)
                m_price = st.number_input("Mustard Price/Qtl (₹)", value=5450.0, step=50.0)
                c_items.append(CropComparisonItem(
                    crop_name="Mustard",
                    estimated_cost_per_acre=m_cost,
                    expected_yield_per_acre=m_yield,
                    expected_price_per_quintal=m_price,
                    gross_revenue_per_acre=m_yield * m_price,
                    net_return_per_acre=(m_yield * m_price) - m_cost,
                    break_even_price=m_cost / m_yield if m_yield > 0 else 0.0,
                    roi_pct=(((m_yield * m_price) - m_cost) / m_cost * 100.0) if m_cost > 0 else 0.0,
                    data_source="assumptions",
                ))

            with c_col3:
                st.markdown("**Option 3: Gram / Chickpea (चना)**")
                g_cost = st.number_input("Gram Cost/Acre (₹)", value=14200.0, step=500.0)
                g_yield = st.number_input("Gram Yield/Acre (Qtl)", value=7.5, step=0.5)
                g_price = st.number_input("Gram Price/Qtl (₹)", value=5440.0, step=50.0)
                c_items.append(CropComparisonItem(
                    crop_name="Gram (Chickpea)",
                    estimated_cost_per_acre=g_cost,
                    expected_yield_per_acre=g_yield,
                    expected_price_per_quintal=g_price,
                    gross_revenue_per_acre=g_yield * g_price,
                    net_return_per_acre=(g_yield * g_price) - g_cost,
                    break_even_price=g_cost / g_yield if g_yield > 0 else 0.0,
                    roi_pct=(((g_yield * g_price) - g_cost) / g_cost * 100.0) if g_cost > 0 else 0.0,
                    data_source="assumptions",
                ))

            comp_req = CropComparisonRequest(
                plot_area_acres=plot_area,
                season=comp_season,
                state="General",
                district="General",
                crops=c_items,
            )
            comp_rep = economics.scenarios.compare_crops(comp_req)

            ui_divider(style="dotted")
            st.markdown("##### Comparative Evaluation Summary")
            ui_alert(f"Highest Total Net Return Crop: {comp_rep.best_net_return_crop} | Highest ROI % Crop: {comp_rep.best_roi_crop}", level="success")

            comp_table = []
            for it in comp_rep.comparison_items:
                tot_c = it.estimated_cost_per_acre * plot_area
                tot_r = it.gross_revenue_per_acre * plot_area
                tot_net = it.net_return_per_acre * plot_area
                comp_table.append({
                    "Crop": it.crop_name,
                    "Cost/Acre (₹)": f"₹{it.estimated_cost_per_acre:,.2f}",
                    "Expected Yield": f"{it.expected_yield_per_acre:.1f} Qtl/ac",
                    "Expected Price": f"₹{it.expected_price_per_quintal:,.2f}/Qtl",
                    "Total Cost (₹)": f"₹{tot_c:,.2f}",
                    "Total Net Return (₹)": f"₹{tot_net:,.2f}",
                    "Break-Even Price": f"₹{it.break_even_price:,.2f}",
                    "ROI (%)": f"{it.roi_pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

            st.caption(f"Disclaimer: {comp_rep.disclaimer}")
