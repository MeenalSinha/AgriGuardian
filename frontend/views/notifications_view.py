"""Streamlit View for Proactive Notifications, Alert Orchestration & Smart Farm Reminders.

Phase 9: Proactive Agricultural Assistance, Weather Conflict Analysis, and Outbox Telemetry.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import streamlit as st

from backend.app.core.config import get_settings
from backend.app.integrations.notifications.provider_registry import NotificationProviderRegistry
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import CropCycleStatus, Farm, UserRole
from backend.app.schemas.notification import (
    ConsentUpdateRequest,
    FarmReminderCreate,
    FarmReminderRecord,
    FarmReminderSnooze,
    NotificationCategory,
    NotificationChannel,
    NotificationChannelConsent,
    NotificationPreference,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    OutboxStatus,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.services.notifications.notification_service import NotificationService
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider, ui_metric_card


def get_notification_bundle():
    """Retrieve singleton notification service and repository."""
    repo = get_farm_repo()
    service = NotificationService.get_instance(repo)
    return repo, service


def render_notifications_view():
    """Main entrypoint for Notifications and Smart Reminders view."""
    st.markdown("## 🔔 Proactive Notifications & Smart Farm Reminders")
    st.caption(
        "Real-Time Weather Risk Warnings • Market Price Surges • Agronomic Milestones • "
        "Multi-Channel Delivery • Zero Data Loss Outbox"
    )

    repo, service = get_notification_bundle()
    settings = get_settings()

    # Determine Active Farmer Context
    current_farmer = st.session_state.get("current_farmer")
    all_farmers = repo.list_farmers(limit=50)

    # Context Header & Farmer Selector Bar
    col_f1, col_f2, col_f3 = st.columns([3, 2, 2])
    with col_f1:
        if all_farmers:
            farmer_options = {f.farmer_id: f"{f.full_name} ({f.district or 'Farm'}, {f.state or 'IN'})" for f in all_farmers}
            default_index = 0
            if current_farmer and current_farmer.farmer_id in farmer_options:
                default_index = list(farmer_options.keys()).index(current_farmer.farmer_id)

            selected_farmer_id = st.selectbox(
                "Active Farmer Context",
                options=list(farmer_options.keys()),
                index=default_index,
                format_func=lambda x: farmer_options[x],
                key="notif_farmer_selector",
            )
            active_farmer = repo.get_farmer_profile(selected_farmer_id)
        else:
            active_farmer = current_farmer

    with col_f2:
        if active_farmer:
            unread_count = service.get_unread_count(active_farmer.farmer_id)
            st.metric("Unread Notifications", f"{unread_count} New", delta=None)
        else:
            st.info("No registered farmer found.")
            return

    with col_f3:
        view_lang = st.radio(
            "Display Language",
            ["हिन्दी (Hindi)", "English"],
            horizontal=True,
            key="notif_view_lang",
        )
        is_hindi = "हिन्दी" in view_lang

    ui_divider(style="dotted")

    # Main Navigation Tabs
    tab_inbox, tab_reminders, tab_prefs, tab_admin = st.tabs([
        f"📬 Notification Inbox ({unread_count})",
        "⏰ Smart Farm Reminders",
        "⚙️ Notification Preferences & Consents",
        "🛡️ Outbox Queue & Channel Telemetry",
    ])

    with tab_inbox:
        render_inbox_tab(active_farmer.farmer_id, service, repo, is_hindi)

    with tab_reminders:
        render_reminders_tab(active_farmer.farmer_id, service, repo, is_hindi)

    with tab_prefs:
        render_preferences_tab(active_farmer.farmer_id, service, repo, is_hindi)

    with tab_admin:
        render_admin_outbox_tab(service, repo, settings)


# ==============================================================================
# TAB 1: NOTIFICATION INBOX
# ==============================================================================

def render_inbox_tab(farmer_id: str, service: NotificationService, repo: FarmDatabaseRepository, is_hindi: bool):
    """Render notification cards with filters, read status, and dismiss actions."""
    st.subheader("📬 Your Farm Notifications")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        cat_filter = st.selectbox(
            "Category Filter",
            [
                "All Categories",
                "Weather Alerts",
                "Market Intelligence",
                "Expert Reviews",
                "Crop Reminders",
                "Advisory Updates",
            ],
            key="inbox_cat_filter",
        )
    with col_ctrl2:
        unread_only = st.checkbox("Show Unread Only", value=False, key="inbox_unread_only")

    with col_ctrl3:
        if st.button("Mark All as Read ✓", use_container_width=True, key="inbox_mark_all_read"):
            cnt = service.mark_all_as_read(farmer_id)
            st.success(f"Marked {cnt} notifications as read.")
            st.rerun()

    # Map category filter to enum
    category_map = {
        "Weather Alerts": NotificationCategory.WEATHER_ALERT.value,
        "Market Intelligence": NotificationCategory.MARKET_INTELLIGENCE.value,
        "Expert Reviews": NotificationCategory.EXPERT_REVIEW.value,
        "Crop Reminders": NotificationCategory.CROP_REMINDER.value,
        "Advisory Updates": NotificationCategory.ADVISORY_UPDATE.value,
    }
    selected_cat = category_map.get(cat_filter)

    notifications = service.list_notifications(
        farmer_id=farmer_id,
        category=selected_cat,
        unread_only=unread_only,
        limit=50,
    )

    if not notifications:
        st.info("No notifications match your current filter.")
        # Offer quick demo trigger
        st.caption("Tip: You can emit a demo alert from the smart reminders or weather tab to test live dispatch.")
        return

    priority_icons = {
        NotificationPriority.EMERGENCY: "🚨 EMERGENCY",
        NotificationPriority.HIGH: "⚠️ HIGH",
        NotificationPriority.MEDIUM: "ℹ️ INFO",
        NotificationPriority.LOW: "🔹 NOTICE",
    }
    category_icons = {
        NotificationCategory.WEATHER_ALERT: "🌦️",
        NotificationCategory.MARKET_INTELLIGENCE: "📈",
        NotificationCategory.EXPERT_REVIEW: "👨‍⚕️",
        NotificationCategory.CROP_REMINDER: "⏰",
        NotificationCategory.ADVISORY_UPDATE: "🌾",
        NotificationCategory.SYSTEM: "⚙️",
    }

    for n in notifications:
        is_unread = n.status == NotificationStatus.UNREAD
        card_border = "#FF5722" if n.priority == NotificationPriority.EMERGENCY else ("#2E7D32" if is_unread else "#E0E0E0")
        card_bg = "#FFF9C4" if n.priority == NotificationPriority.EMERGENCY else ("#F1F8E9" if is_unread else "#FAFAFA")

        title = n.title_hi if is_hindi and n.title_hi else n.title_en
        body = n.body_hi if is_hindi and n.body_hi else n.body_en
        p_label = priority_icons.get(n.priority, "INFO")
        cat_icon = category_icons.get(n.category, "📢")

        with st.container():
            st.markdown(
                f"""
                <div style="border-left: 6px solid {card_border}; background-color: {card_bg}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.05rem; color: #1B5E20;">{cat_icon} {title}</span>
                        <span style="font-size: 0.8rem; background-color: #E0E0E0; padding: 2px 8px; border-radius: 12px; font-weight: 600;">{p_label}</span>
                    </div>
                    <p style="margin-top: 8px; margin-bottom: 8px; font-size: 0.95rem; color: #333; line-height: 1.4;">{body}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #757575;">
                        <span>Received: {n.created_at[:16].replace('T', ' ')}</span>
                        <span>Status: <b>{n.status.value.upper()}</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 4])
            with btn_col1:
                if is_unread:
                    if st.button("Mark as Read ✓", key=f"read_{n.notification_id}", use_container_width=True):
                        service.mark_as_read(n.notification_id, farmer_id=farmer_id)
                        st.rerun()
            with btn_col2:
                if st.button("Dismiss 🗑️", key=f"dismiss_{n.notification_id}", use_container_width=True):
                    service.dismiss_notification(n.notification_id, farmer_id=farmer_id)
                    st.rerun()
            with btn_col3:
                if n.action_url:
                    st.caption(f"🔗 Related Section: `{n.action_url}`")


# ==============================================================================
# TAB 2: SMART FARM REMINDERS & WEATHER CONFLICTS
# ==============================================================================

def render_reminders_tab(farmer_id: str, service: NotificationService, repo: FarmDatabaseRepository, is_hindi: bool):
    """Render upcoming and overdue farm activity reminders with weather conflict badges."""
    st.subheader("⏰ Smart Farm Activity Reminders")
    st.caption(
        "Agronomic Milestones • Proactive Spray Warnings • Rain Conflict Analysis • Snooze & Reschedule"
    )

    # Sync action bar
    sync_col1, sync_col2 = st.columns([3, 1])
    with sync_col1:
        st.markdown(
            "Smart Reminders are synchronized with your active crop cycles from **Phase 6 Farm Management**. "
            "Chemical applications and irrigations are cross-referenced with meteorological risk models."
        )
    with sync_col2:
        if st.button("🔄 Sync Crop Cycle Tasks", use_container_width=True, key="sync_reminders_btn"):
            created = service.generate_reminders_from_cycles(farmer_id)
            if created:
                st.success(f"Generated {len(created)} new agronomic reminders from active crop cycles!")
            else:
                st.info("All crop cycle milestones are already synchronized.")
            st.rerun()

    # List reminders
    reminders = service.list_reminders(farmer_id=farmer_id)

    # Split into Pending/Overdue and Completed
    active_reminders = [r for r in reminders if r.status in (ReminderStatus.PENDING, ReminderStatus.SNOOZED)]
    completed_reminders = [r for r in reminders if r.status == ReminderStatus.COMPLETED]

    today_str = datetime.now().strftime("%Y-%m-%d")

    if not active_reminders:
        st.info("No active reminders scheduled. Add a reminder below or sync from active crop cycles.")
    else:
        st.markdown(f"#### Active Reminders ({len(active_reminders)})")
        for rem in active_reminders:
            is_overdue = rem.scheduled_date < today_str
            status_chip = "🔴 OVERDUE" if is_overdue else ("⏸️ SNOOZED" if rem.status == ReminderStatus.SNOOZED else "📅 SCHEDULED")
            border_color = "#D32F2F" if is_overdue else ("#FF9800" if rem.has_weather_conflict else "#388E3C")

            with st.container():
                st.markdown(
                    f"""
                    <div style="border-left: 6px solid {border_color}; background-color: #FAFAFA; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.05rem; color: #1B5E20;">
                                {rem.activity_type.value.title()}: {rem.title_hi if is_hindi and rem.title_hi else rem.title_en}
                            </span>
                            <span style="font-size: 0.8rem; background-color: #E0E0E0; padding: 2px 8px; border-radius: 12px; font-weight: 600;">{status_chip}</span>
                        </div>
                        <p style="margin-top: 6px; margin-bottom: 6px; font-size: 0.92rem; color: #424242;">
                            {rem.description_hi if is_hindi and rem.description_hi else rem.description_en}
                        </p>
                        <div style="font-size: 0.82rem; color: #616161;">
                            <b>Scheduled Date:</b> {rem.scheduled_date} &nbsp;•&nbsp; <b>Activity:</b> {rem.activity_type.value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Weather Conflict Warning Banner
                if rem.has_weather_conflict:
                    conflict_text = rem.weather_risk_reason_hi if is_hindi and rem.weather_risk_reason_hi else rem.weather_risk_reason_en
                    st.warning(
                        f"⚠️ **Adverse Weather Conflict Detected:**\n\n{conflict_text}\n\n"
                        f"*Recommendation:* Snooze this activity to avoid pesticide runoff or fertilizer leaching."
                    )

                # Action Controls
                act_col1, act_col2, act_col3, act_col4 = st.columns([2, 1.5, 1.5, 2])
                with act_col1:
                    if st.button("Complete Task ✅", key=f"comp_{rem.reminder_id}", use_container_width=True):
                        service.complete_reminder(rem.reminder_id, farmer_id=farmer_id)
                        st.success("Activity marked as completed!")
                        st.rerun()
                with act_col2:
                    if st.button("Snooze +1d ⏸️", key=f"sn1_{rem.reminder_id}", use_container_width=True):
                        service.snooze_reminder(rem.reminder_id, snooze_days=1, farmer_id=farmer_id)
                        st.info("Snoozed for 1 day.")
                        st.rerun()
                with act_col3:
                    if st.button("Snooze +3d ⏸️", key=f"sn3_{rem.reminder_id}", use_container_width=True):
                        service.snooze_reminder(rem.reminder_id, snooze_days=3, farmer_id=farmer_id)
                        st.info("Snoozed for 3 days.")
                        st.rerun()
                with act_col4:
                    custom_date = st.date_input(
                        "Reschedule To",
                        value=datetime.now() + timedelta(days=2),
                        key=f"cust_date_{rem.reminder_id}",
                    )
                    if st.button("Set Date 🗓️", key=f"set_d_{rem.reminder_id}", use_container_width=True):
                        service.snooze_reminder(
                            rem.reminder_id,
                            custom_date=custom_date.strftime("%Y-%m-%d"),
                            farmer_id=farmer_id,
                        )
                        st.info(f"Rescheduled to {custom_date.strftime('%Y-%m-%d')}.")
                        st.rerun()

                st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    # Add Custom Reminder Form
    ui_divider(style="dotted")
    with st.expander("➕ Add Custom Farm Reminder", expanded=False):
        farms = repo.get_farms_by_farmer(farmer_id)
        farm_choices = {f.farm_id: f.farm_name for f in farms}
        farm_choices[None] = "General / No Farm"

        with st.form("create_reminder_form"):
            col_cr1, col_cr2 = st.columns(2)
            with col_cr1:
                sel_farm = st.selectbox(
                    "Select Farm",
                    options=list(farm_choices.keys()),
                    format_func=lambda x: farm_choices[x],
                )
                act_type = st.selectbox(
                    "Activity Type",
                    options=[e.value for e in ReminderActivityType],
                    format_func=lambda x: x.replace("_", " ").title(),
                )
                title_en = st.text_input("Title (English) *", placeholder="e.g. Apply Fungicide Spray")
                title_hi = st.text_input("Title (Hindi Optional)", placeholder="e.g. फफूंदनाशक छिड़काव करें")

            with col_cr2:
                due_date = st.date_input("Scheduled Date *", value=datetime.now() + timedelta(days=1))
                desc_en = st.text_area("Description / Instructions (English) *", placeholder="Specify dosage and plot notes...")
                desc_hi = st.text_area("Description (Hindi Optional)", placeholder="निर्देश और खुराक...")

            sub_rem = st.form_submit_button("Schedule Reminder 🗓️", use_container_width=True)
            if sub_rem:
                if not title_en or not desc_en:
                    st.error("Please fill in Title and Description.")
                else:
                    new_rem = service.create_reminder(
                        farmer_id=farmer_id,
                        request=FarmReminderCreate(
                            farm_id=sel_farm,
                            activity_type=ReminderActivityType(act_type),
                            title_en=title_en,
                            title_hi=title_hi or title_en,
                            description_en=desc_en,
                            description_hi=desc_hi or desc_en,
                            scheduled_date=due_date.strftime("%Y-%m-%d"),
                        ),
                    )
                    st.success(f"Reminder '{new_rem.title_en}' scheduled for {new_rem.scheduled_date}!")
                    if new_rem.has_weather_conflict:
                        st.warning(f"⚠️ Weather Warning: {new_rem.weather_risk_reason_en}")
                    st.rerun()


# ==============================================================================
# TAB 3: NOTIFICATION PREFERENCES & CONSENTS
# ==============================================================================

def render_preferences_tab(farmer_id: str, service: NotificationService, repo: FarmDatabaseRepository, is_hindi: bool):
    """Render channel switches, quiet hours, and DPDP-compliant channel consent audit."""
    st.subheader("⚙️ Notification Preferences & Channel Consents")
    st.caption("Manage Channel Subscriptions • Quiet Hours Protection • Explicit DPDP Consent Tracking")

    pref = service.get_preferences(farmer_id)
    consents = service.get_channel_consents(farmer_id)
    consent_dict = {c.channel: c for c in consents}

    with st.form("pref_form"):
        st.markdown("#### 📡 Communication Channels")
        st.caption("Select which channels AgriGuardian may use to reach you:")

        col_ch1, col_ch2, col_ch3 = st.columns(3)
        with col_ch1:
            ch_inapp = st.checkbox("In-App Notification Center", value=True, disabled=True, help="In-app alerts are always active.")
            ch_sms = st.checkbox(
                "SMS Text Messages",
                value=NotificationChannel.SMS in pref.channels_enabled,
                help="Requires explicit opt-in consent under DPDP Act.",
            )
        with col_ch2:
            ch_wa = st.checkbox(
                "WhatsApp Alerts",
                value=NotificationChannel.WHATSAPP in pref.channels_enabled,
                help="Rich cards and localized alerts over WhatsApp.",
            )
            ch_voice = st.checkbox(
                "Voice Calls (Gnani Timbre)",
                value=NotificationChannel.VOICE_CALL in pref.channels_enabled,
                help="Spoken alerts for high priority and emergency warnings.",
            )
        with col_ch3:
            ch_email = st.checkbox(
                "Email Digests",
                value=NotificationChannel.EMAIL in pref.channels_enabled,
            )

        ui_divider(style="dotted")

        st.markdown("#### 🌙 Quiet Hours")
        st.caption("During quiet hours, routine notifications will only appear in-app, deferring noisy external channels.")

        col_qh1, col_qh2, col_qh3 = st.columns(3)
        with col_qh1:
            qh_start = st.text_input("Quiet Hours Start (HH:MM)", value=pref.quiet_hours_start)
        with col_qh2:
            qh_end = st.text_input("Quiet Hours End (HH:MM)", value=pref.quiet_hours_end)
        with col_qh3:
            em_override = st.checkbox(
                "Emergency Weather Alerts Bypass Quiet Hours",
                value=pref.emergency_override,
                help="Critical flash floods, hailstorms, and extreme warnings will immediately ring/SMS even at night.",
            )

        ui_divider(style="dotted")

        st.markdown("#### 📂 Subscribed Categories")
        col_cat1, col_cat2 = st.columns(2)
        with col_cat1:
            cat_weather = st.checkbox("Weather Hazards & Advisories", value=NotificationCategory.WEATHER_ALERT in pref.categories_enabled)
            cat_market = st.checkbox("Mandi Market Price Surges", value=NotificationCategory.MARKET_INTELLIGENCE in pref.categories_enabled)
        with col_cat2:
            cat_rem = st.checkbox("Smart Crop Activity Reminders", value=NotificationCategory.CROP_REMINDER in pref.categories_enabled)
            cat_exp = st.checkbox("Agronomist Expert Review Updates", value=NotificationCategory.EXPERT_REVIEW in pref.categories_enabled)

        save_btn = st.form_submit_button("Save Notification Preferences 💾", use_container_width=True)
        if save_btn:
            selected_channels = [NotificationChannel.IN_APP]
            if ch_sms:
                selected_channels.append(NotificationChannel.SMS)
            if ch_wa:
                selected_channels.append(NotificationChannel.WHATSAPP)
            if ch_voice:
                selected_channels.append(NotificationChannel.VOICE_CALL)
            if ch_email:
                selected_channels.append(NotificationChannel.EMAIL)

            selected_categories = []
            if cat_weather:
                selected_categories.append(NotificationCategory.WEATHER_ALERT)
            if cat_market:
                selected_categories.append(NotificationCategory.MARKET_INTELLIGENCE)
            if cat_rem:
                selected_categories.append(NotificationCategory.CROP_REMINDER)
            if cat_exp:
                selected_categories.append(NotificationCategory.EXPERT_REVIEW)

            updated_pref = NotificationPreference(
                farmer_id=farmer_id,
                channels_enabled=selected_channels,
                categories_enabled=selected_categories,
                quiet_hours_start=qh_start,
                quiet_hours_end=qh_end,
                emergency_override=em_override,
                preferred_language=pref.preferred_language,
            )
            service.save_preferences(updated_pref)
            st.success("Preferences updated successfully!")
            st.rerun()

    # Consent Audit Trail (DPDP Compliance)
    ui_divider(style="dotted")
    st.markdown("#### 📋 DPDP Channel Consent Audit Trail")
    st.caption("Under the Digital Personal Data Protection Act, farmers have explicit rights to view, grant, and withdraw channel consents.")

    for ch in [NotificationChannel.SMS, NotificationChannel.WHATSAPP, NotificationChannel.VOICE_CALL, NotificationChannel.EMAIL]:
        c_record = consent_dict.get(ch)
        is_granted = c_record.is_granted if c_record else False
        status_text = "🟢 Consent Granted" if is_granted else "⚪ Consent Not Granted / Revoked"
        consent_time = c_record.updated_at[:16].replace("T", " ") if c_record else "Never"

        c_col1, c_col2, c_col3 = st.columns([3, 2, 2])
        with c_col1:
            st.markdown(f"**Channel:** `{ch.value.upper()}` — {status_text}")
        with c_col2:
            st.caption(f"Last Modified: {consent_time}")
        with c_col3:
            btn_label = "Revoke Consent" if is_granted else "Grant Opt-In"
            if st.button(btn_label, key=f"c_btn_{ch.value}"):
                service.update_channel_consent(
                    farmer_id=farmer_id,
                    req=ConsentUpdateRequest(
                        channel=ch,
                        is_granted=not is_granted,
                        consent_source="farmer_preferences_ui",
                    ),
                )
                st.info(f"Consent for {ch.value} updated.")
                st.rerun()


# ==============================================================================
# TAB 4: ADMIN OUTBOX QUEUE & TELEMETRY (ROLE-GATED)
# ==============================================================================

def render_admin_outbox_tab(service: NotificationService, repo: FarmDatabaseRepository, settings):
    """Render Outbox queue health, worker batch processing, and truthful provider telemetry."""
    st.subheader("🛡️ Transactional Outbox & Channel Gateway Telemetry")
    st.caption(
        "Guaranteed At-Least-Once Delivery • Worker Lease Recovery • Dead-Letter Queue • Truth-First Telemetry"
    )

    # 1. Telemetry Metrics
    telemetry = service.get_outbox_telemetry()
    status_counts = telemetry.get("status_counts", {})
    pending_cnt = status_counts.get("pending", 0)
    processing_cnt = status_counts.get("processing", 0)
    delivered_cnt = status_counts.get("delivered", 0)
    failed_cnt = status_counts.get("failed", 0)
    dead_letter_cnt = status_counts.get("dead_letter", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Pending In Queue", pending_cnt)
    with m2:
        st.metric("Processing Leases", processing_cnt)
    with m3:
        st.metric("Successfully Delivered", delivered_cnt)
    with m4:
        st.metric("Retrying (Transient)", failed_cnt)
    with m5:
        st.metric("Dead-Letter (DLQ)", dead_letter_cnt)

    ui_divider(style="dotted")

    # 2. Worker Trigger Action Bar
    w_col1, w_col2 = st.columns([3, 1])
    with w_col1:
        st.markdown(
            "The background worker claims pending jobs with an **atomic lease** to prevent double-delivery. "
            "Expired leases (>60s) from aborted workers are automatically reclaimed."
        )
    with w_col2:
        if st.button("🚀 Process Outbox Batch", use_container_width=True, key="proc_outbox_batch_btn"):
            res = service.process_outbox(batch_size=20)
            st.success(
                f"Batch Complete: {res['processed']} processed | {res['delivered']} delivered | "
                f"{res['failed']} failed | {res['dead_lettered']} moved to dead-letter."
            )
            st.rerun()

    # 3. Channel Provider Readiness Status Cards
    st.markdown("#### 🔌 Physical Channel Gateways Readiness")
    providers_telemetry = service.get_channels_telemetry()

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        for ch_key, info in list(providers_telemetry.items())[:3]:
            status_icon = "🟢" if info["configured"] else "🟡"
            status_label = "Configured & Active" if info["configured"] else "Provider Not Configured"
            st.markdown(
                f"""
                <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <b>{status_icon} Channel: {ch_key.upper()}</b><br>
                    <span style="font-size: 0.85rem; color: #555;">Gateway: {info['provider']} &nbsp;•&nbsp; Status: <b>{status_label}</b></span><br>
                    <span style="font-size: 0.78rem; color: #777;">Mode: {info['environment_mode']} &nbsp;•&nbsp; Note: {info['note']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_g2:
        for ch_key, info in list(providers_telemetry.items())[3:]:
            status_icon = "🟢" if info["configured"] else "🟡"
            status_label = "Configured & Active" if info["configured"] else "Provider Not Configured"
            st.markdown(
                f"""
                <div style="background-color: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <b>{status_icon} Channel: {ch_key.upper()}</b><br>
                    <span style="font-size: 0.85rem; color: #555;">Gateway: {info['provider']} &nbsp;•&nbsp; Status: <b>{status_label}</b></span><br>
                    <span style="font-size: 0.78rem; color: #777;">Mode: {info['environment_mode']} &nbsp;•&nbsp; Note: {info['note']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Outbox Queue Inspection Table
    ui_divider(style="dotted")
    st.markdown("#### 📋 Outbox Queue Inspection")
    q_filter = st.selectbox(
        "Queue Status Filter",
        ["All Outbox Items", "dead_letter", "pending", "failed", "delivered", "processing"],
        key="outbox_status_filter",
    )
    selected_status = None if q_filter == "All Outbox Items" else q_filter

    items = repo.list_outbox_items(status=selected_status, limit=30)
    if not items:
        st.info("No outbox records match the selected status.")
    else:
        for item in items:
            with st.expander(
                f"[{item.status.value.upper()}] {item.channel.value.upper()} → {item.recipient} ({item.outbox_id})",
                expanded=False,
            ):
                st.caption(f"**Payload Title:** {item.payload.get('title', 'N/A')}")
                st.caption(f"**Body:** {item.payload.get('body', 'N/A')}")
                st.caption(f"**Retry Count:** {item.retry_count} / {item.max_retries}")
                if item.last_error:
                    st.error(f"Last Error: {item.last_error}")

                if item.status == OutboxStatus.DEAD_LETTER:
                    if st.button("Retry Dead-Letter Item 🔄", key=f"dlq_retry_{item.outbox_id}"):
                        ok = service.retry_dead_letter(item.outbox_id)
                        if ok:
                            st.success("Re-queued for immediate dispatch!")
                            st.rerun()
