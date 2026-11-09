"""Streamlit View for Offline-First Synchronization & PWA Companion Management.

Phase 10: Low-Connectivity Resilience, Device Registration, Change Logs, and Conflict Resolution.
"""

from datetime import datetime, timezone
from typing import List, Optional
import streamlit as st

from backend.app.core.config import get_settings
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import ConflictResolutionStrategy, ConflictResolveRequest
from backend.app.services.sync.sync_service import SyncService
from frontend.components.ui_widgets import ui_alert, ui_card, ui_divider, ui_metric_card


def get_sync_bundle():
    """Retrieve singleton sync service and farm database repository."""
    repo = get_farm_repo()
    service = SyncService.get_instance(repo)
    return repo, service


def render_offline_sync_view():
    """Render the comprehensive Offline Sync & PWA management dashboard."""
    repo, sync_service = get_sync_bundle()
    settings = get_settings()

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
                    color: white; padding: 20px; border-radius: 12px; margin-bottom: 24px;">
            <h1 style="color: white; margin: 0; font-size: 26px;">
                🔄 Offline-First Synchronization & PWA Companion
            </h1>
            <p style="margin: 8px 0 0 0; opacity: 0.92; font-size: 15px;">
                Resilient field operations for Indian agriculture: zero data loss during network disruptions,
                durable IndexedDB journal queuing, monotonic change-feed replication, and intelligent conflict resolution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📱 Mobile PWA Portal",
        "💻 Registered Devices",
        "📜 Change Feed Log",
        "⚡ Mutations & Idempotency",
        "🛡️ Conflict Resolution",
        "📊 Telemetry & Health",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Mobile PWA Portal
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("📲 Installable Progressive Web App (PWA)")
        st.markdown(
            """
            AgriGuardian includes an **installable Progressive Web App** specifically designed for farmers
            working in fields where cellular reception is weak or completely absent.
            """
        )

        col_pwa_1, col_pwa_2 = st.columns([2, 1])
        with col_pwa_1:
            ui_card(
                title="🌾 Open Farmer PWA",
                content=(
                    "**PWA Endpoint URL:** `http://localhost:8000/pwa/index.html`\n\n"
                    "**Core Offline Capabilities:**\n"
                    "- 💾 **Full Offline Farm Context:** My Farms, Plots, Crops, Soil, and Saved Calendars.\n"
                    "- 🚜 **Offline Farm Logging:** Record irrigation, spray, weeding, and fertilizer actions.\n"
                    "- 📷 **Offline Leaf Symptom Photo Drafts:** Capture photos in field; auto-upload for AI diagnosis.\n"
                    "- ⏰ **Smart Reminders:** Offline completion with optimistic locking.\n"
                    "- 🛡️ **Zero Data Loss:** Durable IndexedDB mutation queue surviving browser refreshes.\n"
                    "- 🌐 **Bilingual Touch UI:** English & Hindi with large touch targets."
                ),
            )
            st.markdown(
                """
                <a href="http://localhost:8000/pwa/index.html" target="_blank"
                   style="display: inline-block; background-color: #2d6a4f; color: white;
                          padding: 12px 24px; border-radius: 8px; text-decoration: none;
                          font-weight: 700; margin-top: 10px;">
                    🚀 Launch Farmer PWA in Browser / Mobile
                </a>
                """,
                unsafe_allow_html=True,
            )

        with col_pwa_2:
            st.markdown("### 📋 Installation Guide")
            st.info(
                "**On Android (Chrome):**\n"
                "1. Open `/pwa/index.html`\n"
                "2. Tap browser menu (⋮) -> 'Install app' or 'Add to Home screen'\n"
                "3. AgriGuardian installs as a standalone native-like app\n"
                "4. Works even in Airplane mode!"
            )
            st.success(
                "**Service Worker State:**\n"
                "- Cache: `agriguardian-shell-v1.10.0`\n"
                "- App Shell: Pre-cached\n"
                "- Storage: IndexedDB (`AgriGuardianOfflineDB`)"
            )

    # -------------------------------------------------------------------------
    # TAB 2: Registered Devices
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("📱 Registered Sync Devices")
        st.caption("Active and revoked devices authorized to synchronize farmer context.")

        with repo.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT d.device_id, d.user_id, d.farmer_id, d.device_name, d.device_type,
                       d.app_version, d.last_sync_at, d.is_revoked, d.created_at
                FROM sync_devices d
                ORDER BY d.created_at DESC
                """
            )
            devices = [dict(r) for r in cursor.fetchall()]

        if not devices:
            st.info("No physical or PWA devices registered yet. Devices register automatically upon opening the PWA.")
        else:
            for dev in devices:
                with st.expander(
                    f"{'🔴 [REVOKED]' if dev['is_revoked'] else '🟢 [ACTIVE]'} {dev['device_name']} ({dev['device_type']})",
                    expanded=not dev['is_revoked'],
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Device ID:** `{dev['device_id']}`")
                        st.markdown(f"**Device Type:** `{dev['device_type']}`")
                    with c2:
                        st.markdown(f"**User ID:** `{dev['user_id']}`")
                        st.markdown(f"**Farmer ID:** `{dev.get('farmer_id') or 'N/A'}`")
                    with c3:
                        st.markdown(f"**App Version:** `{dev['app_version']}`")
                        st.markdown(f"**Last Sync:** `{dev.get('last_sync_at') or 'Never'}`")

                    if not dev["is_revoked"]:
                        if st.button(f"🚫 Revoke Device Access", key=f"rev_{dev['device_id']}"):
                            sync_service.revoke_device(dev["device_id"], dev["user_id"])
                            st.warning(f"Device {dev['device_id']} revoked. It cannot sync future data.")
                            st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: Change Feed Log
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("📜 Monotonic Sync Change Feed (`sync_change_log`)")
        st.caption("Immutable append-only change log powering incremental pull synchronization.")

        with repo.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT sequence_id, resource_type, resource_id, farmer_id, operation,
                       version_number, created_at
                FROM sync_change_log
                ORDER BY sequence_id DESC
                LIMIT 50
                """
            )
            changes = [dict(r) for r in cursor.fetchall()]

        if not changes:
            st.info("No changes recorded in change feed yet.")
        else:
            st.dataframe(changes, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Mutations & Idempotency
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("⚡ Inbound Client Mutations Journal")
        st.caption("Durable queue of pushed offline mutations with idempotency execution results.")

        with repo.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT m.mutation_id, m.device_id, m.resource_type, m.operation,
                       m.status AS mutation_status, r.status AS result_status,
                       r.server_record_id, r.server_version, m.received_at
                FROM sync_mutations m
                LEFT JOIN sync_mutation_results r ON m.mutation_id = r.mutation_id
                ORDER BY m.received_at DESC
                LIMIT 50
                """
            )
            mutations = [dict(r) for r in cursor.fetchall()]

        if not mutations:
            st.info("No mutations received yet.")
        else:
            st.dataframe(mutations, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: Conflict Resolution
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("🛡️ Edit Conflict Management (`sync_conflicts`)")
        st.caption("Safely resolves concurrent edits without silent last-write-wins data loss.")

        with repo.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT conflict_id, mutation_id, device_id, farmer_id, resource_type,
                       client_record_id, server_record_id, resolution_strategy, status, created_at
                FROM sync_conflicts
                ORDER BY created_at DESC
                """
            )
            conflicts = [dict(r) for r in cursor.fetchall()]

        if not conflicts:
            st.success("✅ No conflicting edits recorded. Local and cloud data remain aligned.")
        else:
            for c in conflicts:
                with st.expander(f"⚠️ Conflict {c['conflict_id']} ({c['resource_type']}) - Status: {c['status']}"):
                    st.json(c)
                    if c["status"] == "pending":
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            if st.button("📱 Keep Client Version", key=f"cw_{c['conflict_id']}"):
                                sync_service.resolve_conflict(
                                    conflict_id=c["conflict_id"],
                                    farmer_id=c["farmer_id"],
                                    req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.CLIENT_WINS),
                                )
                                st.success("Resolved using CLIENT_WINS.")
                                st.rerun()
                        with col_c2:
                            if st.button("☁️ Keep Server Version", key=f"sw_{c['conflict_id']}"):
                                sync_service.resolve_conflict(
                                    conflict_id=c["conflict_id"],
                                    farmer_id=c["farmer_id"],
                                    req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.SERVER_WINS),
                                )
                                st.info("Resolved using SERVER_WINS.")
                                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 6: Telemetry & Health
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("📊 Sync Telemetry & System Health")

        status_info = sync_service.get_sync_status()
        st.json(status_info)

        with repo.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, device_id, sync_type, duration_ms, pull_count,
                       push_count, conflict_count, status, created_at
                FROM sync_telemetry
                ORDER BY created_at DESC
                LIMIT 25
                """
            )
            telemetry = [dict(r) for r in cursor.fetchall()]

        if telemetry:
            st.markdown("### Recent Synchronization Sessions")
            st.dataframe(telemetry, use_container_width=True)
