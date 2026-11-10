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


def get_sync_bundle():
    """Retrieve singleton sync service and farm database repository."""
    repo = get_farm_repo()
    service = SyncService.get_instance(repo)
    return repo, service


def render_offline_sync_view():
    """Render the comprehensive Offline Sync & PWA management dashboard."""
    repo, sync_service = get_sync_bundle()
    settings = get_settings()

    ui_page_header(
        title="Offline-First Synchronization & PWA Companion",
        subtitle="Zero data loss during network disruptions, durable IndexedDB journals, monotonic change-feed replication, and conflict resolution.",
        pills=["IndexedDB Storage", "Idempotent Queue", "Two-Way Sync Center", "Conflict Resolution Desk"],
        icon_name="refresh-cw",
    )

    tabs = st.tabs([
        "Mobile PWA Portal",
        "Registered Devices",
        "Change Feed Log",
        "Mutations & Idempotency",
        "Conflict Resolution",
        "Telemetry & Health",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Mobile PWA Portal
    # -------------------------------------------------------------------------
    with tabs[0]:
        ui_section_header("Installable Progressive Web App (PWA)", "Offline-capable companion application for fields with weak or absent cellular connectivity.", "wifi-off")

        col_pwa_1, col_pwa_2 = st.columns([2, 1])
        with col_pwa_1:
            ui_card(
                title="Farmer PWA Client",
                content=(
                    "**PWA Endpoint URL:** `http://localhost:8000/pwa/index.html`\n\n"
                    "**Core Offline Capabilities:**\n"
                    "- **Full Offline Farm Context:** My Farms, Plots, Crops, Soil, and Saved Calendars.\n"
                    "- **Offline Farm Logging:** Record irrigation, spray, weeding, and fertilizer actions.\n"
                    "- **Offline Leaf Symptom Photo Drafts:** Capture photos in field; auto-upload for AI diagnosis.\n"
                    "- **Smart Reminders:** Offline completion with optimistic locking.\n"
                    "- **Zero Data Loss:** Durable IndexedDB mutation queue surviving browser refreshes.\n"
                    "- **Bilingual Touch UI:** English and Hindi with high-contrast accessibility targets."
                ),
            )
            st.markdown(
                """
                <a href="http://localhost:8000/pwa/index.html" target="_blank"
                   style="display: inline-block; background-color: var(--primary); color: white;
                          padding: 12px 24px; border-radius: var(--radius-md); text-decoration: none;
                          font-weight: 700; margin-top: 10px; box-shadow: var(--shadow-sm);">
                    Launch Farmer PWA in Browser / Mobile
                </a>
                """,
                unsafe_allow_html=True,
            )

        with col_pwa_2:
            st.markdown("#### Installation Guide")
            st.info(
                "**On Android (Chrome):**\n"
                "1. Open `/pwa/index.html`\n"
                "2. Tap browser menu (three dots) -> 'Install app' or 'Add to Home screen'\n"
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
        ui_section_header("Registered Sync Devices", "Active and revoked devices authorized to synchronize farmer context.", "activity")

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
            ui_alert("No physical or PWA devices registered yet. Devices register automatically upon opening the PWA.", level="info")
        else:
            for dev in devices:
                status_text = "[REVOKED]" if dev['is_revoked'] else "[ACTIVE]"
                with st.expander(
                    f"{status_text} {dev['device_name']} ({dev['device_type']})",
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
                        if st.button("Revoke Device Access", key=f"rev_{dev['device_id']}"):
                            sync_service.revoke_device(dev["device_id"], dev["user_id"])
                            ui_alert(f"Device {dev['device_id']} revoked. It cannot sync future data.", level="warning")
                            st.rerun()

    # -------------------------------------------------------------------------
    # TAB 3: Change Feed Log
    # -------------------------------------------------------------------------
    with tabs[2]:
        ui_section_header("Monotonic Sync Change Feed (sync_change_log)", "Immutable append-only change log powering incremental pull synchronization.", "database")

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
            ui_alert("No changes recorded in change feed yet.", level="info")
        else:
            st.dataframe(changes, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Mutations & Idempotency
    # -------------------------------------------------------------------------
    with tabs[3]:
        ui_section_header("Inbound Client Mutations Journal", "Durable queue of pushed offline mutations with idempotency execution results.", "activity")

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
            ui_alert("No mutations received yet.", level="info")
        else:
            st.dataframe(mutations, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: Conflict Resolution
    # -------------------------------------------------------------------------
    with tabs[4]:
        ui_section_header("Edit Conflict Management (sync_conflicts)", "Safely resolves concurrent edits without silent last-write-wins data loss.", "shield-check")

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
            ui_alert("No conflicting edits recorded. Local and cloud data remain aligned.", level="success")
        else:
            for c in conflicts:
                with st.expander(f"Conflict {c['conflict_id']} ({c['resource_type']}) - Status: {c['status']}"):
                    st.json(c)
                    if c["status"] == "pending":
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            if st.button("Keep Client Version", key=f"cw_{c['conflict_id']}"):
                                sync_service.resolve_conflict(
                                    conflict_id=c["conflict_id"],
                                    farmer_id=c["farmer_id"],
                                    req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.CLIENT_WINS),
                                )
                                ui_alert("Resolved using CLIENT_WINS.", level="success")
                                st.rerun()
                        with col_c2:
                            if st.button("Keep Server Version", key=f"sw_{c['conflict_id']}"):
                                sync_service.resolve_conflict(
                                    conflict_id=c["conflict_id"],
                                    farmer_id=c["farmer_id"],
                                    req=ConflictResolveRequest(resolution_strategy=ConflictResolutionStrategy.SERVER_WINS),
                                )
                                ui_alert("Resolved using SERVER_WINS.", level="info")
                                st.rerun()

    # -------------------------------------------------------------------------
    # TAB 6: Telemetry & Health
    # -------------------------------------------------------------------------
    with tabs[5]:
        ui_section_header("Sync Telemetry & System Health", "Operational latency, transaction volume, and network status.", "bar-chart")

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
            st.markdown("#### Recent Synchronization Sessions")
            st.dataframe(telemetry, use_container_width=True)
