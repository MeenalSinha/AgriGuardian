# AGRIGUARDIAN — PHASE 9 FINAL ENGINEERING REPORT
## Proactive Notifications, Intelligent Alert Orchestration & Smart Farm Reminders

**System Version:** 2.9.0-prod  
**Date of Completion:** November 9, 2026  
**Operating Environment:** Windows / Python 3.14.2 / PyTorch 2.10.0+cpu / SQLite 3  
**Verified Test Suite:** 240 Passed, 2 Skipped, 0 Failed (100% Pass Rate across Phases 0–9)

---

## 1. Executive Summary

Phase 9 transforms AgriGuardian from a reactive platform—where farmers only receive information when manually opening the dashboard—into a **proactive agricultural companion**. AgriGuardian continuously monitors weather forecasts, mandi price dynamics, agronomic milestones, and human agronomist reviews, autonomously dispatching contextual, bilingual, and actionable alerts to farmers.

### Key Achievements:
1. **Phase 8 Verification & Security Hardening Gate:**
   - Completed independent audit of Phase 8 Human-in-the-Loop Expert Verification platform.
   - Fixed object-level authorization bypasses across expert dossier endpoints, case badges, and reviews.
   - Replaced fabricated static ICAR facts with genuine queries against `FactRepository` filtered by crop.
   - Enforced atomic review case claiming (`assigned_expert_id IS NULL AND status = 'queued'`) to prevent race conditions.
   - Fully documented findings and fixes in [`docs/PHASE_8_VERIFICATION.md`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_8_VERIFICATION.md).
2. **Centralized Notification Engine & Bilingual Template System:**
   - Standardized `NotificationEvent` bus connecting Weather (Phase 4), Market (Phase 5), Farm Context & Crop Cycles (Phase 6), and Human Expert Reviews (Phase 8).
   - Pre-seeded and dynamic template engine supporting bilingual (English & Hindi) rendering with agronomic terminology, action links, and variable interpolation.
3. **Intelligent Policy Engine:**
   - Sliding-window deduplication (60m window for market/reminders, 2h for weather hazards) preventing spam.
   - Quiet-hours enforcement (default 22:00–06:00) suppressing external noisy channels while allowing in-app logging.
   - Safety-critical emergency override ensuring severe weather hazards (flash floods, hailstorms) immediately bypass quiet hours and reach farmers over SMS/Voice.
4. **Smart Farm Reminders & Microclimate Conflict Analysis:**
   - Automated generation of agronomic task reminders from Phase 6 active crop cycles and calendar milestones.
   - Microclimate conflict evaluation: flags scheduled chemical spraying or fertilization when rain is forecast (e.g. `🌧️ Rain predicted on Tuesday - postpone spray!`), preventing agrochemical runoff and cost waste.
   - Interactive Snooze (+1d, +3d, custom date) and Completion tracking.
5. **Transactional Outbox with Atomic Worker Leasing:**
   - High-reliability database outbox pattern ensuring **Zero Message Loss**.
   - Atomic worker lease acquisition (`lease_token`, `lease_expires_at`) preventing double-delivery across distributed workers.
   - Automatic reclamation of expired leases from crashed or timed-out workers.
   - Exponential backoff retry strategy (`2^retry_count * 30s`) routing persistent failures (>3 retries) to a Dead-Letter Queue (DLQ).
6. **Multi-Channel Provider Architecture & Truthful Telemetry:**
   - Modular adapters for In-App Center, SMS, WhatsApp Business, Voice Calls (Gnani Timbre TTS), and Email.
   - Truth-first telemetry: gateways without live credentials explicitly report `not_configured` or `0` rather than fabricating delivery confirmations.
7. **DPDP Act Compliant Consent Tracking:**
   - Explicit channel consent ledger (`notification_channel_consents`) tracking opt-ins, revocations, timestamps, and consent sources per the Digital Personal Data Protection Act.
8. **End-to-End User Experience:**
   - Built [`frontend/views/notifications_view.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/frontend/views/notifications_view.py) featuring 4 responsive tabs: Notification Inbox, Smart Reminders, Preferences & Consents, and Admin Outbox Telemetry.
   - Integrated into [`app.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/app.py) with unread notification badge counters and live service indicators.

---

## 2. Architecture & System Flow

```mermaid
flowchart TD
    subgraph Sources ["Upstream Trigger Sources"]
        W[Phase 4: Weather Hazard]
        M[Phase 5: Market Price Surge]
        C[Phase 6: Crop Cycle Milestones]
        E[Phase 8: Agronomist Review]
    end

    subgraph Core ["Phase 9 Notification Core"]
        EB[NotificationEventBus]
        PE[NotificationPolicyEngine]
        TE[NotificationTemplateService]
        RS[ReminderService]
    end

    subgraph Storage ["SQLite Relational Storage (farm_guardian.db)"]
        T19[(notification_events)]
        T20[(notifications)]
        T21[(notification_preferences)]
        T22[(notification_channel_consents)]
        T23[(notification_outbox)]
        T24[(notification_delivery_attempts)]
        T25[(notification_templates)]
        T26[(farm_reminders)]
    end

    subgraph OutboxWorker ["Transactional Outbox Worker"]
        OB[NotificationOutboxService]
        PR[NotificationProviderRegistry]
    end

    subgraph Channels ["Channel Gateways"]
        CH1[In-App Notification Center]
        CH2[SMS Text Gateway]
        CH3[WhatsApp Business API]
        CH4[Voice TTS: Gnani Timbre]
        CH5[Email Dispatcher]
    end

    W --> EB
    M --> EB
    C --> RS
    E --> EB

    RS --> EB
    EB --> T19
    EB --> PE
    PE -->|Check Dedup & Quiet Hours| T21
    PE -->|Verify Consent| T22
    EB --> TE
    TE --> T25

    EB -->|Persist Inbox| T20
    EB -->|Enqueue Jobs| T23

    OB -->|Atomic Lease Claim| T23
    OB --> PR
    PR --> CH1
    PR --> CH2
    PR --> CH3
    PR --> CH4
    PR --> CH5
    PR -->|Log Attempts| T24
```

---

## 3. Database Schema Extensions (Tables 19–26)

The relational SQLite repository [`backend/app/repositories/farm_db.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/repositories/farm_db.py) was expanded with 8 production tables and indexes:

| Table Name | Description | Key Constraints & Indexes |
|---|---|---|
| `notification_events` | Immutable log of all generated raw events | `event_id` (PK), index on `(dedup_key, created_at)` |
| `notifications` | Farmer in-app notification inbox | `notification_id` (PK), FK to `farmer_profiles(farmer_id)`, index on `(farmer_id, status)` |
| `notification_preferences` | Farmer channels, quiet hours, and categories | `farmer_id` (PK, FK to `farmer_profiles`) |
| `notification_channel_consents` | DPDP-compliant channel opt-in/opt-out ledger | Composite PK `(farmer_id, channel)`, FK to `farmer_profiles` |
| `notification_outbox` | Guaranteed-delivery transactional outbox | `outbox_id` (PK), indexes on `(status, next_attempt_at)`, `(lease_token, lease_expires_at)` |
| `notification_delivery_attempts` | Physical provider delivery telemetry log | `attempt_id` (PK), FK to `notification_outbox(outbox_id)` |
| `notification_templates` | Multilingual message templates | `template_id` (PK), index on `(event_type, language)` |
| `farm_reminders` | Smart farm activity calendar with weather flags | `reminder_id` (PK), FKs to `farmer_profiles`, `farms`, `crop_cycles` |

### Default Pre-Seeded Templates:
Eight production bilingual templates are seeded upon database initialization:
1. `weather_alert_extreme` (Flash floods, severe storms, hailstorms)
2. `weather_warning_moderate` (Unseasonal rains, temperature anomalies)
3. `market_price_surge` (Favorable price spike at local APMC mandis)
4. `market_price_drop` (Price collapse warning with storage recommendation)
5. `expert_review_approved` (Agronomist verified diagnostic & ICAR treatment)
6. `expert_review_rejected` (Advisory flagged with corrective prescription)
7. `crop_activity_reminder` (Agronomic task scheduled from crop cycle)
8. `weather_spray_conflict` (Proactive rain warning during pesticide/fertilizer window)

---

## 4. Smart Reminders & Microclimate Conflict Analysis

In [`backend/app/services/notifications/reminder_service.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/services/notifications/reminder_service.py), the `ReminderService` evaluates scheduled activities against meteorological risk conditions:
- **Chemical Runoff Prevention:** If `activity_type == PEST_SPRAY` or `FERTILIZATION`, the system queries the farm's microclimate forecast.
- **Conflict Trigger:** If rain $\ge 2.0\text{ mm}$ or high wind speeds ($\ge 25\text{ km/h}$) are forecasted on or adjacent to the scheduled activity date:
  - `has_weather_conflict = True`
  - Bilingual warning attached:
    - *English:* *"Rain forecast (4.5mm) during scheduled spray. Risk of chemical runoff and reduced efficacy. Postpone application to clear weather."*
    - *Hindi:* *"निर्धारित छिड़काव के दौरान बारिश का पूर्वानुमान (4.5mm)। कीटनाशक धुलने का खतरा। कृपया मौसम साफ होने तक छिड़काव टालें।"*
- **Interactive Lifecycle:** Farmers can mark tasks as completed (`complete_farm_reminder`), snooze by 1 day, 3 days, 1 week, or pick a custom date (`snooze_farm_reminder`). Re-evaluates weather risks upon rescheduling.

---

## 5. Transactional Outbox & Worker Lease Recovery

To guarantee **At-Least-Once Delivery** without double-sending or losing messages:
1. **Atomic Lease Claiming:**
   ```sql
   UPDATE notification_outbox
   SET status = 'processing',
       lease_token = ?,
       lease_expires_at = ?
   WHERE outbox_id IN (
       SELECT outbox_id FROM notification_outbox
       WHERE (status = 'pending' OR (status = 'failed' AND next_attempt_at <= ?))
       LIMIT ?
   )
   ```
2. **Expired Lease Worker Recovery:**
   If a worker crashes while processing messages, any lease where `lease_expires_at < now` is automatically reset to `status = 'pending'`, allowing surviving workers to immediately recover the work.
3. **Exponential Backoff & Dead-Letter Routing:**
   - Attempt 1: Backoff 30 seconds
   - Attempt 2: Backoff 60 seconds
   - Attempt 3: Backoff 120 seconds
   - Failure beyond max retries (3): `status = 'dead_letter'`
4. **Manual DLQ Replay:** Administrators can inspect dead-letter payloads, diagnose provider errors, and trigger manual replay with `retry_dead_letter(outbox_id)`.

---

## 6. Channel Provider Implementations & Truth-First Telemetry

Channel providers are implemented in [`backend/app/integrations/notifications/`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/integrations/notifications/):
- **In-App Center Provider (`in_app_provider.py`):** Always configured, zero-latency local dispatch directly to farmer inbox table.
- **Voice TTS Alert Provider (`voice_provider.py`):** Integrates directly with **Gnani Timbre TTS** (`GNANI_API_KEY`). Generates natural spoken audio alerts for farmers with voice-primary literacy preferences.
- **SMS Gateway (`sms_provider.py`):** Supports standard Indian SMS aggregators (CDAC / NIC / Twilio / MSG91).
- **WhatsApp Provider (`whatsapp_provider.py`):** Formats rich interactive notification cards.
- **Email Provider (`email_provider.py`):** Formats HTML advisory digests.
- **Zero Fabrication Principle:** If a third-party gateway lacks live API credentials, it reports `configured: False` and logs physical delivery as `not_configured`. Delivery counts are never fabricated.

---

## 7. Verification & Automated Test Results

The platform was thoroughly tested with 19 dedicated Phase 9 automated tests and verified against the entire legacy test suite:

### Notifications Test Suite (`tests/notifications/`):
- `test_notification_event_bus.py`:
  - `test_weather_alert_event_dispatched_and_rendered`: **PASSED**
  - `test_market_price_surge_event_generation`: **PASSED**
  - `test_duplicate_events_suppressed_by_dedup_key`: **PASSED**
  - `test_expert_review_event_renders_verified_provenance`: **PASSED**
- `test_notification_policy_engine.py`:
  - `test_quiet_hours_blocks_non_emergency_external_channels`: **PASSED**
  - `test_emergency_weather_overrides_quiet_hours`: **PASSED**
  - `test_farmer_category_opt_out_suppresses_event`: **PASSED**
- `test_notification_outbox_and_worker.py`:
  - `test_outbox_atomic_lease_claiming_prevents_double_processing`: **PASSED**
  - `test_expired_lease_worker_recovery`: **PASSED**
  - `test_repeated_failure_routes_to_dead_letter_queue`: **PASSED**
  - `test_manual_retry_from_dead_letter`: **PASSED**
- `test_smart_reminders_and_weather_risk.py`:
  - `test_custom_reminder_lifecycle`: **PASSED**
  - `test_auto_generate_reminders_from_crop_cycles`: **PASSED**
  - `test_weather_spray_conflict_flagging`: **PASSED**
- `test_notification_rbac_and_isolation.py`:
  - `test_farmer_a_cannot_view_farmer_b_notifications`: **PASSED**
  - `test_farmer_a_cannot_mark_or_dismiss_farmer_b_notification`: **PASSED**
  - `test_farmer_a_cannot_snooze_or_complete_farmer_b_reminder`: **PASSED**
  - `test_farmer_cannot_access_admin_outbox_queue`: **PASSED**
  - `test_admin_outbox_process_and_telemetry`: **PASSED**

### Overall Test Suite Execution:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-8.2.2, pluggy-1.6.0
collected 242 items

240 passed, 2 skipped in 49.35s (100% pass rate)
```

---

## 8. Conclusion

AgriGuardian Phase 9 is fully operational, thoroughly tested, and rigorously secured. The platform bridges meteorological intelligence, market data, and expert recommendations directly into the farmer's daily workflow with zero data loss and proactive risk prevention.

