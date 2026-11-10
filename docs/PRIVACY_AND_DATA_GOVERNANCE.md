# AGRIGUARDIAN — PRIVACY & DATA GOVERNANCE POLICY
**Compliance Matrix with India's Digital Personal Data Protection (DPDP) Act 2023**  
**Author:** AgriGuardian Privacy & Governance Office  
**Date:** November 10, 2026  
**Status:** Approved & Enforced in Code  

---

## 1. Regulatory Context & Framework

AgriGuardian processes digital agricultural, financial, geospatial, and voice data of Indian farmers. Under the **Digital Personal Data Protection (DPDP) Act 2023** and relevant provisions of the **Information Technology Act 2000**, AgriGuardian functions as a **Data Fiduciary**, and the farmer functions as the **Data Principal**.

This document establishes the technical, organizational, and architectural data governance controls implemented across AgriGuardian's database schema, backend APIs, and frontend interfaces.

---

## 2. DPDP Act 2023 Compliance Matrix

| DPDP Act Provision | Legal Obligation | AgriGuardian Technical Implementation | Verification Method |
|---|---|---|---|
| **Section 6(1)** | Lawful, specific, and informed consent | Explicit consent collection specifying processing purpose, policy version, and timestamps recorded in Table 45 (`user_consents`). | `test_consent_recording_and_listing` |
| **Section 6(4)** | Right to withdraw consent easily | Dedicated endpoint `POST /api/v1/privacy/consents/withdraw` updating consent status to `withdrawn` without penalizing unrelated features. | `test_consent_withdrawal` |
| **Section 8(1)** | Purpose limitation & data minimization | Audio recordings processed in ephemeral memory buffers; GPS coordinates truncated to village/district centroids for regional weather/mandi intelligence unless farm boundary precision is explicitly granted. | Source audit in `main.py` & `audio_utils.py` |
| **Section 8(7)** | Right to erasure (deletion of data) | Dedicated endpoint `POST /api/v1/privacy/delete-account` performing irreversible anonymization of PII (name, phone, username, device tokens). | `test_farmer_account_anonymized_deletion` |
| **Section 11(1)** | Right to access & data portability | Dedicated endpoint `GET /api/v1/privacy/export-data` generating structured JSON archive containing farmer profile, farms, crops, expenses, harvests, sales, and consent history. | `test_farmer_full_data_export` |
| **Section 8(5)** | Security safeguards against breaches | Argon2id password hashing, JWT token revocation, ASGI security headers, sensitive data redaction in JSON logs, and encrypted TLS communications. | `test_security_hardening.py` |

---

## 3. Data Classification & Minimization Policy

| Data Category | Data Elements Collected | Justification & Purpose | Retention Period | Storage Location |
|---|---|---|---|---|
| **Personal Identity (PII)** | Full Name, Phone Number, Preferred Language, Username | Farmer account creation, personalized multilingual TTS/STT, and SMS/WhatsApp notifications. | Retained until account deletion; anonymized upon erasure request. | `farmer_profiles`, `users` |
| **Geospatial & Farm Profile** | State, District, Village, Coordinates, Farm/Plot Area | Localized agricultural weather forecasts from Open-Meteo, mandi price alerts, and irrigation advisory. | Retained while farm active; anonymized on deletion. | `farms`, `plots`, `soil_profiles` |
| **Crop Diagnostic Media** | Leaf images, visual observations, confidence scores | MobileNetV3 disease inference, ICAR advisory recommendations, and expert agronomist validation queue. | 3 years for seasonal trend analytics; stripped of PII upon erasure. | `advisory_cases`, `sync_attachments` |
| **Voice & Speech Telemetry** | Temporary audio waveform files (.wav) | Transcribed via Gnani AI STT to query agricultural models. | **Ephemeral:** Deleted immediately after inference completed (<60s). | In-memory / `/tmp/` transient |
| **Financial & Economic Ledgers** | Production expenses, harvest yields, mandi produce sale prices | Cultivation cost calculation, profitability intelligence, and yield break-even analytics. | 7 years (statutory compliance with Indian commercial records regulations). | `farm_expenses`, `farm_harvests`, `farm_sales` |

---

## 4. Consent Architecture & Lifecycle

### 4.1 Consent Types Catalog

Consents are tracked via enum `ConsentType` in [`backend/app/schemas/privacy.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/schemas/privacy.py):
1. `location`: Localized Open-Meteo weather and APMC mandi market price discovery.
2. `voice_recording`: Speech audio processing via Gnani AI Timbre/Prisma models.
3. `crop_diagnostics`: Storage and agronomic analysis of plant pathology images.
4. `expert_sharing`: Forwarding diagnostic dossier to authorized agronomists for secondary human review.
5. `marketing_notifications`: Proactive promotional notifications (strictly opt-in, disabled by default).

### 4.2 Consent Record Schema (Table 45 `user_consents`)

```sql
CREATE TABLE IF NOT EXISTS user_consents (
    consent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
    consent_type TEXT NOT NULL,
    status TEXT NOT NULL,          -- 'granted', 'withdrawn', 'declined'
    policy_version TEXT NOT NULL,  -- e.g. 'v1.2'
    purpose TEXT NOT NULL,
    ip_address TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### 4.3 Consent APIs

- `GET /api/v1/privacy/consents`: Returns all active and historical consent records for authenticated farmer.
- `POST /api/v1/privacy/consents`: Grant or update consent for a purpose.
- `POST /api/v1/privacy/consents/withdraw`: Withdraw consent. Immediately halts data processing for that purpose.

---

## 5. Data Subject Rights (Portability & Erasure)

### 5.1 Data Portability Export (`GET /api/v1/privacy/export-data`)

The farmer can download their complete digital record at any time in machine-readable JSON format conforming to DPDP Act Section 11(1). The export bundle includes:
- Account and profile information
- Registered farms, plots, and irrigation systems
- Historical and active crop cycles
- Cultivation expense records
- Harvest and production quantities
- Market produce sales ledgers
- Active sync devices and cursors
- Full consent audit trail

### 5.2 Right to Erasure & Account Anonymization (`POST /api/v1/privacy/delete-account`)

When a farmer requests account deletion:
1. **PII Scrubbing:**
   - `users.username` replaced with randomized pseudonymous string `deleted_user_<random_hex8>`.
   - `users.password_hash` replaced with sentinel value `'ACCOUNT_DELETED'`.
   - `users.salt` replaced with `'DELETED'`.
   - `farmer_profiles.full_name` replaced with `'Deleted Farmer Account'`.
   - `farmer_profiles.phone_number` set to `NULL`.
2. **Session & Device Revocation:**
   - Active tokens revoked in `revoked_tokens`.
   - Registered mobile/PWA devices marked revoked (`sync_devices.is_revoked = 1, revoked_at = <timestamp>`).
3. **Consent Withdrawal:**
   - All active consents updated to `withdrawn`.
4. **Preservation of Anonymized Ledgers:**
   - Farm expense and sales records are retained without personal identifiers to preserve statutory financial accounting integrity.

---

## 6. Audit Trail & Verification

All privacy operations were verified against unit and integration tests in [`tests/security/test_privacy_and_data_rights.py`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/tests/security/test_privacy_and_data_rights.py):
- `test_consent_recording_and_listing`: PASSED (0.25s)
- `test_consent_withdrawal`: PASSED (0.24s)
- `test_farmer_full_data_export`: PASSED (0.25s)
- `test_farmer_account_anonymized_deletion`: PASSED (0.27s)

