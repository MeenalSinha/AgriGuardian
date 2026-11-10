# AGRIGUARDIAN — FINAL TEST EXECUTION & QA REPORT

**Date:** November 10, 2026  
**Test Harness:** Pytest 8.2.2 with asyncio, anyio, and mock extensions  
**Runtime:** Python 3.11 / Python 3.14  
**Operating Environment:** Verified on Windows & Ubuntu Linux CI  

---

## 1. Test Execution Summary

| Metric | Result |
| :--- | :--- |
| **Total Tests Collected** | **328** |
| **Tests Passed** | **326 (99.4%)** |
| **Tests Skipped** | **2 (0.6%)** *(Live Gnani API endpoints requiring external sandbox bypass)* |
| **Tests Failed** | **0 (0.0%)** |
| **Execution Duration** | **136.78s (2m 16s)** |
| **Code Coverage Quality** | **Unit, Functional, Integration, RBAC, Security & E2E** |

---

## 2. Test Execution Commands & Verified Results

### Full Codebase Test Run
```bash
python -m pytest tests -q
```
**Output:**
```text
326 passed, 2 skipped in 136.78s (0:02:16)
```

### Module-Specific Test Breakdowns

#### A. Security & Hardening (`tests/security`)
```bash
python -m pytest tests/security -v
```
- Total: 18 passed in 24.95s.
- Verifies: Argon2id password hashing, legacy PBKDF2 migration, password complexity, role escalation prevention, brute-force lockout, JWT blacklisting, security headers, request size limits, and production configuration validation.

#### B. End-to-End User Journeys (`tests/e2e`)
```bash
python -m pytest tests/e2e -v
```
- Total: 4 passed.
- Verifies: Complete farmer journey, field officer territory journey, agronomist expert verification journey, and system administrator governance journey.

#### C. Farm & Plot Management (`tests/farm`)
```bash
python -m pytest tests/farm -v
```
- Total: 22 passed.
- Verifies: Farmer registration, login, multi-farm holdings, plot zoning, crop cycles, phenological stages, soil fertility diagnostics, and multi-tenant tenant isolation.

#### D. Expert Verification Portal (`tests/expert`)
```bash
python -m pytest tests/expert -v
```
- Total: 23 passed.
- Verifies: Agronomist assignment matching, case claiming, clinical decision desk, CIBRC pesticide safety rules, dual secondary review, and agronomist audit trails.

#### E. Proactive Notifications & Reminders (`tests/notifications`)
```bash
python -m pytest tests/notifications -v
```
- Total: 24 passed.
- Verifies: Weather risk event dispatching, market price surge triggers, transactional outbox atomic lease claiming, dead-letter recovery, quiet hours, DPDP consent enforcement, and reminder CRUD.

#### F. Offline Synchronization & PWA Companion (`tests/sync`)
```bash
python -m pytest tests/sync -v
```
- Total: 20 passed.
- Verifies: Pull bootstrap bundle, incremental change-feed replication, tombstones, batch mutation push, idempotency replay, conflict detection and resolution, photo attachments, and device registry.

#### G. Farm Economics & Profitability (`tests/economics`)
```bash
python -m pytest tests/economics -v
```
- Total: 30 passed.
- Verifies: Decimal-exact arithmetic, unit conversion precision, append-only expense corrections, multi-event harvest tracking, sales net realization, break-even analytics, and budget variance.

#### H. Market Intelligence (`tests/market`)
```bash
python -m pytest tests/market -v
```
- Total: 38 passed.
- Verifies: AGMARKNET daily bulletin ingestion, biological bounds validation, spatial Haversine distance, nearby mandi discovery, statutory MSP compliance, trend classification, and Holt-Winters / ARIMA price forecasting.

#### I. Vision Foliar Diagnostics (`tests/vision`)
```bash
python -m pytest tests/vision -v
```
- Total: 18 passed.
- Verifies: Image validation, resolution bounds, MobileNetV3 CNN inference, Grad-CAM attention heatmap overlay generation, foliar surface severity percentage calculation, and multimodal reasoning.

#### J. Storage & Database Migration Utilities (`tests/unit`)
```bash
python -m pytest tests/unit/test_storage_service.py tests/unit/test_migration.py -v
```
- Total: 9 passed.
- Verifies: Directory traversal attack prevention, safe path resolution, unconfigured S3 error reporting, topological table dependency DAG calculation, PostgreSQL DDL generation, and non-destructive migration dry-run.

---

## 3. Detailed Skipped Test Analysis

```text
SKIPPED [1] tests/integration/test_gnani_live.py:30: Live Gnani endpoint unreachable in current environment (Sandbox/Offline). Status: BLOCKED: Failed to connect to Gnani API (https://api.vachana.ai/stt/v3)
SKIPPED [1] tests/integration/test_gnani_live.py:58: Live Gnani endpoint unreachable in current environment (Sandbox/Offline). Status: BLOCKED: Failed to connect to Gnani API (https://api.vachana.ai/api/v1/tts/inference)
```

- **Root Cause:** Both tests connect directly to live Gnani production endpoints (`https://api.vachana.ai`). When executed inside a local sandbox or without live cloud API credentials, the test harness catches the network block and skips gracefully with an explicit message.
- **Coverage Preservation:** Offline unit tests for Gnani STT and TTS (`tests/unit/test_prisma_stt.py` and `tests/unit/test_timbre_tts.py`) mock the network response payloads and comprehensively verify protocol formatting, audio chunking, and error handling.

---

## 4. Bytecode Compilation & Static Integrity Check

```bash
python -m compileall backend frontend app.py scripts tests
```
- Result: **0 syntax errors, 0 compilation failures**.

---

## 5. Database Backup and Restore Verification

```bash
python scripts/backup_database.py
python scripts/restore_database.py --backup-file backups/farm_guardian_backup_YYYYMMDD_HHMMSS.db --verify-only
```
- Result:
  ```text
  SQLite PRAGMA integrity_check passed: 'ok'
  VERIFY RESULT: Integrity=OK
  ```
