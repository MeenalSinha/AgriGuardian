# AGRIGUARDIAN — PHASE 7 PRE-FLIGHT VERIFICATION & SECURITY AUDIT

**Audit Date:** November 2026  
**Auditor:** AgriGuardian Quality Assurance, AI Safety & Security Architecture Team  
**System Under Audit:** Phase 7 Role-Based Dashboards, Field Officer Intelligence & Platform Analytics  
**Status:** REMEDIATION REQUIRED PRIOR TO PHASE 8  

---

## 1. EXECUTIVE SUMMARY & VERIFICATION AUDIT MATRIX

Before initiating Phase 8 (Agricultural Expert Verification and Human-in-the-Loop AI), a comprehensive pre-flight audit of Phase 7 was conducted. While core relational migrations and role-based test cases were successfully implemented, the audit discovered several critical security vulnerabilities, fabricated operational telemetry, and insecure UI fallbacks that must be remediated immediately.

| Component / Subsystem | Item Inspected | Pre-Flight Finding | Audit Status |
| :--- | :--- | :--- | :---: |
| **Authentication & RBAC** | Public Self-Registration | `UserRegistrationRequest` permits user-supplied `role`, and `AuthService.register_farmer()` persisted it directly, allowing self-escalation to `ADMINISTRATOR` or `FIELD_OFFICER`. | **FAIL (Critical)** |
| **Authentication Security** | Cryptographic Signing Key | Hardcoded default signing key (`agri-guardian-secure-phase6-secret-key-3848201`) present in `config.py` and `auth_service.py`. | **FAIL (Critical)** |
| **Platform Telemetry** | Administrator Operational Metrics | `platform_analytics.py` contained hardcoded `99.95%` uptime, `0.04%` error rate, `96.2%` data quality, and artificial usage additions (`max(total_cases, 12)`, `+ 18`, `* 3`). | **FAIL (Critical)** |
| **Farmer Dashboard UI** | Anonymous Farmer Fallback | `dashboard_view.py` queried the first farmer in `farmer_profiles` when unauthenticated, leaking private farmer data to anonymous visitors. | **FAIL (Critical)** |
| **Field Officer UI** | Demo Identity & Auto-Assignment | `dashboard_view.py` defaulted unauthenticated users to `usr_officer_demo` and automatically self-assigned Karnal and Meerut territories. | **FAIL (Critical)** |
| **Field Officer UI** | Unchecked Admin Privilege Flag | Follow-up modal in `dashboard_view.py` passed `is_admin=True` unconditionally, bypassing server territory restrictions. | **FAIL (Critical)** |
| **Field Officer UI** | Treatment Prescription Boundary | Follow-up form included free-text pesticide dosage prescription, conflating field observations with expert medical/agronomic authorization. | **FAIL (Critical)** |
| **Frontend Navigation** | Unrestricted Role Selector | Role dropdown in `dashboard_view.py` allowed any user to switch between Farmer, Field Officer, and Administrator views regardless of session credentials. | **FAIL (Critical)** |
| **Relational Schema** | Advisory Case Relational Core | `advisory_cases`, `field_officer_assignments`, `case_follow_ups` correctly created with UUIDs, foreign keys, and indexes in SQLite. | **PASS** |
| **API Endpoints** | FastAPI RBAC Endpoints | Endpoints in `main.py` enforce `require_roles` dependencies with proper 401/403 status codes. | **PASS** |
| **Privacy Protection** | Small-Cohort Anonymization | Village clusters with $k < 3$ cases successfully mask individual farmer names as `"Confidential Farmer"`. | **PASS** |
| **Test Suite** | Automated Tests | 23 analytics tests and 174 regression tests pass (197 passed, 2 skipped). | **PASS** |

---

## 2. DETAILED FINDINGS & REMEDIATION PLAN

### Finding 1: Public Self-Registration Privilege Escalation (CRITICAL)
- **Problem:** `UserRegistrationRequest` accepts a client-provided `role` field. An attacker can register an account with `role="administrator"` or `role="field_officer"`.
- **Remediation:**
  1. Strictly enforce that `AuthService.register_farmer()` forces `role = UserRole.FARMER`.
  2. Create dedicated administrative provisioning methods/endpoints for privileged roles (`ADMINISTRATOR`, `FIELD_OFFICER`, and Phase 8 `AGRONOMIST`/Expert).
  3. Reject or overwrite any client-supplied privileged role in public registration.

### Finding 2: Hardcoded Production Signing Key (CRITICAL)
- **Problem:** `backend/app/core/config.py` and `auth_service.py` fall back to `"agri-guardian-secure-phase6-secret-key-3848201"`. Anyone with codebase access could forge valid JWT session tokens.
- **Remediation:**
  1. Remove the static key string.
  2. Read `JWT_SECRET_KEY` from environment variables.
  3. If unconfigured in local development/test, generate a secure dynamic 32-byte secret via `secrets.token_hex(32)`.

### Finding 3: Fabricated Administrator Operational Telemetry (CRITICAL)
- **Problem:** `platform_analytics.py` hardcoded `99.95%` uptime, `0.04%` error rate, and calculated artificial feature usage (`max(total_cases, 12)`, `total_cases + 18`, etc.).
- **Remediation:**
  1. Replace `admin_system_uptime` with true live health check ratio: $\frac{N_{\text{active}}}{N_{\text{probes}}} \times 100\%$ across the 5 verified dependency health checks.
  2. Mark unmeasured telemetry (`admin_api_error_rate`) with `is_available=False` and explicit reason: `"Continuous API error rate monitoring requires OpenTelemetry/Prometheus tracing exporter."`
  3. Compute `admin_data_quality_score` directly from verified database records: ratio of complete farmer profiles and verified sources.
  4. Compute feature usage strictly from genuine database counts without artificial additions.

### Finding 4: Insecure Anonymous & Demo Fallbacks in UI (CRITICAL)
- **Problem:** `dashboard_view.py` fell back to `SELECT farmer_id FROM farmer_profiles LIMIT 1`, `usr_officer_demo`, and `adm_system` when unauthenticated.
- **Remediation:**
  1. Remove all fallback queries and demo identifiers.
  2. If unauthenticated, render an informative authentication warning and prompt the user to log in or register.
  3. If an officer has no assigned territory, display an unassigned state rather than automatically provisioning Karnal/Meerut.
  4. Ensure `is_admin` is derived strictly from `current_user.role == UserRole.ADMINISTRATOR`.

### Finding 5: Role Switcher & Role Routing Boundary (CRITICAL)
- **Problem:** Frontend dropdown let any user switch between Farmer, Field Officer, and Admin surfaces.
- **Remediation:**
  1. Restrict view choices to roles authorized for the authenticated user.
  2. Administrators can view all dashboards for system management; farmers and officers only access authorized surfaces.

### Finding 6: Clear Separation of Field Observations vs. Expert Guidance (CRITICAL)
- **Problem:** Field officer follow-up form contained free-text pesticide dosage inputs. Field officers do not have statutory authority to prescribe chemical treatments.
- **Remediation:**
  1. Re-label field officer follow-up form to "Field Visit Observations & Reported Interventions".
  2. Channel all expert diagnosis approvals and agronomic treatment prescriptions to Phase 8 Agricultural Expert workflows.

---

## 3. REMEDIATION GATE STATUS

Phase 8 implementation is blocked until all 6 critical remediation actions are executed, re-tested, and verified.
