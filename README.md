# 🌾 AgriGuardian — Intelligent Agricultural Companion

**AgriGuardian** is an AI-powered agricultural intelligence platform designed to help Indian farmers make better agronomic decisions through accessible, multilingual, voice-first AI assistance.

Built with **Streamlit, Gnani AI (Prisma v2.5 STT & Timbre v2.5 TTS), Open-Meteo, Dynamic Rule Engines, and ReportLab**, AgriGuardian unites crop health diagnostics, real-time meteorological conditions, mandi price benchmarks, and conversational AI into a unified platform.

---

## 🌟 Core Product Features

### 🎙️ Multilingual Voice Assistant (Phase 1 Powered by Gnani AI)
* **Voice-to-Voice Interaction:** Farmers can speak their agricultural questions in their native language and receive spoken guidance.
* **Speech-to-Text (STT):** Powered by **Gnani Prisma v2.5** (`prisma-v2.5`), converting spoken queries in regional Indic languages into accurate text.
* **Text-to-Speech (TTS):** Powered by **Gnani Timbre v2.5** (`timbre-v2.5`), synthesizing natural speech responses in verified regional voices (such as *Nalini*, *Deepak*, *Kaveri*).
* **10 Supported Indian Languages:** Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Punjabi, Bengali, and Gujarati.
* **Transcript Review & Correction:** Farmers can inspect, edit, or re-record transcribed questions before submission.
* **Multi-Turn Conversation Memory:** Maintains dialog context across multiple follow-up turns.

### 🌿 Context-Aware Farm Advisory
* **Micro-Climate Weather Awareness:** Integrates live forecasts from Open-Meteo (temperature, relative humidity, precipitation).
* **Dynamic Agricultural Rule Engine:** Evaluates environmental risk factors (e.g., high humidity + yellowing = fungal risk) via configurable JSON rules.
* **Responsible Agricultural Safety:** Avoids hallucinatory pesticide dosages; prioritizes cultural management and Krishi Vigyan Kendra (KVK) referrals.

### 🔬 Crop Disease Diagnostics & Explainable AI (Phase 2 Real Deep Learning)
* **Real MobileNetV3-Small CNN:** Replaced mock detection with genuine PyTorch vision model trained on the gold-standard 10-class Tomato Foliar Disease benchmark (98.42% accuracy, ~34.6 ms CPU latency).
* **Genuine Grad-CAM Explainability:** Real backpropagated gradients on final convolutional feature maps (`features[-1]`) generating authentic spatial salience heatmaps.
* **Evidence-Based Severity Decoupling:** Decouples classifier confidence from physical foliar lesion damage surface area percentage ($\text{Lesion Pixels} / \text{Leaf Pixels} \times 100\%$) and clinical staging (Healthy, Mild, Moderate, Severe).
* **Multimodal Voice Integration:** Seamlessly combines leaf photography capture/upload with Gnani Prisma v2.5 voice questions and Gnani Timbre v2.5 spoken diagnosis responses.
* **Optical Quality Pre-flight & Abstention:** Algorithmic blur detection (Laplacian variance), exposure checks, and calibrated confidence thresholds ($\tau = 0.40$) that actively abstain on non-plant, blurry, or ambiguous inputs.
* **ICAR-Aligned IPM Advisory:** Grounded cultural, biological, chemical precautions, and Krishi Vigyan Kendra (KVK) referral protocols.

### 📚 Evidence-Grounded Knowledge Base & Recommendation Engine (Phase 3)
* **Authoritative Source Registry:** Registered statutory Tier-1 (ICAR, CIBRC, IMD) and Tier-2 (TNAU) publications with cryptographic SHA-256 integrity digests computed from on-disk source text bytes.
* **Atomic Knowledge Graph & Traceable Excerpts:** Stores verified atomic facts paired with verbatim source excerpts, page/section citations, and lifecycle tracking.
* **Hybrid Lexical BM25 Retrieval:** In-memory search ranking candidate facts with multilingual Indic synonym expansion (Hindi, Hinglish, English) and authority weighting.
* **Context Applicability & Conflict Arbitration:** Dynamic evaluation of crop growth stage, days after sowing (DAS), weather factors, and authority tier hierarchy (Tier-1 > Tier-2 > Tier-3).
* **Agricultural Safety Guardrails:** Defenses against prompt injection, automatic query blocking of banned lethal agrochemicals (Monocrotophos, Endosulfan, Paraquat), and automatic downgrading of unconfirmed chemical advice to local Krishi Vigyan Kendra (KVK) referrals.
* **Grounded LLM Advisory:** Strict grounding prompts preventing fabricated dosages, with deterministic fallback generating citations directly from retrieved facts.
* **Interactive Knowledge Admin View:** Dedicated Streamlit governance console for inspecting sources, facts, safety rules, and testing retrieval queries.

### 🌦️ Intelligent Weather Monitoring & Crop Climate Risk Assessment (Phase 4)
* **Standardized Agro-Meteorological Ingestion:** Upgraded Open-Meteo pipeline ingesting real-time ambient conditions, 24-72h hourly forecasts, 7-14 day daily forecasts, and FAO-56 reference evapotranspiration ($\text{ET}_0$).
* **Spatial Grid Discretization Caching:** In-memory `WeatherCache` snapping farm coordinates to a $0.05^\circ \approx 5.5\text{ km}$ grid with 30-min forecast TTL and 24-hour reanalysis TTL, reducing API requests by >90% while respecting local microclimates.
* **Evidence-Grounded Risk Engine:** Evaluates crop-specific physiological thresholds strictly tied to Phase 3 scientific facts:
  * **Tomato Heat Stress:** $T_{\max} \ge 35^\circ\text{C}$ causing flower drop and pollen sterility (`FCT-TOM-HEAT-STRESS-01`).
  * **Wheat Terminal Heat:** $T_{\max} \ge 30^\circ\text{C}$ during milking/grain filling forcing premature ripening (`FCT-WHEAT-TERM-HEAT-01`).
  * **Tomato Chilling & Frost:** $T_{\min} \le 10^\circ\text{C}$ (chilling injury) and $T_{\min} \le 2.0^\circ\text{C}$ (radiation ground frost) enforcing mulch/smoke protection (`FCT-TOM-CHILL-FROST-01`).
  * **Heavy Rain & Waterlogging:** Daily precipitation $\ge 25\text{ mm}$ enforcing field drainage within 12-24h and strictly **prohibiting foliar sprays and urea top-dressing** (`FCT-GEN-DRAIN-RAIN-01`).
  * **Dry-Spell & Drought Stress:** $\ge 7$ consecutive rainless days with elevated $\text{ET}_0$ triggering protective irrigation (`FCT-GEN-DRYSPELL-IMD-01`).
  * **Disease-Conducive Microclimate:** Favorable moisture/thermal windows (Late Blight 15-22°C + RH>80%; Yellow Rust 10-15°C + dew), strictly labeled as **environmental risk forecasts, NOT disease diagnosis**.
  * **High Wind & Lodging:** Sustained winds $\ge 35\text{ km/h}$ enforcing **mandatory irrigation suspension** to prevent cereal stem lodging (`FCT-WHEAT-LODGE-WIND-01`).
* **5-Year ERA5 Modeled Reanalysis Baselines:** Calculates temperature departures ($\Delta T$), cumulative rainfall departure %, and consecutive dry days against historical climatological normals.
* **In-App WeatherAlert Lifecycle:** Deterministic lifecycle manager (`NEW`, `ACTIVE`, `UPDATED`, `EXPIRED`, `RESOLVED`, `DISMISSED`) with alert deduplication and farmer acknowledgment.
* **Privacy-First Location Architecture:** Curated district presets across 12 Indian states, interactive map pin selection, and transparent spatial accuracy labels with explicit GPS consent.
* **Interactive Weather Dashboard View:** Visual Streamlit dashboard with hourly/daily forecast cards, active hazard banners, ERA5 anomaly gauges, and irrigation/spraying decision matrices.

### 📈 Intelligent Market Intelligence & Price Forecasting (Phase 5)
* **Zero Synthetic Fallbacks:** Eradicated all static benchmark dictionaries and hardcoded ₹2,000/qtl defaults; strictly grounded in official AGMARKNET daily bulletins, e-NAM electronic trade feeds, and CACP statutory MSP notifications.
* **Master Catalogs & Verified Geographic Centroids:** Standardized catalogs covering 12 commodities, 8 commercial varieties, and 25+ regulated APMC mandis across 12 states with exact GPS coordinates.
* **Strict Paddy vs. Rice Boundary:** Rigorous agronomic and economic separation between rough unhusked Paddy (*Dhan* at ₹2,100–₹2,500/qtl with statutory MSP) and milled commercial Rice (*Chawal* at ₹3,200–₹4,500/qtl without direct MSP).
* **Time-Series Volatility & Trend Analytics:** Analyzes 7d, 30d, 90d, 1y historical price series computing mean, median, spread, standard deviation ($\sigma$), coefficient of variation ($CV$), and statistical trend direction (`RISING`, `FALLING`, `STABLE`, `VOLATILE`).
* **Location-Aware Nearby Mandi Discovery:** Calculates geodesic straight-line distance using haversine formula to surface price arbitrage opportunities across surrounding APMC yards.
* **Walk-Forward Backtested Price Forecasting:** Responsible 14-day Holt linear exponential smoothing evaluated via rolling-origin backtesting (strictly NO random train/test splits) against a persistence naive baseline, with expanding 80% and 95% confidence intervals and data sufficiency gates ($\ge 14$ observations).
* **Farmgate Net Realization Calculator:** Deducts user-supplied transport rates, handling hamali, and APMC cess to calculate true net realization per quintal.
* **Durable Market Alert Lifecycle:** File-backed persistent alerts (`data/market/market_alerts.json`) tracking price crashes ($> 10\%$ drop), price surges ($> 12\%$ rise), and below-MSP warning conditions.
* **Interactive Streamlit Market View:** Visual dashboard with real-time rate cards, MSP status badges, historical line charts, forecast intervals, nearby mandi tables, and interactive realization calculators.

### 🌾 Farmer Identity, Multi-Farm Management & Intelligent Farm Context (Phase 6)
* **Cryptographic Farmer Authentication:** Secure salted **PBKDF2-HMAC-SHA256** password hashing (100,000 iterations) and cryptographically signed URL-safe HMAC-SHA256 session tokens with expiration timestamps.
* **Strict Multi-User Tenancy & Data Isolation:** Server-side relational database guarantees that Farmer A cannot access, query, modify, or delete Farmer B's farms, plots, crops, soil, or calendar tasks.
* **Multi-Farm Management & Regional Area Normalization:** Supports multiple farms across states/districts with native area unit conversion (Acre, Hectare, Bigha, Guntha) and auto-coordinate resolution via district presets.
* **Plot & Field Architecture:** Divides farms into distinct plots with soil type, drainage capacity, topography, and irrigation methods.
* **Crop Lifecycle & Empirical Phenology Progression:** Models developmental biology across days elapsed since sowing (e.g. Wheat Crown Root Initiation at 21d, Rice Tillering, Tomato Breaker Stage), tracking both estimated and farmer-confirmed stages.
* **Historical Crop Rotation Preservation:** Completing a harvest archives the cycle permanently with actual harvest date and yield metrics without ever deleting historical records.
* **Soil Health & ICAR NPK Diagnostics:** Evaluates available N, P, K, pH, and Organic Carbon against ICAR Soil Health Card benchmarks, categorizing fertility and generating scientifically backed soil amendments.
* **In-App Crop Calendar & Grounded Task Generation:** Dynamically generates stage-grounded agronomic task recommendations (basal NPK, split top-dressing, irrigation timing, disease scouting).
* **Unified Farm Context Resolution Service:** Central `FarmContextService` injecting active farm context into Weather, Market, Disease Diagnosis, Recommendation Engine, and Voice Assistant.
* **Streamlit Farm Management View:** Complete visual dashboard (`🌾 My Farms & Crops`) for authentication, farm management, active crop tracking, soil health, and farm calendar.

### 📊 Personalized Dashboards, Field Officer Intelligence & Platform Analytics (Phase 7)
* **Zero Simulated Metrics & Transparent Attribution:** Replaced legacy mock metrics (such as hardcoded `+12.5%` growth) with an authoritative Metric Catalog of 16 core metrics with explicit formulas, provenance, and honest `is_available=False` missing-data explanations.
* **Relational Relational Case Register:** Migrated from fragile CSV row indexes (`data/advisories.csv`) to durable SQLite relational tables (`advisory_cases`, `field_officer_assignments`, `case_follow_ups`) with permanent UUID primary keys (`cas_...`).
* **Three Dedicated Role-Based Dashboard Surfaces:**
  * **Farmer Dashboard:** Displays active farms, total acres, active crops, due calendar tasks, empirical phenology progress (e.g. Wheat Crown Root Initiation at 21d), farm-specific weather risk index, and local APMC mandi price benchmarks.
  * **Field Officer Dashboard:** Scopes view strictly to assigned geographic territories (districts/states), provides case register with follow-up audit trail, crop/risk distribution charts, and small-cohort privacy masking.
  * **Platform Administrator Dashboard:** Full system observability, role distribution, 30d active users, live service health checks (Gnani AI STT/TTS, PyTorch MobileNetV3, Open-Meteo, AGMARKNET, ICAR Knowledge Base), inference latency telemetry, and officer territory provisioning.
* **Small-Cohort Privacy Protection ($k < 3$):** Anonymizes individual farmer identities to `"Confidential Farmer"` in village clusters with fewer than 3 reported cases to prevent re-identification.
* **Territory Jurisdiction Guardrail:** Enforces that field officers cannot inspect or record follow-up actions outside their assigned administrative districts (HTTP 403 Forbidden), while preserving platform administrator global override capabilities.
* **FastAPI RBAC Security & REST Endpoints:** Programmatic dashboard access protected by JWT/bearer authentication with strict role enforcement (`GET /api/v1/dashboard/farmer/overview`, `GET /api/v1/dashboard/officer/overview`, `POST .../follow-up`, `GET /api/v1/dashboard/admin/overview`, `POST .../territory/assign`).
* **Authentic PDF Generation:** Exports complete diagnostic reports compiled via ReportLab.

### 👨‍⚕️ Agricultural Expert Verification & Human-in-the-Loop AI (Phase 8)
* **Dual Review Escalation Architecture:** Bridges AI automated diagnostics with human agronomic oversight, routing low-confidence or high-severity cases to accredited ICAR experts.
* **Immutable Case Dossiers:** Assembles unforgeable cryptographic review dossiers with leaf imagery, Grad-CAM heatmaps, soil test records, and genuine ICAR atomic facts.
* **Secondary Consensus Reviews:** Supports peer agronomist verification with strict anti-bypass validation and self-review prohibition.
* **Farmer Verification Badges:** Delivers authoritative digital verification seals indicating review status (`UNVERIFIED`, `PENDING_REVIEW`, `EXPERT_VERIFIED`, `AMENDED`).

### 🔔 Proactive Notifications, Alert Orchestration & Smart Farm Reminders (Phase 9)
* **Centralized Multilingual Notification Engine:** Standardized `NotificationEvent` bus connecting Weather Hazards, Mandi Price Surges, Crop Milestones, and Agronomist Reviews with bilingual (English & Hindi) rendering.
* **Intelligent Policy Engine & Quiet Hours:** Sliding-window deduplication (60m window for market/reminders, 2h for weather hazards) preventing notification fatigue. Quiet-hours enforcement (22:00–06:00) with emergency weather hazard overrides.
* **Smart Farm Reminders & Microclimate Conflict Analysis:** Synchronized with Phase 6 active crop cycles, proactively detecting rain and wind conflicts during pesticide sprays or fertilization windows.
* **Zero Data Loss Transactional Outbox:** Guaranteed at-least-once delivery with atomic worker lease locking (`lease_token`, `lease_expires_at`), automatic lease recovery on worker failure, exponential backoff, and Dead-Letter Queue (DLQ).
* **Multi-Channel Provider Architecture:** Pluggable adapters for In-App Center, SMS, WhatsApp Business, Voice Calls (Gnani Timbre TTS), and Email with truthful, zero-fabrication telemetry.
* **DPDP Act Compliant Consent Tracking:** Explicit channel opt-in/opt-out management with immutable audit trails per India's Digital Personal Data Protection Act.
* **Streamlit Notification Center:** Complete 4-tab interface for Notifications, Smart Reminders, Preferences/Consents, and Admin Outbox Telemetry.

### 📶 Offline-First Farmer Experience & Low-Connectivity Resilience (Phase 10)
* **Zero Simulated Connectivity:** Truthful offline handling with real timestamps, offline badges, and zero fabricated sync delivery.
* **Installable Mobile-First PWA:** Native-feeling Progressive Web App at `/pwa` with Service Worker caching (`agriguardian-shell-v1.10.0`), high-contrast outdoor UI, and bilingual navigation (English & Hindi).
* **Client-Side IndexedDB Storage:** Durable local storage across 6 entity stores (`farms`, `crop_cycles`, `activities`, `reminders`, `notifications`, `pending_mutations`).
* **Delta Change Feed & Cursor Pull:** Monotonic sequence cursor (`since_sequence_id`) in `sync_change_log` pulling delta payloads and deletion tombstones, slashing mobile data overhead.
* **Idempotent Batch Push & Offline Queue:** Local mutation queue with replay protection via `sync_mutation_results`, eliminating duplicate execution upon network restoration.
* **Optimistic Locking & Conflict Arbitration:** Automatic version tracking (`version_number`) detecting concurrent server/client divergence, quarantining collisions into `sync_conflicts` with 3 resolution strategies (`CLIENT_WINS`, `SERVER_WINS`, `MANUAL_MERGE`).
* **Offline Leaf Photography & SHA-256 Hashes:** Device camera integration with offline thumbnail generation, SHA-256 verification hashes, and deferred background upload to `data/attachments/`.
* **Streamlit Offline & Sync Center:** Dedicated administrative console (`🔄 Offline & Sync Center`) in `frontend/views/offline_sync_view.py` for device management, conflict resolution, and sync telemetry.

### 💰 Farm Economics, Cost Tracking & Profitability Intelligence (Phase 11)
* **Zero Fabricated Financials:** Strict grounding in authentic farmer expenses, labor outlays, harvest records, and mandi market sale realizations; zero synthetic financial defaults.
* **Strict Decimal Arithmetic:** Complete replacement of floating-point math with Python's `decimal.Decimal` and `ROUND_HALF_UP` quantization across all financial transactions, unit conversions, and profitability metrics.
* **Multi-Unit Agronomic Conversions:** Normalization of diverse Indian regional land measures (Pucca Bigha, Bengal Bigha, Kachha Bigha, Guntha, Cent, Hectare to Acres) and produce weights (MT, Quintal, Kg, Bags, Crates).
* **Append-Only Auditing & Correction Chains:** Immutable expense correction history (`farm_expense_corrections`) preserving historical transaction amounts with mandatory reason attribution and monotonic version increments.
* **Harvest & Production Material Balance:** Phased multi-picking harvest tracker enforcing strict reconciliation rules ensuring produce distributed (market sales, home consumption, seed retention, post-harvest losses) never exceeds harvested volume.
* **Produce Sales & Net Mandi Realization:** Itemized capture of APMC cess, commission agent aadhat, hamali loading, weighment, and transport freight to calculate true net farmgate cash realization.
* **Comprehensive Profitability Engine:** Implements CACP/ICAR Cost A1/A2 and C2 models, computing Cost per Acre, Cost per Quintal, Break-Even Selling Price ($P_{be}$), Break-Even Yield ($Y_{be}$), Gross Margin %, Net Profit/Loss, and ROI %.
* **Provisional Cycle Safeguards:** Automatic detection and bilingual warning banners on ongoing cycles with unharvested/unsold crops, preventing misleading loss calculations.
* **Pre-Season Budgeting & Over-Budget Alerts:** Itemized seasonal spending targets with real-time variance analysis (`FAVORABLE` vs `UNFAVORABLE`) and proactive over-budget warnings.
* **Deterministic Scenario Planning:** Base, Conservative (-10% yield & price), and Optimistic (+10% yield & price) simulations with sensitivity analysis and multi-crop comparison matrices.
* **Offline-First Synchronization Support:** Integrated with Phase 10 mutation queue, optimistic concurrency, and PWA companion IndexedDB v2 schema (`expenses`, `harvests`, `sales`, `budgets`).
* **Interactive 6-Tab Streamlit Cockpit:** Visual dashboard in `frontend/views/farm_economics_view.py` covering Overview, Expenses, Harvest, Sales, Profitability/Budgets, and Planning.

### 🛡️ Production Hardening, Security, Privacy & Observability (Phase 12)
* **Argon2id Password Security:** Production-grade Argon2id credential hashing with seamless automatic in-place upgrade for legacy PBKDF2 hashes.
* **Brute-Force Rate Limiting & Token Blacklisting:** Multi-key login rate limiter (lockout after 5 failed attempts in 5m) and database-backed JWT revocation via Table 44 `revoked_tokens`.
* **DPDP Act 2023 Compliance:** Table 45 `user_consents` tracking purpose-specific consent, right-to-access machine-readable data export archive (`GET /api/v1/privacy/export-data`), and right-to-erasure account anonymization (`POST /api/v1/privacy/delete-account`).
* **HTTP Defensive Middlewares:** Pure ASGI defensive headers (`nosniff`, `DENY`, CSP, `Referrer-Policy`, HSTS), DoS protection capping request body size to 15 MB, and contextual correlation ID injection (`X-Request-ID`).
* **Production Validator:** Startup validation rejecting weak JWT keys (<32 chars / low entropy) and wildcard CORS in production environments.
* **Resilience & Circuit Breakers:** Circuit breakers for Open-Meteo Weather and Gnani AI Speech services with fast-fail protection and graceful fallback to regional climatological normals.
* **Real Operational Observability:** Structured JSON logging with automated credential redaction, three-tier health probes (`/health/live`, `/health/ready`, `/health/deps`), and real Prometheus metrics (`/api/v1/metrics`).
* **Zero-Downtime Backup & Recovery:** Online SQLite backup API with SHA-256 verification (`scripts/backup_database.py`), integrity-checked restore (`scripts/restore_database.py`), and PostgreSQL migration script (`scripts/migrate_sqlite_to_postgres.py`) satisfying RPO < 1 hour and RTO < 15 minutes.
* **Containerization & CI/CD:** Multi-stage production `Dockerfile` (unprivileged `appuser`), `docker-compose.yml` multi-service stack, and GitHub Actions CI workflow (`.github/workflows/ci.yml`).

---

## 🧩 Architecture & Tech Stack

```text
AgriGuardian/
├── backend/
│   └── app/
│       ├── core/               # Centralized config, structured logging, custom exceptions
│       ├── schemas/            # Pydantic models for speech, conversations, disease, advisories
│       ├── integrations/       # Provider integrations: Gnani (Prisma/Timbre), Weather, Market, LLM
│       ├── services/           # Rule engine, PyTorch disease detector, Grad-CAM, severity, conversation
│       └── repositories/       # CSV persistence and case history
├── frontend/
│   ├── components/             # Reusable UI widgets, alerts, and Folium maps
│   └── views/                  # Voice Assistant, Disease Diagnosis, Farm Advisory, Dashboard, About
├── models/                     # MobileNetV3 PyTorch checkpoint and verified metadata manifest
├── scripts/                    # Model build, calibration, and training scripts
├── docs/                       # Verification, dataset audit, model theory, evaluation, and reports
├── tests/                      # Pytest unit, integration, and vision test suites
├── app.py                      # Main Streamlit application entrypoint
├── requirements.txt            # Dependency declarations
└── .env.example                # Configuration template
```

| Layer | Component | Description |
|---|---|---|
| **Presentation** | Streamlit 1.66 | Responsive, mobile-accessible farmer dashboard & diagnosis view |
| **Speech Recognition (STT)** | Gnani Prisma v2.5 | Automatic Speech Recognition for 10 Indic languages |
| **Speech Synthesis (TTS)** | Gnani Timbre v2.5 | Natural Text-to-Speech synthesis in regional voices |
| **Deep Learning Vision** | PyTorch 2.10 + TorchVision | MobileNetV3-Small foliar classifier (98.4% accuracy, 34.6 ms latency) |
| **Explainable AI (XAI)** | Grad-CAM | Backward gradient attribution hooks and saliency heatmaps |
| **Damage Quantification** | Severity Assessor | Evidence-based foliar lesion area segmentation and clinical staging |
| **Headless REST API** | FastAPI + Uvicorn | Programmatic voice, vision, and advisory endpoints |
| **Weather Integration** | Open-Meteo REST API | Live meteorological data, ET0, and ERA5 historical reanalysis |
| **Market Intelligence** | AGMARKNET & e-NAM + Holt Forecasting | Verified daily closing APMC rates, spatial discovery, and walk-forward forecasting |
| **Document Archival** | ReportLab 4.4 | Authentic PDF advisory report generation |
| **Offline-First PWA** | Service Worker + IndexedDB | Installable mobile web app at `/pwa`, local persistence, mutation queue |
| **Data Synchronization** | FastAPI Sync Engine | Change feed deltas, optimistic locking, idempotent push, conflict resolver |
| **Farm Economics & Profitability** | Decimal Arithmetic + CACP/ICAR Models | Exact cost per acre/quintal, break-even price/yield, ROI %, append-only audit |

---

## ⚙️ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/username/AgriGuardian.git
cd AgriGuardian
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```
Key variables in `.env`:
```env
GNANI_API_BASE_URL=https://api.vachana.ai
GNANI_API_KEY=your_gnani_api_key_here
GNANI_PRISMA_MODEL=prisma-v2.5
GNANI_TIMBRE_MODEL=timbre-v2.5
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run AgriGuardian
```bash
# Run the Streamlit desktop application
streamlit run app.py

# Or launch the headless FastAPI backend with PWA static mount
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
- Open Streamlit: `http://localhost:8501`
- Open Installable PWA: `http://localhost:8000/pwa`
- Open FastAPI Swagger Docs: `http://localhost:8000/docs`

---

## 🧪 Testing

Execute the comprehensive test suite with `pytest`:
```bash
# Run all unit, integration, offline sync, economics, and security tests (326 passed, 2 skipped)
python -m pytest tests -q

# Run dedicated Phase 12 security hardening, privacy, and e2e test suite (22 passed)
python -m pytest tests/security tests/e2e -v

# Run dedicated Phase 11 farm economics test suite (30 passed)
python -m pytest tests/economics -v

# Run dedicated Phase 10 synchronization test suite (20 passed)
python -m pytest tests/sync -v

# Run storage security & database migration test suites (9 passed)
python -m pytest tests/unit/test_storage_service.py tests/unit/test_migration.py -v

# Run with integration tests against live Gnani APIs (requires live GNANI_API_KEY)
python -m pytest tests/integration/test_gnani_live.py -v
```

---

## 📚 Documentation Links
* [Final Audit Report (`docs/FINAL_AUDIT_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/FINAL_AUDIT_REPORT.md)
* [Final Test Execution Report (`docs/FINAL_TEST_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/FINAL_TEST_REPORT.md)
* [Final Production Release Checklist (`docs/FINAL_RELEASE_CHECKLIST.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/FINAL_RELEASE_CHECKLIST.md)
* [Phase 12 Final Engineering Report (`docs/PHASE_12_FINAL_ENGINEERING_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_12_FINAL_ENGINEERING_REPORT.md)
* [Phase 12 Security Hardening Report (`docs/SECURITY_HARDENING_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/SECURITY_HARDENING_REPORT.md)
* [Phase 12 Privacy & Data Governance (`docs/PRIVACY_AND_DATA_GOVERNANCE.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PRIVACY_AND_DATA_GOVERNANCE.md)
* [Phase 12 Observability & Operations Guide (`docs/OBSERVABILITY_AND_OPERATIONS.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/OBSERVABILITY_AND_OPERATIONS.md)
* [Phase 12 Backup & Disaster Recovery Plan (`docs/BACKUP_AND_DISASTER_RECOVERY.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/BACKUP_AND_DISASTER_RECOVERY.md)
* [Phase 12 Production Deployment Guide (`docs/PRODUCTION_DEPLOYMENT_GUIDE.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PRODUCTION_DEPLOYMENT_GUIDE.md)
* [Phase 12 Production Readiness Checklist (`docs/PRODUCTION_READINESS_CHECKLIST.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PRODUCTION_READINESS_CHECKLIST.md)
* [Phase 11 Audit & Remediation (`docs/PHASE_11_AUDIT_AND_REMEDIATION.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_11_AUDIT_AND_REMEDIATION.md)
* [Phase 11 Final Engineering Report (`docs/PHASE_11_FINAL_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_11_FINAL_REPORT.md)
* [Phase 11 REST API Documentation (`docs/PHASE_11_API_DOCUMENTATION.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_11_API_DOCUMENTATION.md)
* [Phase 11 Financial & Agronomic Methodology (`docs/PHASE_11_FINANCIAL_METHODOLOGY.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_11_FINANCIAL_METHODOLOGY.md)
* [Phase 11 User & Field Guide (`docs/PHASE_11_USER_GUIDE.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_11_USER_GUIDE.md)
* [Phase 10 Verification Gate (`docs/PHASE_10_VERIFICATION.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_10_VERIFICATION.md)
* [Phase 10 Final Engineering Report (`docs/PHASE_10_FINAL_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_10_FINAL_REPORT.md)
* [Phase 10 Sync API Documentation (`docs/PHASE_10_SYNC_API_DOCUMENTATION.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_10_SYNC_API_DOCUMENTATION.md)
* [Phase 10 Offline Operations Guide (`docs/PHASE_10_OFFLINE_OPERATIONS_GUIDE.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_10_OFFLINE_OPERATIONS_GUIDE.md)
* [Phase 9 Verification Gate (`docs/PHASE_9_VERIFICATION.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_9_VERIFICATION.md)
* [Phase 9 Final Engineering Report (`docs/PHASE_9_FINAL_REPORT.md`)](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docs/PHASE_9_FINAL_REPORT.md)
* [Phase 9 REST API Documentation (`docs/PHASE_9_API_DOCUMENTATION.md`)](file:///docs/PHASE_9_API_DOCUMENTATION.md)
* [Phase 9 Operations & Maintenance Guide (`docs/PHASE_9_OPERATIONS_GUIDE.md`)](file:///docs/PHASE_9_OPERATIONS_GUIDE.md)
* [Phase 8 Verification & Security Hardening (`docs/PHASE_8_VERIFICATION.md`)](file:///docs/PHASE_8_VERIFICATION.md)
* [Phase 4 Final Engineering Report (`docs/PHASE_4_FINAL_REPORT.md`)](file:///docs/PHASE_4_FINAL_REPORT.md)
* [Weather Risk Evaluation & Calibration Report (`docs/WEATHER_RISK_EVALUATION.md`)](file:///docs/WEATHER_RISK_EVALUATION.md)
* [Phase 3 Verification & Remediation Audit (`docs/PHASE_3_VERIFICATION.md`)](file:///docs/PHASE_3_VERIFICATION.md)
* [Phase 3 Final Engineering Report (`docs/PHASE_3_FINAL_REPORT.md`)](file:///docs/PHASE_3_FINAL_REPORT.md)
* [Knowledge Architecture Specification (`docs/KNOWLEDGE_ARCHITECTURE.md`)](file:///docs/KNOWLEDGE_ARCHITECTURE.md)
* [Authoritative Source Registry (`docs/SOURCE_REGISTRY.md`)](file:///docs/SOURCE_REGISTRY.md)
* [Knowledge Governance & Safety Manual (`docs/KNOWLEDGE_GOVERNANCE.md`)](file:///docs/KNOWLEDGE_GOVERNANCE.md)
* [Recommendation Engine Pipeline (`docs/RECOMMENDATION_ENGINE.md`)](file:///docs/RECOMMENDATION_ENGINE.md)
* [Retrieval Benchmark Evaluation (`docs/RETRIEVAL_EVALUATION_REPORT.md`)](file:///docs/RETRIEVAL_EVALUATION_REPORT.md)
* [Phase 2 Pre-flight Verification Report (`docs/PHASE_2_VERIFICATION.md`)](file:///docs/PHASE_2_VERIFICATION.md)
* [Comprehensive Codebase Audit (`docs/PROJECT_AUDIT.md`)](file:///docs/PROJECT_AUDIT.md)
* [System Architecture (`docs/ARCHITECTURE.md`)](file:///docs/ARCHITECTURE.md)
* [Configuration Guide (`docs/CONFIGURATION.md`)](file:///docs/CONFIGURATION.md)
* [API Integrations (`docs/API_INTEGRATIONS.md`)](file:///docs/API_INTEGRATIONS.md)

---

## 🪴 License
This project is licensed under the MIT License.
