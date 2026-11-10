# AgriGuardian — Hackathon Demo Video Visual Index & Presentation Script

This document catalogs the complete suite of **13 genuine application screenshots** captured from the live, running AgriGuardian platform at `1440x900` resolution. It is designed as an end-to-end storyboard and voiceover script for a high-impact **2–3 minute hackathon demonstration video**.

---

## 🌾 Video Storyboard & Screenshot Catalog

```
+---------------------------------------------------------------------------------------------------+
|                                      AGRIGUARDIAN DEMO SEQUENCE                                   |
+---------------------------------------------------------------------------------------------------+
|  01. Product Identity       -->  02. Farm Overview           -->  03. Vernacular Voice AI         |
|  (Landing / Login)              (Farmer Dashboard)               (Gnani AI STT/TTS)               |
+---------------------------------------------------------------------------------------------------+
|  04. Crop Vision AI         -->  05. Agricultural Advisory   -->  06. Microclimate Forecast       |
|  (MobileNetV3 Diagnosis)         (ICAR / Dynamic Rules)          (Open-Meteo Weather)             |
+---------------------------------------------------------------------------------------------------+
|  07. APMC Market Intel      -->  08. Farm Management         -->  09. Human-in-the-Loop           |
|  (Price Surges & Mandis)         (Plots & Crop Stages)           (Agronomist Portal)              |
+---------------------------------------------------------------------------------------------------+
|  10. Proactive Alerts       -->  11. Low-Connectivity PWA    -->  12. Farm Financials            |
|  (Weather/Price Outbox)          (IndexedDB Sync Engine)         (Cost, Profit & Break-Even)      |
+---------------------------------------------------------------------------------------------------+
|                                  13. Unified Impact Dashboard                                     |
|                                  (Holistic Farm Intelligence)                                     |
+---------------------------------------------------------------------------------------------------+
```

---

### Screenshot 01: Product Identity & Secure Multi-Role Portal
- **File:** [01-landing-login.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/01-landing-login.png)
- **Application Surface:** `http://localhost:8501/` (Login & Identity Selection)
- **Key Capabilities Highlighted:**
  - Modern, farmer-first branding and responsive layout.
  - Role-based authentication supporting **Farmers**, **Field Extension Officers**, **Certified Agronomists**, and **Platform Administrators**.
  - Secure credential verification with BCrypt hashing and JWT token issuance.
- **Suggested Voiceover (0:00 – 0:12):**
  > *"Meet AgriGuardian: an AI-driven agricultural companion purpose-built for India's 140 million smallholder farmers. From secure onboarding to role-specific interfaces for farmers, field officers, and agronomists, AgriGuardian establishes trustworthy agricultural intelligence from day one."*

---

### Screenshot 02: Personalized Farmer Dashboard
- **File:** [02-farmer-dashboard.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/02-farmer-dashboard.png)
- **Application Surface:** `http://localhost:8501/` (Dashboard View — Farmer Context)
- **Key Capabilities Highlighted:**
  - Summary metrics: 32 registered farms, 360 acres of managed land, active crop cycles.
  - Live APMC mandi benchmark pricing (₹2,426/quintal for Wheat in Khanna, Punjab).
  - Real-time crop phenological stage tracking (Standard Wheat in Sowing/Germination stage).
- **Suggested Voiceover (0:12 – 0:25):**
  > *"When farmer Ramesh Patel opens AgriGuardian, he immediately sees his unified operational picture: 32 active farm records across 360 acres, his current crop cycle in the critical germination phase, and live APMC benchmark prices right from his local mandi."*

---

### Screenshot 03: Multilingual Vernacular Voice Assistant (Gnani AI)
- **File:** [03-voice-assistant.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/03-voice-assistant.png)
- **Application Surface:** `http://localhost:8501/` (Voice Assistant View)
- **Key Capabilities Highlighted:**
  - Gnani.ai Prisma STT (Speech-to-Text) and Timbre TTS (Text-to-Speech) integration.
  - Native Hindi and vernacular voice interactions enabling access for low-literacy farmers.
  - End-to-end voice query resolution with agricultural context persistence.
- **Suggested Voiceover (0:25 – 0:40):**
  > *"Recognizing that many farmers prefer voice over typing, AgriGuardian integrates Gnani AI for natural Hindi voice interactions. Farmers can speak freely about crop symptoms or market queries and receive spoken, dialect-aware guidance in seconds."*

---

### Screenshot 04: Edge-Ready Crop Disease Diagnosis (Vision AI)
- **File:** [04-crop-disease-detection.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/04-crop-disease-detection.png)
- **Application Surface:** `http://localhost:8501/` (Crop Disease Diagnosis View)
- **Key Capabilities Highlighted:**
  - PyTorch MobileNetV3 deep learning classifier for rapid leaf disease detection.
  - Live inference result: **Tomato Early Blight (*Alternaria solani*)** detected with **71.6% confidence** (High Confidence).
  - Grad-CAM attention heatmap verification highlighting pathogenic lesions.
  - Immediate organic and CIBRC-compliant chemical intervention protocols.
- **Suggested Voiceover (0:40 – 0:58):**
  > *"When a farmer spots disease symptoms, our computer vision pipeline instantly diagnoses the issue from a phone photo. Here, MobileNetV3 identifies Tomato Early Blight with 71.6% confidence, verified via Grad-CAM heatmaps, and immediately delivers CIBRC-compliant spray protocols with statutory pre-harvest intervals."*

---

### Screenshot 05: Evidence-Grounded Agricultural Advisory
- **File:** [05-agricultural-knowledge.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/05-agricultural-knowledge.png)
- **Application Surface:** `http://localhost:8501/` (Farm Advisory & Knowledge Base)
- **Key Capabilities Highlighted:**
  - Curated agronomic knowledge retrieval backed by ICAR, TNAU, and State Agricultural Universities.
  - Fertilizer schedules, dosage calculators, and pre-harvest interval (PHI) compliance guidelines.
  - Bilingual guidance rendering (English & Hindi) with actionable farmer instructions.
- **Suggested Voiceover (0:58 – 1:12):**
  > *"Every recommendation is grounded in verified agricultural science from ICAR and State Agricultural Universities. Dosage calculators eliminate dangerous over-application, protect groundwater, and ensure strict food safety compliance."*

---

### Screenshot 06: Weather Intelligence & Crop-Risk Warnings
- **File:** [06-weather-intelligence.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/06-weather-intelligence.png)
- **Application Surface:** `http://localhost:8501/` (Weather Intelligence View)
- **Key Capabilities Highlighted:**
  - Real-time weather data integration via Open-Meteo API.
  - Multi-day temperature, rainfall, and wind forecasts tailored to farm coordinates.
  - Rule-based microclimate risk warnings (e.g., frost danger, high humidity blight advisories, and spraying windows).
- **Suggested Voiceover (1:12 – 1:25):**
  > *"Our weather engine monitors hyper-local forecasts and dynamically generates crop-risk alerts. Farmers are warned before heavy rainfall hits, preventing washed-out fertilizers and optimizing spraying schedules."*

---

### Screenshot 07: APMC Mandi Market Intelligence & Price Forecasting
- **File:** [07-market-intelligence.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/07-market-intelligence.png)
- **Application Surface:** `http://localhost:8501/` (Market Intelligence View)
- **Key Capabilities Highlighted:**
  - Real-time mandi modal price tracking across regional APMC markets.
  - Price comparisons against statutory Minimum Support Price (MSP) benchmarks.
  - Price trends and volatility indicators to help farmers select the most profitable mandi.
- **Suggested Voiceover (1:25 – 1:40):**
  > *"Market intelligence gives smallholders bargaining power. AgriGuardian monitors nearby APMC mandis, highlighting price surges above statutory MSPs and guiding farmers on exactly when and where to sell for maximum revenue."*

---

### Screenshot 08: Multi-Farm & Plot Lifecycle Management
- **File:** [08-farm-management.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/08-farm-management.png)
- **Application Surface:** `http://localhost:8501/` (Farm Management View)
- **Key Capabilities Highlighted:**
  - Multi-farm tenancy: individual plot geometries, soil profiles, and irrigation types.
  - Crop cycle lifecycle tracking from sowing to harvest with stage-by-stage task calendars.
  - Strict farmer data ownership and privacy boundaries.
- **Suggested Voiceover (1:40 – 1:52):**
  > *"Farmers manage their entire agricultural portfolio across plots, soil types, and crop seasons. Automated crop calendars schedule irrigation, fertilizer, and weeding tasks tailored to each plot's soil profile."*

---

### Screenshot 09: Human-in-the-Loop Expert Agronomist Verification
- **File:** [09-expert-verification.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/09-expert-verification.png)
- **Application Surface:** `http://localhost:8501/` (Expert Agronomist Portal)
- **Key Capabilities Highlighted:**
  - ICAR-certified expert portal authenticated under Dr. Sharma (`exp_4e1ad1b2e3a1`).
  - Active review dossier: Wheat Yellow Rust case in Karnal, Haryana.
  - Formal agronomist override/approval workflow with feedback loop to refine AI diagnostic models.
- **Suggested Voiceover (1:52 – 2:08):**
  > *"AI shouldn't operate in a vacuum. AgriGuardian features a human-in-the-loop expert portal where certified ICAR agronomists review high-stakes diagnoses, verify chemical prescriptions, and continuously refine our knowledge base."*

---

### Screenshot 10: Proactive Multi-Channel Notifications & Outbox Telemetry
- **File:** [10-notifications.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/10-notifications.png)
- **Application Surface:** `http://localhost:8501/` (Alerts & Reminders View)
- **Key Capabilities Highlighted:**
  - Multilingual notification inbox (Emergency Weather Warnings, High APMC Price Alerts, Expert Verifications).
  - Multi-channel delivery engine: SMS, WhatsApp, and Voice IVR with strict consent management.
  - Durable, zero-data-loss outbox telemetry ensuring reliable delivery even during network dips.
- **Suggested Voiceover (2:08 – 2:22):**
  > *"AgriGuardian proactively pushes time-critical alerts to farmers across SMS, WhatsApp, and Voice IVR. Our zero-data-loss outbox guarantees that life-saving storm warnings and expert approvals reach farmers without delay."*

---

### Screenshot 11: Offline-First Progressive Web App (PWA) & Sync Engine
- **File:** [11-offline-pwa.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/11-offline-pwa.png)
- **Application Surface:** `http://localhost:8000/pwa/index.html` (Mobile PWA Companion)
- **Key Capabilities Highlighted:**
  - Installable PWA with durable IndexedDB local storage and Service Worker caching.
  - Full offline field capability: logging activities, queuing symptom photos, and managing reminders without connectivity.
  - Monotonic change-feed replication and intelligent conflict resolution upon network restoration.
- **Suggested Voiceover (2:22 – 2:38):**
  > *"In remote fields where internet fails completely, AgriGuardian's Offline-First PWA keeps running. Farmers log observations, schedule irrigation, and queue photos in IndexedDB; as soon as connectivity returns, changes synchronize seamlessly with the cloud."*

---

### Screenshot 12: Farm Economics, Profitability & Break-Even Intelligence
- **File:** [12-farm-economics.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/12-farm-economics.png)
- **Application Surface:** `http://localhost:8501/` (Farm Economics & Profitability View)
- **Key Capabilities Highlighted:**
  - Decimal-exact accounting: Total Cost (₹2,700), Realized Revenue (₹98,000), Net Profit (₹95,300).
  - Category-wise expense distribution and multi-event harvest tracking.
  - Break-even price and yield calculators coupled with seasonal budget variance monitoring.
- **Suggested Voiceover (2:38 – 2:50):**
  > *"Farming is a business. Our Farm Economics engine delivers decimal-exact financial tracking—monitoring input expenses, labor costs, harvest yields, and mandi sales to compute gross margins, net profit, and break-even selling prices."*

---

### Screenshot 13: Final Unified Intelligence & Platform Value Dashboard
- **File:** [13-final-dashboard.png](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/demo/screenshots/13-final-dashboard.png)
- **Application Surface:** `http://localhost:8501/` (Unified Platform Dashboard)
- **Key Capabilities Highlighted:**
  - Complete operational lifecycle synthesis: Land, Crops, Tasks, Weather Risk, and Market Prices.
  - Vega-Lite data visualizations showing crop area distribution and phenological stage progression.
  - Proves the holistic transformational value AgriGuardian delivers to Indian agriculture.
- **Suggested Voiceover (2:50 – 3:00):**
  > *"From voice assistance and edge vision to offline resilience and financial intelligence, AgriGuardian empowers Indian farmers to protect their crops, maximize profits, and cultivate a resilient future. Thank you."*

---

## 🛠️ Verification & Reproducibility Note

- **Environment Viewport:** All screenshots were captured at a standardized `1440x900` browser viewport.
- **Data Integrity:** All displayed diagnoses, weather feeds, mandi prices, and financial ledgers originate from genuine backend inference, SQLite database storage, and active service integrations.
- **Repository Location:** All image files are stored in `demo/screenshots/` and tracked in version control on branch `main`.
