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

### 🔬 Crop Disease Diagnostics & Explainability
* **Symptom & Vision Assessment:** Identifies foliar pathogens and provides actionable management steps.
* **Salience Visualization (Grad-CAM):** Highlights critical leaf regions under diagnostic focus.
* **Compound Severity Scoring:** Factors ambient weather into diagnostic risk assessments.

### 📊 Field Officer Dashboard & PDF Exports
* **Regional Outbreak Analytics:** Tracks crop health queries, risk distributions, and peak advisory hours.
* **Case Management:** Review and update advisory statuses and agronomist field feedback.
* **Authentic PDF Generation:** Exports complete diagnostic reports compiled via ReportLab.

---

## 🧩 Architecture & Tech Stack

```text
AgriGuardian/
├── backend/
│   └── app/
│       ├── core/               # Centralized config, structured logging, custom exceptions
│       ├── schemas/            # Pydantic models for speech, conversations, and advisories
│       ├── integrations/       # Provider integrations: Gnani (Prisma/Timbre), Weather, Market, LLM
│       ├── services/           # Rule engine, disease detector, conversation engine, orchestrator
│       └── repositories/       # CSV persistence and case history
├── frontend/
│   ├── components/             # Reusable UI widgets, alerts, and Folium maps
│   └── views/                  # Voice Assistant, Farm Advisory, Dashboard, Demo Cases, About
├── docs/                       # Project audit, architecture, configuration, and API guides
├── tests/                      # Pytest unit and integration test suite
├── app.py                      # Main Streamlit application entrypoint
├── requirements.txt            # Dependency declarations
└── .env.example                # Configuration template
```

| Layer | Component | Description |
|---|---|---|
| **Presentation** | Streamlit 1.66 | Responsive, mobile-accessible farmer dashboard |
| **Speech Recognition (STT)** | Gnani Prisma v2.5 | Automatic Speech Recognition for Indic languages |
| **Speech Synthesis (TTS)** | Gnani Timbre v2.5 | Natural Text-to-Speech synthesis |
| **Core Architecture** | Python 3.14 + Pydantic v2 | Typed configuration and modular services |
| **Weather Integration** | Open-Meteo REST API | Live meteorological data and precipitation forecasts |
| **Market Intelligence** | Agmarknet Benchmark Model | Mandi pricing trends with source transparency |
| **Document Archival** | ReportLab 4.4 | Authentic PDF advisory report generation |

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
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing

Execute the comprehensive test suite with `pytest`:
```bash
# Run all unit and pipeline tests
python -m pytest tests -v

# Run with integration tests against live Gnani APIs (requires GNANI_API_KEY)
python -m pytest tests/integration/test_gnani_live.py -v
```

---

## 📚 Documentation Links
* [Comprehensive Codebase Audit (`docs/PROJECT_AUDIT.md`)](file:///docs/PROJECT_AUDIT.md)
* [System Architecture (`docs/ARCHITECTURE.md`)](file:///docs/ARCHITECTURE.md)
* [Configuration Guide (`docs/CONFIGURATION.md`)](file:///docs/CONFIGURATION.md)
* [API Integrations (`docs/API_INTEGRATIONS.md`)](file:///docs/API_INTEGRATIONS.md)

---

## 🪴 License
This project is licensed under the MIT License.
