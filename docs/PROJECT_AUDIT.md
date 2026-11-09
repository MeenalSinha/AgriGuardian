# AgriGuardian — Comprehensive Codebase Audit & Baseline Report

**Date:** November 2026  
**Auditor:** Senior Software Architect & AI/ML Systems Engineer  
**Scope:** AgriGuardian Codebase (Phase 0 Audit)  
**Status:** Baseline Completed  

---

## 1. Executive Summary

AgriGuardian is an AI-powered agricultural intelligence application built with Streamlit, aimed at assisting Indian farmers with crop disease diagnosis, weather insights, market prices, and contextual recommendations.

Prior to refactoring, the application existed as a single monolithic script (`app.py`, 2,536 lines, ~102 KB) alongside an exploratory Jupyter notebook (`AgriGuardian.ipynb`, 116 cells). While the core user interface demonstrated an ambitious and intuitive vision, the codebase exhibited severe architectural fragilities, unhandled runtime crashes, simulated ML components masquerading as production models, and hardcoded configurations.

This audit details the verified state of all repository components, identifies breaking issues, establishes a baseline, and outlines the foundational refactoring required for Phase 0 and Phase 1.

---

## 2. Feature Inventory & Verification Matrix

Each existing feature was inspected and verified against the actual codebase. Features are classified into one of six standard statuses:
1. **Fully implemented and verified**
2. **Implemented but requiring improvements**
3. **Partially implemented**
4. **Simulated or using placeholder data**
5. **Broken or non-functional**
6. **Not implemented**

| Feature / Component | Classification | Verification Findings & Reality Check |
|---|---|---|
| **Application Startup (`app.py`)** | **5. Broken / Non-functional** | Calling `AgriGuardian()` at line 1402 instantiated `LLMAdvisor()` with `None`, immediately throwing `openai.OpenAIError: The api_key client option must be set...`. The app could not start without `OPENAI_API_KEY` pre-configured in the environment. |
| **OpenAI Client Initialization** | **5. Broken / Non-functional** | `openai 1.30.0` with `httpx 0.28.1` on Python 3.14 failed with `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` because `openai`'s internal wrapper used the deprecated `proxies` keyword. |
| **Crop Disease Detection** | **4. Simulated / Placeholder** | Despite the README claiming a trained ResNet50 model with 92% validation accuracy on PlantVillage, `PlantDiseaseDetector.predict()` returned hardcoded mock dictionaries (`Early Blight`, 0.87 confidence). |
| **Grad-CAM Explainability** | **4. Simulated / Placeholder** | `_generate_mock_gradcam()` generated random noise `np.random.rand(224, 224)` and plotted it with a `jet` colormap over the image. No real gradient calculation or neural layer backpropagation was performed. |
| **Disease Severity Scoring** | **2. Implemented (needs improvement)** | `calculate_severity()` applies heuristic weights based on disease severity, recent rain, high temperature, and humidity. Conceptually sound, but used arbitrary magic numbers (`* 25`, thresholds > 75, > 50, > 25). |
| **Weather Integration** | **2. Implemented (needs improvement)** | Live integration using Open-Meteo REST API (`https://api.open-meteo.com/v1/forecast`) for temperature, humidity, precipitation, and 3-day forecast. Functional, but default coordinates were hardcoded to New Delhi (28.6139, 77.2090) and lacked graceful degradation. |
| **Market Intelligence** | **4. Simulated / Placeholder** | Function named `get_market_data_real` actually queried a static hardcoded Python dictionary (`market_database`) for 8 crops (wheat, rice, cotton, sugarcane, potato, tomato, onion, maize). Presented to farmers as "real-time market prices" without source disclosure. |
| **Dynamic Rule Engine** | **1. Fully implemented and verified** | Loaded rules from `config/rules.json` or fallback rules. Evaluated symptoms keywords against weather conditions. Works as designed. |
| **LLM Advisory Generation** | **2. Implemented (needs improvement)** | Constructed prompt for GPT-3.5 with context (crop, symptoms, weather, market, rule results) and parsed JSON. Had a fallback advisory function for failures. Required insulation from missing API keys. |
| **Speech Generation (gTTS)** | **2. Implemented (needs improvement)** | Used Google Text-to-Speech (`gTTS`) to synthesize an MP3 file. Functional, but synchronous by default, prone to network stalls, and did not use Gnani AI's native Indian language speech engine. |
| **PDF Report Generation** | **4. Simulated / Placeholder** | `PDFReportGenerator.generate_report()` actually generated a plain `.txt` file containing formatted text, saved it with `.txt` extension, and logged `PDF report generated`. No actual PDF binary was created. |
| **Data Persistence (CSV)** | **2. Implemented (needs improvement)** | `DataManager` initialized `data/advisories.csv` and appended records. Worked, but lacked thread-safe file locking, data validation, and sanitized inputs. |
| **Analytics Dashboard** | **1. Fully implemented and verified** | `EnhancedDashboardAnalytics` computed risk distributions, crop-wise issues, daily counts, and response time metrics from the CSV. Functional when records existed. |
| **Demo Cases** | **1. Fully implemented and verified** | 3 pre-configured scenarios (Rajesh Kumar - Wheat, Suresh Patel - Cotton, Anita Devi - Tomato) with geographic coordinates and symptoms. Functional. |
| **Location Map (Folium)** | **1. Fully implemented and verified** | Rendered interactive map centered on selected farm coordinates using `folium` and `streamlit_folium`. Functional. |
| **Gnani Prisma STT** | **6. Not implemented** | No integration existed prior to this phase. |
| **Gnani Timbre TTS** | **6. Not implemented** | No integration existed prior to this phase. |
| **Voice Conversation Engine** | **6. Not implemented** | No multi-turn voice conversation orchestration existed prior to this phase. |

---

## 3. Deep Component Analysis

### 3.1 Crop Disease Detection & Vision Pipeline
* **Claim vs. Reality:** Documentation claimed a pre-trained ResNet50 model achieving 92% validation accuracy. In reality, `PlantDiseaseDetector` contained zero neural network weights, zero PyTorch/TensorFlow inference code, and simply returned a mock prediction dict with `Early Blight` at 0.87 confidence.
* **Grad-CAM:** The `_generate_mock_gradcam` method computed `heatmap = np.random.rand(img.shape[0], img.shape[1])` and layered it over the uploaded image. This was an ungrounded random visualization.
* **Remediation for Phase 0:** Document this transparently. Preserve the interface contracts (`predict`, `calculate_severity`) so existing UI flows remain uninterrupted, but decouple the mock logic, provide honest labeling, and build a modular structure in `services/agriculture/disease_detector.py` that supports real model plug-ins without pretending random numbers are AI explainability.

### 3.2 Weather Integration
* **Provider:** Open-Meteo REST API (`https://api.open-meteo.com/v1/forecast`).
* **Parameters:** `latitude`, `longitude`, `current=temperature_2m,relative_humidity_2m,precipitation,weather_code`, `daily=...`, `timezone=Asia/Kolkata`.
* **Strengths:** Genuine, free, does not require an API key, fast response.
* **Weaknesses:** Hardcoded default coordinates, no circuit breaker or exponential backoff on network failures, hardcoded temperature thresholds (e.g., `temp > 35` for `temp_high`, `humidity > 80` for `humidity_high`).
* **Remediation for Phase 0:** Move to `integrations/weather/open_meteo.py`, inject timeout and retry settings from centralized config, and document weather parameters.

### 3.3 Market Intelligence
* **Current Implementation:** `get_market_data_real` was misleadingly named. It queried a static hardcoded dictionary with values such as Wheat = ₹2100/quintal, Rice = ₹2800/quintal.
* **Safety Concern:** Recommending farmers to sell or hold based on static numbers ("Good time to sell. Prices increasing") could lead to financial harm if presented as real-time market data.
* **Remediation for Phase 0:** Clearly label static data as "Indicative Benchmark Prices (Agmarknet Reference Model)" with explicit timestamps and disclaimer tags. Move to `integrations/market/market_service.py` with extensible interface for future live eNAM/Agmarknet API integrations.

### 3.4 LLM Advisory
* **Provider:** OpenAI Chat Completions (`gpt-3.5-turbo`).
* **Issues:**
  1. Direct instantiation without API key caused application crash on startup.
  2. Incompatible `openai 1.30.0` with `httpx 0.28.1` resulted in `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`.
  3. Prompts lacked guardrails against hallucinatory chemical dosages.
* **Remediation for Phase 0:**
  1. Fix the client initialization bug by passing a custom `httpx.Client()` instance.
  2. Implement safe fallback when no API key is supplied without crashing the application.
  3. Build an extensible LLM provider interface in `integrations/llm/` supporting OpenAI and ready for Gnani Evon.

### 3.5 PDF Report Generation
* **Current Implementation:** Created a `.txt` file inside `reports/` and claimed it was a PDF.
* **Remediation for Phase 0:** Implement genuine PDF generation using `reportlab` (already present in the environment) with clean styling, while maintaining a robust fallback.

---

## 4. Technical Debt & Security Audit

### 4.1 Security Vulnerabilities
1. **Runtime `os.environ` Mutation:** The sidebar text input directly executed:
   ```python
   if api_key:
       os.environ['OPENAI_API_KEY'] = api_key
       st.session_state.system = AgriGuardian(api_key)
   ```
   In a multi-user or shared Streamlit server, this pollutes process-level environment variables for all concurrent sessions.
2. **Missing Credential Sanitization:** API keys, file paths, and potential PII had no structured redaction in logging.
3. **No `.env` Exclusions:** `.gitignore` contained only `__pycache__` and `.DS_Store`, leaving `.env` and temporary audio files vulnerable to accidental commits.

### 4.2 Architecture Technical Debt
1. **Monolithic File:** All 2,536 lines were crammed into a single file with circular dependencies on Streamlit's global state.
2. **Missing Unit Tests:** Zero tests existed in the repository.
3. **Lack of Configuration Management:** URLs, thresholds, and limits were scattered throughout the code as magic constants.

---

## 5. Baseline Test & Verification Summary

| Test Condition | Result | Notes |
|---|---|---|
| Run `python app.py` without environment variables | **FAILED** | Crashed with `OpenAIError: The api_key client option must be set` |
| Run `python app.py` with dummy `OPENAI_API_KEY` | **FAILED** | Crashed with `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` |
| Run Open-Meteo Weather endpoint | **PASSED** | Returned 200 OK with live meteorological forecast for Delhi |
| Run Dynamic Rule Engine with rules.json | **PASSED** | Correctly parsed 4 rules and evaluated symptom triggers |
| Run DataManager with advisories.csv | **PASSED** | Correctly created CSV header and saved records |
| Run gTTS audio generation | **PASSED** | Generated valid MP3 file in `audio/` |

---

## 6. Phase 0 & Phase 1 Action Plan

1. **Phase 0 (Foundation):**
   - Create centralized Pydantic Settings (`backend/app/core/config.py`).
   - Secure secrets via `.env.example` and update `.gitignore`.
   - Implement structured logging and custom exception hierarchy.
   - Refactor codebase into modular packages (`backend/app/`, `frontend/`).
   - Implement real ReportLab PDF generation.
   - Fix OpenAI initialization bug and implement graceful degraded advisory.
   - Author comprehensive Pytest test suite establishing baseline coverage.
   - Author `ARCHITECTURE.md`, `CONFIGURATION.md`, `API_INTEGRATIONS.md`, and update `README.md`.

2. **Phase 1 (Gnani AI Voice Intelligence):**
   - Implement official Gnani Prisma v2.5 STT client (`integrations/gnani/prisma_stt.py`).
   - Implement official Gnani Timbre v2.5 TTS client (`integrations/gnani/timbre_tts.py`).
   - Implement audio validation, format handling, and safe temporary file cleanup (`audio_utils.py`).
   - Implement multi-turn Voice Conversation Engine (`services/conversation/engine.py`).
   - Connect voice input to existing agricultural reasoning with intent routing.
   - Implement accessible Streamlit Voice Assistant interface.
   - Write unit, integration, and failure scenario tests.

