# AgriGuardian — Configuration Guide

This document provides a comprehensive reference for all configurable parameters in AgriGuardian. Configuration is managed centrally via `backend/app/core/config.py` using `pydantic-settings`.

---

## 1. Quick Setup

1. Copy the example configuration template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your service credentials:
   ```env
   GNANI_API_KEY=your_actual_gnani_api_key_here
   OPENAI_API_KEY=your_actual_openai_key_here
   ```
3. Never commit `.env` to Git. Version control ignores `.env` automatically via `.gitignore`.

---

## 2. Configuration Reference Matrix

### 2.1 Application Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_NAME` | string | `AgriGuardian` | Application display title. |
| `APP_ENV` | string | `development` | Deployment environment (`development`, `staging`, `production`, `test`). |
| `DEBUG` | boolean | `false` | Enable verbose debug logging and diagnostics. |
| `LOG_LEVEL` | string | `INFO` | Logging threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |

### 2.2 Gnani AI Multilingual Voice Intelligence

| Variable | Type | Default | Description |
|---|---|---|---|
| `GNANI_API_BASE_URL` | string | `https://api.vachana.ai` | Official base URL for Gnani speech APIs. |
| `GNANI_API_KEY` | string | `None` *(Required)* | Gnani API Key sent via `X-API-Key-ID` header. |
| `GNANI_PRISMA_MODEL` | string | `prisma-v2.5` | Verified model for Speech-to-Text (STT). |
| `GNANI_PRISMA_ENDPOINT` | string | `/stt/v3` | REST endpoint for speech recognition. |
| `GNANI_PRISMA_DEFAULT_LANGUAGE` | string | `hi-IN` | Default BCP-47 language tag for transcription. |
| `GNANI_PRISMA_TIMEOUT_SECONDS` | float | `20.0` | HTTP request timeout for STT transcription. |
| `GNANI_PRISMA_MAX_RETRIES` | integer | `2` | Number of exponential backoff retries on network failures. |
| `GNANI_TIMBRE_MODEL` | string | `timbre-v2.5` | Verified model for Text-to-Speech (TTS). |
| `GNANI_TIMBRE_ENDPOINT` | string | `/api/v1/tts/inference` | REST endpoint for speech synthesis. |
| `GNANI_TIMBRE_DEFAULT_VOICE` | string | `Nalini` | Default voice identifier for Hindi synthesis. |
| `GNANI_TIMBRE_DEFAULT_LANGUAGE` | string | `hi-IN` | Default language for speech synthesis. |
| `GNANI_TIMBRE_SPEED` | float | `1.0` | Speech playback speed multiplier (0.5 to 2.0). |
| `GNANI_TIMBRE_SAMPLE_RATE` | integer | `48000` | Audio sampling frequency in Hz (8000 to 48000). |
| `GNANI_TIMBRE_CONTAINER` | string | `wav` | Audio container format (`wav`, `mp3`, `ogg`). |
| `GNANI_TIMBRE_ENCODING` | string | `linear_pcm` | Audio encoding format (`linear_pcm`, `pcm_s16le`). |
| `GNANI_TIMBRE_TIMEOUT_SECONDS` | float | `25.0` | HTTP request timeout for TTS synthesis. |
| `GNANI_TIMBRE_MAX_RETRIES` | integer | `2` | Number of retries on 5xx or connection drops. |

### 2.3 LLM & AI Advisory Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | string | `openai` | Reasoning provider (`openai`, extensible to `evon`). |
| `OPENAI_API_KEY` | string | `None` | OpenAI API key for GPT-3.5 contextual advisory. |
| `LLM_MODEL` | string | `gpt-3.5-turbo` | Target LLM model identifier. |
| `LLM_TEMPERATURE` | float | `0.7` | Sampling temperature for agronomic recommendations. |
| `LLM_MAX_TOKENS` | integer | `600` | Max tokens generated in advisory response. |
| `LLM_TIMEOUT_SECONDS` | float | `15.0` | Timeout for LLM reasoning API calls. |
| `LLM_MAX_RETRIES` | integer | `2` | Number of retry attempts on LLM rate limits. |

### 2.4 Weather & Environmental Settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `OPEN_METEO_BASE_URL` | string | `https://api.open-meteo.com/v1/forecast` | Open-Meteo REST API endpoint. |
| `DEFAULT_LATITUDE` | float | `28.6139` | Fallback farm latitude (Delhi, India). |
| `DEFAULT_LONGITUDE` | float | `77.2090` | Fallback farm longitude (Delhi, India). |
| `DEFAULT_LOCATION_NAME` | string | `Delhi, India` | Default location label. |
| `WEATHER_CACHE_TTL_SECONDS` | integer | `3600` | In-memory cache duration for weather forecasts. |
| `WEATHER_TIMEOUT_SECONDS` | float | `8.0` | Request timeout for meteorological API calls. |

### 2.5 Storage & Operational Limits

| Variable | Type | Default | Description |
|---|---|---|---|
| `DATA_DIR` | string | `data` | Directory for CSV records and datasets. |
| `AUDIO_TEMP_DIR` | string | `audio` | Directory for temporary voice recordings and synthesized audio. |
| `REPORTS_DIR` | string | `reports` | Target directory for generated PDF advisory reports. |
| `LOGS_DIR` | string | `logs` | Directory for structured log files. |
| `CONFIG_DIR` | string | `config` | Directory for dynamic rules and system configurations. |
| `MAX_AUDIO_SIZE_BYTES` | integer | `10485760` (10 MB) | Maximum permitted audio upload size. |
| `MAX_AUDIO_DURATION_SECONDS`| integer | `60` | Maximum audio duration in seconds. |
| `CLEANUP_TEMP_FILES` | boolean | `true` | Automatically purge audio files older than retention TTL. |

---

## 3. Development vs. Production Guidelines

* **Development:**
  * `DEBUG=true`, `LOG_LEVEL=DEBUG`
  * Missing API keys gracefully fall back to local rule-based recommendations.
* **Production:**
  * `DEBUG=false`, `LOG_LEVEL=INFO`
  * Provide verified `GNANI_API_KEY` to enable high-fidelity voice conversations.
  * Ensure storage directories (`audio/`, `reports/`) are mounted on durable persistent storage or cloud buckets.

