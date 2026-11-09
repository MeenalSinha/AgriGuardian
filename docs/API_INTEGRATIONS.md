# AgriGuardian — External API Integrations Guide

This document specifies all third-party external integrations, API contracts, authentication protocols, and error management strategies implemented in AgriGuardian.

---

## 1. Gnani AI Voice Intelligence Stack

AgriGuardian integrates Gnani AI's official production speech models for Indic language voice conversations.

### 1.1 Gnani Prisma v2.5 — Speech-to-Text (STT)
* **Model Identifier:** `prisma-v2.5`
* **Base URL:** `https://api.vachana.ai`
* **Endpoint:** `POST /stt/v3`
* **Authentication Header:** `X-API-Key-ID: <YOUR_GNANI_API_KEY>`
* **Content-Type:** `multipart/form-data`
* **Supported Audio Formats:** WAV, MP3, OGG, FLAC, AAC, M4A, WebM (mono or stereo, 8kHz to 48kHz).
* **Payload Fields:**
  * `audio_file`: Binary file stream.
  * `language_code`: BCP-47 language tag (e.g., `hi-IN`, `ta-IN`, `te-IN`, `kn-IN`).
  * `model`: `prisma-v2.5`.

**Example Request (Python httpx):**
```python
import httpx

headers = {"X-API-Key-ID": "YOUR_API_KEY"}
files = {"audio_file": ("query.wav", open("query.wav", "rb"), "audio/wav")}
data = {"language_code": "hi-IN", "model": "prisma-v2.5"}

with httpx.Client(timeout=20.0) as client:
    response = client.post("https://api.vachana.ai/stt/v3", headers=headers, files=files, data=data)
    result = response.json()
```

**Expected Response Schema:**
```json
{
  "success": true,
  "request_id": "019fd17d-1106-7265-88ab-3ed12d029292",
  "transcript": "मेरे टमाटर के पौधों की पत्तियां पीली हो रही हैं।",
  "model": "prisma-v2.5",
  "detected_language_code": "hi-IN",
  "processing_time": 0.45,
  "end_to_end_latency": 0.52
}
```

---

### 1.2 Gnani Timbre v2.5 — Text-to-Speech (TTS)
* **Model Identifier:** `timbre-v2.5`
* **Base URL:** `https://api.vachana.ai`
* **Endpoint:** `POST /api/v1/tts/inference`
* **Authentication Header:** `X-API-Key-ID: <YOUR_GNANI_API_KEY>`
* **Content-Type:** `application/json`
* **Response Content-Type:** `audio/wav` (or `audio/mpeg` depending on container)

**Example Request:**
```json
{
  "text": "आपकी टमाटर की फसल में फफूंद के लक्षण हैं। कृपया जल निकासी में सुधार करें।",
  "voice": "Nalini",
  "model": "timbre-v2.5",
  "language": "hi-IN",
  "speed": 1.0,
  "audio_config": {
    "sample_rate": 48000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav"
  }
}
```

**Verified Voice Catalog for Timbre v2.5:**
* **Hindi (`hi-IN`):** `Nalini` (Female - default), `Deepak` (Male)
* **English (`en-IN`):** `Kaveri` (Female - default), `Arjun` (Male)
* **Tamil (`ta-IN`):** `Kaveri`
* **Telugu (`te-IN`):** `Kaveri`
* **Kannada (`kn-IN`):** `Kaveri`
* **Malayalam (`ml-IN`):** `Kaveri`
* **Marathi (`mr-IN`):** `Nalini`
* **Punjabi (`pa-IN`):** `Nalini`
* **Bengali (`bn-IN`):** `Nalini`
* **Gujarati (`gu-IN`):** `Nalini`

---

## 2. Open-Meteo Weather API
* **Base URL:** `https://api.open-meteo.com/v1/forecast`
* **Authentication:** None required (Open Access / Non-commercial permitted).
* **Parameters:**
  * `latitude`, `longitude`: Farm GPS coordinates.
  * `current`: `temperature_2m,relative_humidity_2m,precipitation,weather_code`
  * `daily`: `temperature_2m_max,temperature_2m_min,precipitation_sum`
  * `timezone`: `Asia/Kolkata`
  * `forecast_days`: 3
* **Circuit Breaker:** If connection fails or times out, `WeatherService` returns an estimated baseline rather than terminating user requests.

---

## 3. Market Pricing Intelligence
* **Source:** Agmarknet Mandi Benchmarks (Reference Model).
* **Transparency Commitment:** All market estimates are explicitly tagged with `is_live=False` and marked as "Indicative Reference Benchmark" to prevent misleading farmers.
* **Roadmap:** Future phases will integrate live APIs from eNAM (National Agriculture Market) and state APMC portals.

---

## 4. OpenAI / LLM Advisory Service
* **Model:** `gpt-3.5-turbo`
* **Authentication:** `Bearer <OPENAI_API_KEY>`
* **Compatibility Solution:** Instantiates a dedicated `httpx.Client()` to completely bypass the `proxies` argument incompatibility between `openai 1.30.0` and `httpx 0.28.1` on Python 3.14.
* **Fallback:** When unconfigured or unreachable, the system automatically falls back to deterministic expert recommendations derived from `config/rules.json`.

