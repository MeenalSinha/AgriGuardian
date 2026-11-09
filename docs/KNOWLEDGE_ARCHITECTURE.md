# AgriGuardian — Knowledge Architecture & Evidence Retrieval Specification

## 1. Executive Summary

In traditional LLM agricultural chatbots, advisory systems suffer from critical hallucinations: prescribing unapproved pesticides, fabricating chemical dosages, confusing seasonal planting windows, or recommending illegal toxic chemicals (such as Monocrotophos or Paraquat). 

AgriGuardian Phase 3 eliminates these risks by introducing an **Evidence-Grounded Agricultural Knowledge Architecture**. Every agronomic assertion, cultural sanitation measure, bio-control recommendation, and statutory caution is backed by verified atomic facts originating from Tier-1 and Tier-2 Indian agricultural authorities (ICAR, CIBRC, IMD, TNAU).

---

## 2. Architecture Overview

```
                                  +---------------------------+
                                  |   Farmer Multimodal Turn  |
                                  |  (Voice / Text / Leaf ML) |
                                  +-------------+-------------+
                                                |
                                                v
                                 +-----------------------------+
                                 | Safety Guardrail Inspection |
                                 |  (Injection & Banned Chems) |
                                 +--------------+--------------+
                                                |
                                                v
                               +--------------------------------+
                               | Hybrid BM25 Retrieval Engine   |
                               | (Indic Expansion & Token Match)|
                               +----------------+---------------+
                                                |
                                                v
                              +-----------------------------------+
                              | Context Applicability Filter      |
                              | (Crop, Growth Stage, DAS, Region) |
                              +-----------------+-----------------+
                                                |
                                                v
                              +-----------------------------------+
                              | Conflict Resolution Service       |
                              | (Tier-1 > Tier-2 > Tier-3 Weight) |
                              +-----------------+-----------------+
                                                |
                                                v
                              +-----------------------------------+
                              | Chemical Action Sanitization      |
                              | (Downgrade Unconfirmed Chemicals) |
                              +-----------------+-----------------+
                                                |
                                                v
                        +-----------------------------------------------+
                        | Evidence-Grounded LLM Advisor & Speech Engine |
                        |     (Citations, Explanations, Timbre TTS)     |
                        +-----------------------------------------------+
```

---

## 3. Data Model & Pydantic Schemas

### 3.1 `SourceRecord` (`backend/app/schemas/source.py`)
Tracks origin, statutory authority tier, and legal licensing of external agronomic publications:
- `source_id`: Alphanumeric key (e.g. `SRC-ICAR-TOMATO-2024`).
- `organization`: Institutional body (e.g., ICAR-IIHR, CIBRC, IMD).
- `authority_tier`: Tier-1 (Statutory/National), Tier-2 (State Agr. University), Tier-3 (District KVK).
- `canonical_url`: Official government or university URL.
- `document_hash`: SHA-256 cryptographic digest of raw guideline text.
- `crop_scope`: Crops covered by guideline.
- `state_scope`: Applicable Indian states (empty denotes national applicability).
- `next_review_date`: Sunset/re-certification date for periodic peer re-evaluation.

### 3.2 `EvidenceExcerpt` (`backend/app/schemas/evidence.py`)
Traceable snippet linked to a specific page or section of a registered source:
- `evidence_id`: Unique identifier (e.g., `EVD-TOM-001`).
- `source_id`: Foreign key reference to `SourceRecord`.
- `quote_or_excerpt`: Verbatim or faithfully summarized factual statement.
- `page` / `section`: Physical section location in the source document.
- `confidence_score`: Editorial verification confidence (default `1.0`).

### 3.3 `AtomicFact` (`backend/app/schemas/knowledge.py`)
Discrete knowledge unit evaluated in the graph repository:
- `fact_id`: Unique key (e.g., `FCT-TOM-EBLIGHT-CULT-01`).
- `subject`: Core entity (`tomato`, `wheat`, `rice`).
- `predicate`: Semantic relationship (`managed_by_cultural_practice`, `managed_by_biological_agent`, `chemical_precaution_recommended`, `has_critical_irrigation_stage`, `susceptible_to_weather_risk`).
- `value`: Actionable agronomic statement.
- `conditions`: Prerequisites (`crop_stage`, `days_after_sowing`, `severity_stage`).
- `geographic_scope`: Region boundaries.
- `evidence`: List of supporting `EvidenceExcerpt` objects.
- `authority_weight`: Derived ranking weight (Tier 1 = 1.4, Tier 2 = 1.2, Tier 3 = 1.0).

---

## 4. Ingestion & Storage Repositories

### 4.1 Persistence Format
- Sources Registry: `data/sources/registry.json`
- Facts Repository: `data/facts/facts.json`
- Knowledge Manifest: `data/knowledge_manifest.json` (contains SHA-256 integrity signatures).

### 4.2 Repository Singletons
- `SourceRepository`: Provides thread-safe singleton access (`SourceRepository.get_instance()`) with filtering by tier, crop, and lifecycle status.
- `FactRepository`: Provides thread-safe singleton access (`FactRepository.get_instance()`) with predicate indexing, subject queries, and conflict detection.

---

## 5. Retrieval & Multilingual Expansion Engine

### 5.1 BM25 Lexical Scoring
Retrieval evaluates document tokens composed of:
$$\text{Document Text} = \text{fact\_id} + \text{subject} + \text{category} + \text{predicate} + \text{value} + \text{scientific\_name} + \text{conditions} + \text{evidence\_text}$$

BM25-style length normalization penalizes excessively verbose text while rewarding concise, high-density factual matches:
$$\text{Score} = \frac{|\text{Query Tokens} \cap \text{Document Tokens}|}{\log(|\text{Document Tokens}| + 1) + 1.0} \times \text{Authority Multiplier}$$

### 5.2 Multilingual Indic Synonym Mapping (`AGRI_SYNONYMS`)
Enables cross-lingual semantic matching between English scientific terminology and colloquial Indic queries:
- **Tomato:** `टमाटर`, `tamatar`, `solanum lycopersicum`.
- **Wheat:** `गेहूं`, `gehun`, `kanak`, `triticum`.
- **Blight:** `झुलसा`, `jhulsa`, `early blight`, `late blight`, `alternaria`, `phytophthora`.
- **Spot:** `धब्बा`, `धब्बे`, `dhabba`, `spots`, `leaf spot`, `bacterial spot`.
- **Rust:** `गेरुआ`, `रतुआ`, `tula`, `puccinia`, `yellow rust`, `stripe rust`.
- **Irrigation:** `सिंचाई`, `पानी`, `water`, `watering`, `drip`.
- **CRI Stage:** `सीआरआई`, `cri`, `crown root initiation`, `tillering`.

---

## 6. Context Applicability & Conflict Arbitration

1. **Stage Matching:** Checks if farmer's crop stage or Days After Sowing (DAS) matches fact prerequisites. If unspecified, triggers clarifying questions instead of guessing.
2. **Geographic State:** Validates if regional guidelines apply to the farmer's state.
3. **Conflict Arbitration:**
   - Tier 1 statutory guidance (ICAR / CIBRC / IMD) arbitrates over state or district advisories.
   - If two Tier-1 guidelines offer differing controls, the system presents cultural and biological controls first, flagging chemical alternatives for KVK review.

