# AgriGuardian — Recommendation Engine Specification

## 1. Engine Objective

The `RecommendationEngine` (`backend/app/services/agriculture/recommendation_engine.py`) is AgriGuardian's core decision-making service. It synthesizes multimodal inputs (farmer queries, visual foliar diagnostic results, hyper-local weather conditions, and market prices) and generates evidence-grounded, context-aware agricultural action plans.

---

## 2. Eleven-Step Recommendation Pipeline

```
[ Step 1: Security & Injection Inspection ]
                   |
[ Step 2: Context Extraction & Vision Injection ]
                   |
[ Step 3: Multilingual Knowledge Retrieval (BM25) ]
                   |
[ Step 4: Context Applicability Evaluation ]
                   |
[ Step 5: Conflict Resolution & Tier Arbitration ]
                   |
[ Step 6: Traceable Evidence Extraction ]
                   |
[ Step 7: Grounded Action Item Formulation ]
                   |
[ Step 8: Chemical Action Sanitization ]
                   |
[ Step 9: KVK Expert Escalation Assessment ]
                   |
[ Step 10: Clarifying Question Generation ]
                   |
[ Step 11: Final Confidence Scoring & Packaging ]
```

---

## 3. Step Details

### Step 1: Security & Injection Inspection
- Checks for prompt injection attempts (`ignore previous instructions`, `jailbreak`, etc.).
- Scans for banned agrochemicals (Monocrotophos, Endosulfan, Paraquat).
- If flagged, aborts execution and returns an `insufficient_evidence` response with a security refusal notice.

### Step 2: Context Extraction & Vision Injection
- Extracts crop name, state, crop growth stage, and days after sowing (DAS).
- If a PyTorch MobileNetV3 visual leaf diagnosis is present:
  - Injects detected disease (`common_name`, `confidence`, `severity_stage`).
  - Sets diagnosis status to `confirmed` if confidence $\ge 0.75$, or `suspected` / `uncertain` otherwise.

### Step 3: Multilingual Knowledge Retrieval
- Invokes `RetrievalService.retrieve()`.
- Applies BM25 token overlap scoring with Indic synonym expansion (Hindi/Hinglish/English).
- Ranks candidate atomic facts by relevance and source authority weight.

### Step 4: Context Applicability Evaluation
- Evaluates fact applicability using `ApplicabilityService`:
  - Does the crop match?
  - Does the crop stage match?
  - Do weather conditions meet threshold prerequisites (e.g. humidity > 80% for blight)?
- Missing prerequisite variables are recorded for step 10.

### Step 5: Conflict Resolution & Tier Arbitration
- Identifies competing facts with identical subject and predicate.
- Employs `ConflictResolutionService` to resolve discrepancies based on source tier weights:
  $$\text{Weight}(\text{Tier-1}) = 1.4 > \text{Weight}(\text{Tier-2}) = 1.2 > \text{Weight}(\text{Tier-3}) = 1.0$$

### Step 6: Traceable Evidence Extraction
- Gathers all `EvidenceExcerpt` objects linked to the accepted facts.
- Preserves full citations (source title, organization, page, quote).

### Step 7: Grounded Action Item Formulation
- Converts atomic facts into typed `ActionItem` objects categorized by priority:
  - `immediate`: Cultural sanitation, canopy aeration, urgent drainage.
  - `secondary`: Bio-control sprays (*Trichoderma*, NSKE) or registered chemical protectants.
  - `routine`: Soil maintenance and monitoring.

### Step 8: Chemical Action Sanitization
- Evaluates chemical safety using `SafetyGuardrailService`:
  - If visual diagnosis is NOT confirmed, restricted chemical sprays are downgraded to Krishi Vigyan Kendra (KVK) physical referrals.
  - Enforces PPE requirements and statutory dilution precautions.

### Step 9: KVK Expert Escalation Assessment
- Triggers mandatory field officer referral when:
  - Disease severity is `Severe` or `Moderate-to-Severe`.
  - Leaf diagnosis confidence is low or ambiguous (`Unrecognized Foliar Symptom`).
  - Rapid weather risk (continuous heavy rain + cloud cover) threatens total crop loss.

### Step 10: Clarifying Question Generation
- When essential agronomic variables (e.g., exact sowing date, irrigation type) are missing, generates polite clarifying questions rather than guessing.

### Step 11: Final Confidence Scoring & Packaging
- Assembles the complete `PersonalizedRecommendation` Pydantic payload:
  - `recommendation_id`: Unique tracking ID (`REC-XXXXXXXXXX`).
  - `status`: `completed`, `expert_escalation_required`, or `insufficient_evidence`.
  - `assessment`: Detailed diagnostic and clinical summary.
  - `actions`: Sanitized action plan with evidence linkages.
  - `precautions`: Environmental and handling warnings.
  - `supporting_evidence`: Traceable quotes and citations.
  - `confidence_assessment`: Final qualitative confidence level and rationale.

