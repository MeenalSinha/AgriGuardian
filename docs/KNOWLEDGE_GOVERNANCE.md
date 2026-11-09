# AgriGuardian — Knowledge Governance & Safety Compliance Manual

## 1. Principles of Agricultural Knowledge Governance

Agricultural advice carries profound real-world consequences: inaccurate pesticide recommendations can contaminate ground water, cause crop loss, trigger legal liability under the Insecticides Act, or induce acute chemical poisoning among farm workers.

AgriGuardian enforces four inviolable governance principles:
1. **Zero Hallucinated Active Ingredients:** Never generate, invent, or suggest a chemical active ingredient not registered in the CIBRC statutory database.
2. **Authority Primacy:** Tier-1 statutory institutions (ICAR, CIBRC, IMD) hold supreme authority over recommendations.
3. **Traceability:** Every actionable step in an advisory must trace to an `evidence_id` in `data/facts/facts.json`.
4. **Precautionary Escalation:** When foliar visual diagnosis is uncertain, or severity is assessed as Severe, the system MUST refer the farmer to the local Krishi Vigyan Kendra (KVK).

---

## 2. Knowledge Ingestion & Approval Lifecycle

```
[ Raw ICAR / CIBRC Document ] 
             |
             v
   [ Editorial Parsing ] -----> Extract Atomic Facts & Quotes
             |
             v
 [ Specialist Peer Review ] --> Validate Crop Stage, Dosage & Risk
             |
             v
[ Cryptographic Ingestion ] --> Generate SHA-256 Digest & Registry Entry
             |
             v
  [ Active Knowledge Base ] --> Queryable by Retrieval Engine
             |
             v (If past next_review_date or regulatory ban)
   [ Retired / Deprecated ] --> Excluded from Production Retrieval
```

### 2.1 Specialist Review Checklist
Before any atomic fact is granted `"verification_status": "approved"`, a domain specialist must verify:
- [x] Does the fact match the current CIBRC list of approved major uses?
- [x] Are the application instructions cultural or biological wherever possible (IPM hierarchy)?
- [x] Is the chemical intervention classified with `"safety_level": "chemical_restricted"`?
- [x] Are prerequisite conditions (days after sowing, crop stage, humidity) explicitly codified?
- [x] Is the document digest verified against the canonical publication?

---

## 3. Prohibited Chemical Blacklist

AgriGuardian maintains a hard-coded, zero-tolerance block on hazardous chemicals banned or withdrawn under Indian statutory gazettes:

| Prohibited Chemical | Regulatory Status in India | System Policy |
|:---|:---|:---|
| **Monocrotophos** | Banned on vegetables; severe mammalian toxicity | Immediate query interception & refusal |
| **Endosulfan** | Banned pan-India by Supreme Court Order | Immediate query interception & refusal |
| **Paraquat Dichloride** | Highly toxic herbicide, restricted strictly | Intercepted & blocked |
| **Phorate** | Extremely hazardous organophosphate (Class Ia) | Intercepted & blocked |
| **Methyl Parathion** | Banned in India | Intercepted & blocked |
| **Carbofuran** | Highly restricted; banned on foliar vegetable sprays | Downgraded / Intercepted |
| **Dichlorvos** | Restricted use only | Intercepted & blocked |

---

## 4. Conflict Arbitration Protocols

When contradictory recommendations exist between sources:
1. **National vs. Regional:** Tier-1 (ICAR/CIBRC) takes precedence over Tier-2 (State Agr. University) in matters of statutory dosage and banned substances.
2. **State Specificity:** Tier-2 (State Agr. University) takes precedence on localized planting dates and regional varietal resistance.
3. **Biological vs. Chemical:** Prophylactic organic and biological measures (*Trichoderma*, NSKE, crop sanitation) are always prioritized over chemical sprays.
4. **Uncertain Vision Input:** If leaf image confidence is below 75%, chemical protectant actions are automatically downgraded to KVK physical referrals.

---

## 5. Maintenance & Audit Cadence

- **Quarterly Audit:** Automatic scan of all registered sources against `next_review_date`.
- **Statutory Sync:** Bi-annual reconciliation against the Ministry of Agriculture & Farmers Welfare CIBRC gazette updates.
- **Index Rebuild:** Any update to `registry.json` or `facts.json` triggers cryptographic regeneration of `data/knowledge_manifest.json`.
