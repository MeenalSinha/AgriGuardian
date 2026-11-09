# AgriGuardian — Authoritative Source Registry

This registry records all certified statutory and research publications that supply ground-truth agronomic facts for AgriGuardian. Every source document has been authenticated, archived locally in `data/sources/documents/`, and verified using its genuine SHA-256 cryptographic digest.

---

## 1. Statutory & Institutional Authority Tiers

| Tier | Category | Institutions | Authority Weight |
|:---:|:---|:---|:---:|
| **Tier 1** | **National Statutory & Apex Research** | ICAR (Indian Council of Agricultural Research), CIBRC (Central Insecticides Board & Registration Committee), IMD (India Meteorological Department) | **1.4** |
| **Tier 2** | **State Agricultural Universities (SAUs)** | TNAU, PAU, GBPUAT, UAS Bangalore, Dr. YSRHU | **1.2** |
| **Tier 3** | **District Extension & Krishi Vigyan Kendras** | Local KVK Bulletins, District Agriculture Departments | **1.0** |

---

## 2. Registered Authoritative Sources

### Source 1: `SRC-ICAR-TOMATO-2024`
- **Official Title:** Package of Practices for Commercial Tomato Cultivation in India (Technical Bulletin No. 48)
- **Issuing Organization:** ICAR - Indian Institute of Horticultural Research (IIHR), Bengaluru
- **Authority Tier:** Tier-1 (National Statutory / ICAR)
- **Document Type:** Official Agronomic Guideline (`official_guideline`)
- **Publication Date:** 2024-01-15 | **Next Review Due:** 2027-08-20
- **Canonical URL:** [https://iihr.res.in/package-practices-tomato](https://iihr.res.in/package-practices-tomato)
- **Local Document File:** `data/sources/documents/SRC-ICAR-TOMATO-2024.txt` (4,178 bytes)
- **Cryptographic SHA-256 Digest:** `440549f0cdbe11f0a8227bcfb92d6e32aeec1013898f9dd3f65cb668ca7ae6ff`
- **Jurisdiction:** National (India)
- **State Scope:** Karnataka, Maharashtra, Andhra Pradesh, Gujarat, Madhya Pradesh, Uttar Pradesh
- **Target Crop:** Tomato (*Solanum lycopersicum*)
- **Covered Pathogens & Hazards:** Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*), Bacterial Spot (*Xanthomonas*), TYLCV, Heat Stress (>35°C), Chilling/Frost (<10°C/≤2°C)
- **License:** Open Government Data (OGD) India / ICAR Institutional License
- **Technical Reviewer:** Dr. S. K. Ramanathan (Principal Scientist - Plant Pathology)

---

### Source 2: `SRC-CIBRC-FUNGICIDES-2024`
- **Official Title:** Major Uses of Pesticides Registered Under the Insecticides Act, 1968 for Horticultural Crops
- **Issuing Organization:** Central Insecticides Board & Registration Committee (CIB&RC), Ministry of Agriculture & Farmers Welfare, Government of India
- **Authority Tier:** Tier-1 (National Statutory)
- **Document Type:** Statutory Regulatory Gazette (`regulatory_gazette`)
- **Publication Date:** 2024-03-31 | **Next Review Due:** 2026-12-31
- **Canonical URL:** [https://ppqs.gov.in/cibrc/major-uses-pesticides](https://ppqs.gov.in/cibrc/major-uses-pesticides)
- **Local Document File:** `data/sources/documents/SRC-CIBRC-FUNGICIDES-2024.txt` (3,155 bytes)
- **Cryptographic SHA-256 Digest:** `f081e4faa92887b7f1ad4fa3b07ae7843e498c471c4c8dd9071071c351f0f089`
- **Jurisdiction:** National (All Indian States & UTs)
- **State Scope:** All India
- **Target Crops:** Tomato, Potato, Rice, Wheat, Cotton
- **License:** Government of India Gazette / Statutory Regulatory Record
- **Regulatory Reviewer:** Directorate of Plant Protection Regulatory Board

---

### Source 3: `SRC-ICAR-WHEAT-2023`
- **Official Title:** Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains
- **Issuing Organization:** ICAR - Indian Agricultural Research Institute (IARI), New Delhi
- **Authority Tier:** Tier-1 (National Apex Agronomic Research)
- **Document Type:** Package of Practices (`package_of_practices`)
- **Publication Date:** 2023-10-10 | **Next Review Due:** 2027-02-15
- **Canonical URL:** [https://iari.res.in/wheat-agronomy-bulletin](https://iari.res.in/wheat-agronomy-bulletin)
- **Local Document File:** `data/sources/documents/SRC-ICAR-WHEAT-2023.txt` (2,889 bytes)
- **Cryptographic SHA-256 Digest:** `c3dfeb2633385209772a6b228f411516f8670c3ba96a4ce38ea4ba219198642a`
- **Jurisdiction:** Indo-Gangetic Plains
- **State Scope:** Punjab, Haryana, Uttar Pradesh, Bihar, Rajasthan, Madhya Pradesh
- **Target Crop:** Wheat (*Triticum aestivum*)
- **Covered Challenges:** Yellow Rust, Brown Rust, Crown Root Initiation (CRI) Water Stress, Terminal Heat Stress (>30-32°C), High Wind Lodging (>35 km/h)
- **License:** Open Government Data (OGD) India
- **Technical Reviewer:** Dr. V. K. Singh (Head, Division of Agronomy, IARI)

---

### Source 4: `SRC-IMD-AGROMET-2024`
- **Official Title:** Standard Operating Procedure for Weather-Based Crop Risk Warnings & Disease Forecasting
- **Issuing Organization:** India Meteorological Department (IMD) - Agrimet Division, Pune
- **Authority Tier:** Tier-1 (National Statutory)
- **Document Type:** Official Guideline (`official_guideline`)
- **Publication Date:** 2024-05-20 | **Next Review Due:** 2027-05-25
- **Canonical URL:** [https://imdagrimet.gov.in/sop-crop-weather-risk](https://imdagrimet.gov.in/sop-crop-weather-risk)
- **Local Document File:** `data/sources/documents/SRC-IMD-AGROMET-2024.txt` (3,107 bytes)
- **Cryptographic SHA-256 Digest:** `062e8126cbe100d07db5aa70591eebc95f68c78570267a57a16629ff99a9cfb0`
- **Jurisdiction:** National (India)
- **Target Crops:** Tomato, Wheat, Rice, Potato, Cotton, General Crops
- **Scope:** High-humidity spore proliferation warnings (Late Blight), Agricultural Heatwave criteria (Tmax ≥ 40°C or +4.5°C anomaly), Dry Spell criteria (≥7-10 days), Storm Inundation & Waterlogging drainage
- **License:** Open Government Data (OGD) India
- **Reviewing Body:** IMD Agrometeorological Advisory Committee

---

### Source 5: `SRC-TNAU-AGRITECH-2023`
- **Official Title:** Expert System for Crop Protection: Horticultural & Field Crop IPM Protocols
- **Issuing Organization:** Tamil Nadu Agricultural University (TNAU), Coimbatore
- **Authority Tier:** Tier-2 (State Agricultural University)
- **Document Type:** Extension Bulletin (`extension_bulletin`)
- **Publication Date:** 2023-08-14 | **Next Review Due:** 2027-01-25
- **Canonical URL:** [http://agritech.tnau.ac.in/crop_protection/crop_prot.html](http://agritech.tnau.ac.in/crop_protection/crop_prot.html)
- **Local Document File:** `data/sources/documents/SRC-TNAU-AGRITECH-2023.txt` (2,216 bytes)
- **Cryptographic SHA-256 Digest:** `ac0c60b3352e32ba3dfec07a3311893d1469e3ba2c7e09956ef5a8a1e285a88e`
- **Jurisdiction:** South Indian Agroclimatic Zones
- **State Scope:** Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana
- **Target Crops:** Tomato, Rice, Cotton, Onion
- **Scope:** Integrated Pest Management (IPM), *Trichoderma* biocontrol, whitefly yellow sticky trap barriers, rice blast nitrogen management
- **License:** TNAU Educational / Public Extension Resource
- **Reviewing Body:** Department of Plant Pathology, TNAU
