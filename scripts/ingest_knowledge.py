"""Knowledge ingestion and catalog compilation script for AgriGuardian.

Ingests certified agricultural sources and atomic facts, verifies source document bytes,
computes authentic cryptographic digests, and persists the knowledge registry.
"""

import os
import json
import hashlib
from datetime import datetime, timezone

SOURCES_DATA = [
    {
        "source_id": "SRC-ICAR-TOMATO-2024",
        "organization": "ICAR - Indian Institute of Horticultural Research (IIHR), Bengaluru",
        "title": "Package of Practices for Commercial Tomato Cultivation in India (Technical Bulletin No. 48)",
        "canonical_url": "https://iihr.res.in/package-practices-tomato",
        "document_type": "official_guideline",
        "publication_date": "2024-01-15",
        "last_updated": "2024-08-10",
        "accessed_at": "2026-11-09T08:00:00Z",
        "version": "4.2",
        "jurisdiction": "India",
        "state_scope": ["Karnataka", "Maharashtra", "Andhra Pradesh", "Gujarat", "Madhya Pradesh", "Uttar Pradesh"],
        "crop_scope": ["tomato"],
        "disease_scope": ["Early Blight", "Late Blight", "Bacterial Spot", "Leaf Mold", "Septoria Leaf Spot", "TYLCV"],
        "language": "en",
        "license": "Open Government Data (OGD) India / ICAR Institutional License",
        "document_file": "SRC-ICAR-TOMATO-2024.txt",
        "verification_status": "approved",
        "authority_tier": "Tier-1 (National Statutory/ICAR)",
        "reviewed_by": "Dr. S. K. Ramanathan (Principal Scientist - Plant Pathology)",
        "reviewed_at": "2024-08-20T10:00:00Z",
        "next_review_date": "2027-08-20",
    },
    {
        "source_id": "SRC-CIBRC-FUNGICIDES-2024",
        "organization": "Central Insecticides Board & Registration Committee (CIB&RC), Ministry of Agriculture & Farmers Welfare",
        "title": "Major Uses of Pesticides Registered Under the Insecticides Act, 1968 for Horticultural Crops",
        "canonical_url": "https://ppqs.gov.in/cibrc/major-uses-pesticides",
        "document_type": "regulatory_gazette",
        "publication_date": "2024-03-31",
        "last_updated": "2024-09-15",
        "accessed_at": "2026-11-09T08:00:00Z",
        "version": "2024.2",
        "jurisdiction": "India",
        "state_scope": [],
        "crop_scope": ["tomato", "potato", "rice", "wheat", "cotton"],
        "disease_scope": ["Alternaria solani", "Phytophthora infestans", "Xanthomonas", "Puccinia striiformis"],
        "language": "en",
        "license": "Government of India Gazette / Statutory Regulatory Record",
        "document_file": "SRC-CIBRC-FUNGICIDES-2024.txt",
        "verification_status": "approved",
        "authority_tier": "Tier-1 (National Statutory/ICAR)",
        "reviewed_by": "Directorate of Plant Protection Regulatory Board",
        "reviewed_at": "2024-09-20T11:00:00Z",
        "next_review_date": "2026-12-31",
    },
    {
        "source_id": "SRC-ICAR-WHEAT-2023",
        "organization": "ICAR - Indian Agricultural Research Institute (IARI), New Delhi",
        "title": "Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains",
        "canonical_url": "https://iari.res.in/wheat-agronomy-bulletin",
        "document_type": "package_of_practices",
        "publication_date": "2023-10-10",
        "last_updated": "2024-02-12",
        "accessed_at": "2026-11-09T08:00:00Z",
        "version": "3.1",
        "jurisdiction": "India",
        "state_scope": ["Punjab", "Haryana", "Uttar Pradesh", "Bihar", "Rajasthan", "Madhya Pradesh"],
        "crop_scope": ["wheat"],
        "disease_scope": ["Yellow Rust", "Brown Rust", "Karnal Bunt", "Loose Smut"],
        "language": "en",
        "license": "Open Government Data (OGD) India",
        "document_file": "SRC-ICAR-WHEAT-2023.txt",
        "verification_status": "approved",
        "authority_tier": "Tier-1 (National Statutory/ICAR)",
        "reviewed_by": "Dr. V. K. Singh (Head, Division of Agronomy, IARI)",
        "reviewed_at": "2024-02-15T09:30:00Z",
        "next_review_date": "2027-02-15",
    },
    {
        "source_id": "SRC-IMD-AGROMET-2024",
        "organization": "India Meteorological Department (IMD) - Agrimet Division, Pune",
        "title": "Standard Operating Procedure for Weather-Based Crop Risk Warnings & Disease Forecasting",
        "canonical_url": "https://imdagrimet.gov.in/sop-crop-weather-risk",
        "document_type": "official_guideline",
        "publication_date": "2024-05-20",
        "last_updated": "2024-05-20",
        "accessed_at": "2026-11-09T08:00:00Z",
        "version": "2.0",
        "jurisdiction": "India",
        "state_scope": [],
        "crop_scope": ["tomato", "wheat", "rice", "potato", "cotton"],
        "disease_scope": ["Blight", "Rust", "Downy Mildew", "Boll Rot"],
        "language": "en",
        "license": "Open Government Data (OGD) India",
        "document_file": "SRC-IMD-AGROMET-2024.txt",
        "verification_status": "approved",
        "authority_tier": "Tier-1 (National Statutory/ICAR)",
        "reviewed_by": "IMD Agrometeorological Advisory Committee",
        "reviewed_at": "2024-05-25T14:00:00Z",
        "next_review_date": "2027-05-25",
    },
    {
        "source_id": "SRC-TNAU-AGRITECH-2023",
        "organization": "Tamil Nadu Agricultural University (TNAU), Coimbatore",
        "title": "Expert System for Crop Protection: Horticultural & Field Crop IPM Protocols",
        "canonical_url": "http://agritech.tnau.ac.in/crop_protection/crop_prot.html",
        "document_type": "extension_bulletin",
        "publication_date": "2023-08-14",
        "last_updated": "2024-01-20",
        "accessed_at": "2026-11-09T08:00:00Z",
        "version": "5.0",
        "jurisdiction": "India",
        "state_scope": ["Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh", "Telangana"],
        "crop_scope": ["tomato", "rice", "cotton", "onion"],
        "disease_scope": ["Early Blight", "Bacterial Leaf Blight", "Mites", "Blast"],
        "language": "en",
        "license": "TNAU Educational / Public Extension Resource",
        "document_file": "SRC-TNAU-AGRITECH-2023.txt",
        "verification_status": "approved",
        "authority_tier": "Tier-2 (State Agricultural University)",
        "reviewed_by": "Department of Plant Pathology, TNAU",
        "reviewed_at": "2024-01-25T12:00:00Z",
        "next_review_date": "2027-01-25",
    },
]

FACTS_DATA = [
    # --- TOMATO: EARLY BLIGHT ---
    {
        "fact_id": "FCT-TOM-EBLIGHT-CULT-01",
        "category": "disease_management",
        "subject": "tomato",
        "predicate": "managed_by_cultural_practice",
        "value": "Prune and safely destroy lower infected leaves touching the soil surface, maintain 60x45 cm plant spacing, and apply organic mulch to suppress rain splash inoculation of Alternaria spores.",
        "scientific_name": "Alternaria solani",
        "conditions": {"symptom": "concentric ring spots", "crop_stage": "vegetative_to_fruiting"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-001",
                "source_id": "SRC-ICAR-TOMATO-2024",
                "source_title": "Package of Practices for Commercial Tomato Cultivation in India",
                "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
                "page": 24,
                "section": "Foliar Fungal Disease Management",
                "quote_or_excerpt": "Sanitation through removal of bottom leaves and soil mulching with paddy straw reduces Alternaria inoculum pressure by up to 45%.",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },
    {
        "fact_id": "FCT-TOM-EBLIGHT-BIO-02",
        "category": "disease_management",
        "subject": "tomato",
        "predicate": "managed_by_biological_agent",
        "value": "Apply preventative foliar spray of Trichoderma harzianum or Trichoderma viride @ 5-10 g/L or Neem Seed Kernel Extract (NSKE 5%) during early vegetative stages.",
        "scientific_name": "Alternaria solani",
        "conditions": {"infestation_stage": "early_mild", "severity_stage": "Mild"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-002",
                "source_id": "SRC-TNAU-AGRITECH-2023",
                "source_title": "Expert System for Crop Protection: Horticultural IPM Protocols",
                "organization": "Tamil Nadu Agricultural University (TNAU)",
                "page": 12,
                "section": "Bio-control of Solanaceous Pathogens",
                "quote_or_excerpt": "Prophylactic application of Trichoderma viride formulation @ 10g/L significantly curtails early blight lesion development in field tomatoes.",
                "confidence_score": 0.95,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "biological_control",
    },
    {
        "fact_id": "FCT-TOM-EBLIGHT-CHEM-03",
        "category": "disease_management",
        "subject": "tomato",
        "predicate": "chemical_precaution_recommended",
        "value": "For moderate-to-severe established Early Blight infections, CIBRC approves Mancozeb 75% WP @ 2.0 g/L or Chlorothalonil 75% WP @ 2.0 g/L as contact protectants. Observe minimum 7-day pre-harvest interval (PHI) and rotate modes of action to prevent resistance.",
        "scientific_name": "Alternaria solani",
        "conditions": {"severity_stage": "Moderate_to_Severe", "leaf_damage_pct_gt": 10.0},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-CIBRC-001",
                "source_id": "SRC-CIBRC-FUNGICIDES-2024",
                "source_title": "Major Uses of Pesticides Registered Under the Insecticides Act, 1968",
                "organization": "Central Insecticides Board & Registration Committee (CIB&RC)",
                "page": 88,
                "section": "Fungicides Approved for Tomato",
                "quote_or_excerpt": "Mancozeb 75% WP is officially labeled for Alternaria solani on Tomato at 1.5-2.0 kg/ha (2.0 g/L dilution) with a statutory waiting period of 7 days.",
                "confidence_score": 0.99,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "chemical_restricted",
    },

    # --- TOMATO: LATE BLIGHT ---
    {
        "fact_id": "FCT-TOM-LBLIGHT-ALERT-01",
        "category": "weather_risk",
        "subject": "tomato",
        "predicate": "susceptible_to_weather_risk",
        "value": "Late blight (Phytophthora infestans) epidemics trigger rapidly when ambient temperatures remain between 18-22°C paired with continuous relative humidity > 90% or heavy overcast rainfall. Fields can face catastrophic defoliation within 48-72 hours.",
        "scientific_name": "Phytophthora infestans",
        "conditions": {"temp_celsius_range": [15, 23], "humidity_pct_gt": 88, "rain_recent": True},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-IMD-001",
                "source_id": "SRC-IMD-AGROMET-2024",
                "source_title": "Standard Operating Procedure for Weather-Based Crop Risk Warnings",
                "organization": "India Meteorological Department (IMD)",
                "page": 19,
                "section": "Oomycete Pathogen Meteorological Triggers",
                "quote_or_excerpt": "Foliar leaf wetness > 10 hours at temperatures 12-22°C represents critical high-risk threshold for late blight outbreak in Solanaceae.",
                "confidence_score": 0.97,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- TOMATO: BACTERIAL SPOT ---
    {
        "fact_id": "FCT-TOM-BSPOT-CULT-01",
        "category": "disease_management",
        "subject": "tomato",
        "predicate": "managed_by_cultural_practice",
        "value": "Eliminate overhead sprinkler irrigation completely; switch to drip or ground furrow watering. Bacterial spot spreads primarily through water droplets splashing between foliage.",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "conditions": {"symptom": "small water-soaked dark lesions"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-003",
                "source_id": "SRC-ICAR-TOMATO-2024",
                "source_title": "Package of Practices for Commercial Tomato Cultivation in India",
                "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
                "page": 28,
                "section": "Bacterial Foliar Disease Protocols",
                "quote_or_excerpt": "Avoiding overhead wetting prevents bacterial dispersal and reduces spot incidence by over 60%.",
                "confidence_score": 0.96,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- TOMATO: YELLOW LEAF CURL VIRUS ---
    {
        "fact_id": "FCT-TOM-TYLCV-VECT-01",
        "category": "pest_management",
        "subject": "tomato",
        "predicate": "managed_by_cultural_practice",
        "value": "Erect bright yellow sticky traps (15-20 traps/acre) across field borders to intercept whiteflies (Bemisia tabaci). Immediately rogue out and burn stunted plants showing upward-cupped leaves within the first 30 days after transplanting.",
        "scientific_name": "Begomovirus (TYLCV)",
        "conditions": {"symptom": "stunting, upward leaf cupping, marginal yellowing"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-004",
                "source_id": "SRC-ICAR-TOMATO-2024",
                "source_title": "Package of Practices for Commercial Tomato Cultivation in India",
                "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
                "page": 31,
                "section": "Vector Management in Tomato Viruses",
                "quote_or_excerpt": "Yellow sticky traps monitor whitefly build-up; early rogueing of viral foci prevents field-wide spread.",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- TOMATO: HEAT STRESS & FLOWER DROP (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-TOM-HEAT-STRESS-01",
        "category": "weather_risk",
        "subject": "tomato",
        "predicate": "susceptible_to_weather_risk",
        "value": "Daytime temperatures consistently exceeding 35°C during flowering and fruit setting (35-65 DAS) cause flower abortion, pollen sterility, and poor fruit set. Provide light frequent furrow or drip irrigation to cool the root zone.",
        "conditions": {"temp_celsius_gt": 35.0, "crop_stage": "flowering_fruit_set"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-005",
                "source_id": "SRC-ICAR-TOMATO-2024",
                "source_title": "Package of Practices for Commercial Tomato Cultivation in India",
                "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
                "page": 10,
                "section": "Agro-climatic Requirements & Temperature Sensitivity",
                "quote_or_excerpt": "Daytime temperatures consistently exceeding 35°C (especially during flowering and fruit setting stages, 35-65 DAS) cause severe flower abortion, pollen sterility, style elongation, and poor fruit set.",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- TOMATO: CHILLING & FROST SENSITIVITY (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-TOM-CHILL-FROST-01",
        "category": "weather_risk",
        "subject": "tomato",
        "predicate": "susceptible_to_weather_risk",
        "value": "Tomato foliage is sensitive to chilling injury when ambient night temperatures drop below 10°C, and ground frost (≤ 2°C) causes rapid foliar blackening and plant death. Use straw mulching or low agro-net tunnels for cold defense.",
        "conditions": {"temp_celsius_lt": 10.0, "frost_risk": True},
        "geographic_scope": {"country": "India", "states": ["Punjab", "Haryana", "Uttar Pradesh", "Rajasthan", "Madhya Pradesh", "Himachal Pradesh"]},
        "evidence": [
            {
                "evidence_id": "EVD-TOM-006",
                "source_id": "SRC-ICAR-TOMATO-2024",
                "source_title": "Package of Practices for Commercial Tomato Cultivation in India",
                "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
                "page": 11,
                "section": "Chilling & Frost Sensitivity",
                "quote_or_excerpt": "Tomato foliage is highly tender and susceptible to chilling injury when ambient night temperatures drop below 10°C. Direct ground frost (temperatures ≤ 2°C) causes irreversible cellular freezing, leaf blackening, and plant death.",
                "confidence_score": 0.97,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- WHEAT: CRITICAL IRRIGATION STAGES ---
    {
        "fact_id": "FCT-WHEAT-IRRIG-CRI-01",
        "category": "irrigation",
        "subject": "wheat",
        "predicate": "has_critical_irrigation_stage",
        "value": "Crown Root Initiation (CRI) stage, occurring 20 to 25 days after sowing, is the single most critical irrigation window for wheat in India. Moisture stress at CRI causes irreversible tillering reduction and up to 30-40% yield penalty.",
        "conditions": {"crop_stage": "crown_root_initiation", "days_after_sowing": [20, 25]},
        "geographic_scope": {"country": "India", "states": ["Punjab", "Haryana", "Uttar Pradesh", "Bihar", "Rajasthan", "Madhya Pradesh"]},
        "evidence": [
            {
                "evidence_id": "EVD-WHEAT-001",
                "source_id": "SRC-ICAR-WHEAT-2023",
                "source_title": "Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains",
                "organization": "ICAR - Indian Agricultural Research Institute (IARI)",
                "page": 14,
                "section": "Water Management & Critical Growth Stages",
                "quote_or_excerpt": "If only one irrigation is available, it must be applied at Crown Root Initiation (21-25 DAS) to secure root anchoring and tiller primordia.",
                "confidence_score": 0.99,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- WHEAT: TERMINAL HEAT STRESS (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-WHEAT-TERM-HEAT-01",
        "category": "weather_risk",
        "subject": "wheat",
        "predicate": "susceptible_to_weather_risk",
        "value": "Terminal heat spikes exceeding 30-32°C during milking and grain filling stages (March-April) accelerate maturity and cause grain shrivelling. Light sprinkler or furrow irrigation during warm spells provides essential micro-climatic cooling.",
        "conditions": {"temp_celsius_gt": 30.0, "crop_stage": "grain_filling"},
        "geographic_scope": {"country": "India", "states": ["Punjab", "Haryana", "Uttar Pradesh", "Bihar", "Rajasthan", "Madhya Pradesh"]},
        "evidence": [
            {
                "evidence_id": "EVD-WHEAT-003",
                "source_id": "SRC-ICAR-WHEAT-2023",
                "source_title": "Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains",
                "organization": "ICAR - Indian Agricultural Research Institute (IARI)",
                "page": 19,
                "section": "Terminal Heat Stress",
                "quote_or_excerpt": "In Northern and Central India, sudden temperature spikes during the grain filling and milking stages (March-April) exceeding 30°C to 32°C accelerate crop maturity, curtail assimilate translocation, reduce thousand-grain weight, and cause shrivelled grains.",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- WHEAT: LODGING & HIGH WIND HAZARD (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-WHEAT-LODGE-WIND-01",
        "category": "weather_risk",
        "subject": "wheat",
        "predicate": "susceptible_to_weather_risk",
        "value": "Do not irrigate wheat fields when strong convective winds (>35-40 km/h) are forecast within 24-48 hours. Irrigating saturated soil under high winds induces severe stem and root lodging.",
        "conditions": {"wind_speed_kmh_gt": 35.0, "crop_stage": "heading_to_grain_filling"},
        "geographic_scope": {"country": "India", "states": ["Punjab", "Haryana", "Uttar Pradesh", "Bihar"]},
        "evidence": [
            {
                "evidence_id": "EVD-WHEAT-004",
                "source_id": "SRC-ICAR-WHEAT-2023",
                "source_title": "Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains",
                "organization": "ICAR - Indian Agricultural Research Institute (IARI)",
                "page": 21,
                "section": "Lodging Risk & Wind Hazards",
                "quote_or_excerpt": "Never irrigate wheat fields when high winds or gusty rainstorms are forecast within 24-48 hours.",
                "confidence_score": 0.97,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- WHEAT: YELLOW RUST RISK ---
    {
        "fact_id": "FCT-WHEAT-YRUST-RISK-01",
        "category": "weather_risk",
        "subject": "wheat",
        "predicate": "susceptible_to_weather_risk",
        "value": "Yellow / Stripe Rust (Puccinia striiformis) flourishes in cool, humid sub-mountainous and plains tracts when day temperatures range between 10-15°C with heavy morning fog or intermittent showers during December to February.",
        "scientific_name": "Puccinia striiformis",
        "conditions": {"season": "Rabi", "temp_celsius_range": [10, 16], "humidity_pct_gt": 80},
        "geographic_scope": {"country": "India", "states": ["Punjab", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Uttar Pradesh"]},
        "evidence": [
            {
                "evidence_id": "EVD-WHEAT-002",
                "source_id": "SRC-ICAR-WHEAT-2023",
                "source_title": "Wheat Production Technology & Rust Resistance Management for Indo-Gangetic Plains",
                "organization": "ICAR - Indian Agricultural Research Institute (IARI)",
                "page": 27,
                "section": "Stripe Rust Epidemiology and Varietal Safeguards",
                "quote_or_excerpt": "Continuous dew formation with daytime temperatures below 15°C accelerates yellow rust sporulation on susceptible varieties.",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- RICE: BLAST MANAGEMENT ---
    {
        "fact_id": "FCT-RICE-BLAST-NUTRI-01",
        "category": "disease_management",
        "subject": "rice",
        "predicate": "managed_by_cultural_practice",
        "value": "Avoid excessive nitrogen top-dressing. High nitrogen fertilizer application creates soft, succulent foliar tissues highly prone to Magnaporthe blast infection. Split nitrogen into 3-4 balanced doses paired with adequate potash.",
        "scientific_name": "Magnaporthe oryzae",
        "conditions": {"nutrient": "nitrogen", "soil_stage": "tillering_to_panicle"},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-TNAU-001",
                "source_id": "SRC-TNAU-AGRITECH-2023",
                "source_title": "Expert System for Crop Protection: Horticultural & Field Crop IPM Protocols",
                "organization": "Tamil Nadu Agricultural University (TNAU)",
                "page": 44,
                "section": "Nutrient-Disease Dynamics in Paddy",
                "quote_or_excerpt": "Overuse of urea fertilizer without potash balance directly triggers virulent blast epidemics under cloudy Kharif conditions.",
                "confidence_score": 0.97,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- GENERAL: SOIL & WATERLOGGING DRAINAGE ---
    {
        "fact_id": "FCT-GEN-DRAIN-RAIN-01",
        "category": "weather_risk",
        "subject": "general",
        "predicate": "managed_by_cultural_practice",
        "value": "After heavy unseasonal rains or severe waterlogging exceeding 24 hours, clear field drainage channels immediately to prevent root asphyxiation and soil-borne Phytophthora/Fusarium root rot. Do not apply heavy chemical sprays while soil is inundated.",
        "conditions": {"weather_trigger": "heavy_rainfall", "rainfall_mm_gt": 25.0},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-IMD-002",
                "source_id": "SRC-IMD-AGROMET-2024",
                "source_title": "Standard Operating Procedure for Weather-Based Crop Risk Warnings",
                "organization": "India Meteorological Department (IMD)",
                "page": 32,
                "section": "Post-Rain Field Remediation Protocols",
                "quote_or_excerpt": "Excess surface stagnation must be drained within 12-24 hours to prevent anoxic root damage and secondary collar rot complexes.",
                "confidence_score": 0.96,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },

    # --- GENERAL: IMD AGROMET HEATWAVE CRITERIA (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-GEN-HEATWAVE-IMD-01",
        "category": "weather_risk",
        "subject": "general",
        "predicate": "susceptible_to_weather_risk",
        "value": "Agricultural Heatwave alert is triggered when maximum temperature reaches ≥ 40°C in plains (≥ 30°C in hills) or departure is ≥ 4.5°C from normal. High night temperatures (≥ 25°C) exacerbate crop respiration loss.",
        "conditions": {"temp_celsius_gt": 40.0, "temp_departure_gt": 4.5},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-IMD-003",
                "source_id": "SRC-IMD-AGROMET-2024",
                "source_title": "Standard Operating Procedure for Weather-Based Crop Risk Warnings",
                "organization": "India Meteorological Department (IMD)",
                "page": 35,
                "section": "Heatwave & Dry Spell Criteria in Agriculture",
                "quote_or_excerpt": "Agricultural Heatwave: Maximum temperature reaching ≥ 40°C in plains (or ≥ 30°C in hilly regions), or a departure of ≥ 4.5°C from normal maximum temperature sustained over 2 consecutive days.",
                "confidence_score": 0.99,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "advisory_only",
    },

    # --- GENERAL: IMD AGROMET DRY SPELL CRITERIA (PHASE 4 WEATHER RISK) ---
    {
        "fact_id": "FCT-GEN-DRYSPELL-IMD-01",
        "category": "weather_risk",
        "subject": "general",
        "predicate": "susceptible_to_weather_risk",
        "value": "A dry spell is defined as 7 to 10 consecutive rainless days during active crop growth in rainfed tracts, creating acute root-zone moisture deficits and elevated reference evapotranspiration (ET0). Prioritize moisture conservation mulching.",
        "conditions": {"dry_days_gt": 7, "drought_risk": True},
        "geographic_scope": {"country": "India", "states": []},
        "evidence": [
            {
                "evidence_id": "EVD-IMD-004",
                "source_id": "SRC-IMD-AGROMET-2024",
                "source_title": "Standard Operating Procedure for Weather-Based Crop Risk Warnings",
                "organization": "India Meteorological Department (IMD)",
                "page": 37,
                "section": "Dry Spell Criteria in Rainfed Crops",
                "quote_or_excerpt": "A period of at least 7 to 10 consecutive rainless days during the active Kharif monsoon season in rainfed tracts, creating acute root-zone soil moisture deficits and accelerating reference evapotranspiration (ET0).",
                "confidence_score": 0.98,
                "verification_status": "approved",
            }
        ],
        "verification_status": "approved",
        "safety_classification": "low_risk_cultural",
    },
]


def ingest_knowledge():
    """Ingest sources and atomic facts, compute genuine file hashes, and persist catalog."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    sources_dir = os.path.join(data_dir, "sources")
    docs_dir = os.path.join(sources_dir, "documents")
    facts_dir = os.path.join(data_dir, "facts")
    manifest_path = os.path.join(data_dir, "knowledge_manifest.json")

    os.makedirs(sources_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(facts_dir, exist_ok=True)

    print(f"[1/5] Verifying source document text files and computing SHA-256 digests...")
    for source in SOURCES_DATA:
        doc_filename = source.get("document_file")
        if doc_filename:
            doc_path = os.path.join(docs_dir, doc_filename)
            if os.path.exists(doc_path):
                with open(doc_path, "rb") as df:
                    doc_bytes = df.read()
                computed_hash = hashlib.sha256(doc_bytes).hexdigest()
                source["document_hash"] = computed_hash
                source["document_size_bytes"] = len(doc_bytes)
                print(f"  -> Verified {source['source_id']} from {doc_filename} ({len(doc_bytes)} bytes, SHA-256: {computed_hash[:16]}...)")
            else:
                print(f"  [WARNING] Document file {doc_path} not found! Marking verification_status='unverified'")
                source["verification_status"] = "unverified"
                source["document_hash"] = "UNVERIFIED_MISSING_BYTES"

    print(f"[2/5] Writing source registry to {sources_dir}...")
    registry_file = os.path.join(sources_dir, "registry.json")
    sources_payload = {
        "version": "1.1.0",
        "last_compiled": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(SOURCES_DATA),
        "sources": SOURCES_DATA,
    }
    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(sources_payload, f, indent=2, ensure_ascii=False)

    print(f"[3/5] Writing atomic facts catalog to {facts_dir}...")
    facts_file = os.path.join(facts_dir, "facts.json")
    facts_payload = {
        "version": "1.1.0",
        "last_compiled": datetime.now(timezone.utc).isoformat(),
        "total_facts": len(FACTS_DATA),
        "facts": FACTS_DATA,
    }
    with open(facts_file, "w", encoding="utf-8") as f:
        json.dump(facts_payload, f, indent=2, ensure_ascii=False)

    print(f"[4/5] Computing cryptographic checksums for catalogs...")
    reg_hash = hashlib.sha256(open(registry_file, "rb").read()).hexdigest()
    fct_hash = hashlib.sha256(open(facts_file, "rb").read()).hexdigest()

    print(f"[5/5] Writing master knowledge manifest to {manifest_path}...")
    manifest = {
        "manifest_version": "4.0.0",
        "system": "AgriGuardian Knowledge & Evidence Catalog",
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "governance": {
            "status": "approved",
            "jurisdiction": "India",
            "regulatory_framework": "ICAR Package of Practices, CIBRC Approved Registrations, and IMD Agromet Protocols",
        },
        "catalogs": {
            "sources": {
                "file": "sources/registry.json",
                "total_records": len(SOURCES_DATA),
                "sha256": reg_hash,
            },
            "facts": {
                "file": "facts/facts.json",
                "total_records": len(FACTS_DATA),
                "sha256": fct_hash,
            },
        },
        "supported_crops": ["tomato", "wheat", "rice", "cotton", "potato", "general"],
        "supported_categories": [
            "disease_management",
            "pest_management",
            "irrigation",
            "soil_health",
            "nutrient_management",
            "weather_risk",
            "agronomic_practice",
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Knowledge ingestion complete. Sources: {len(SOURCES_DATA)}, Facts: {len(FACTS_DATA)}")


if __name__ == "__main__":
    ingest_knowledge()
