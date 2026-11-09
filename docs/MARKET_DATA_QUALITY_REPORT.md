# AgriGuardian — Agricultural Market Data Quality & Ingestion Report

**Phase 5: Intelligent Agricultural Market Intelligence, Real-Time Mandi Prices, Price Trend Analysis & Predictive Market Insights**  
**Version:** 1.0.0  
**Date:** November 2026  
**Status:** Verified & Operational  

---

## 1. Executive Summary

This report establishes the data quality, integrity protocols, ingestion mechanisms, and empirical coverage for AgriGuardian's Phase 5 Agricultural Market Intelligence module. All legacy static price dictionaries (`INDICATIVE_MARKET_BENCHMARKS`) and hardcoded fallback prices (₹2,000/quintal) have been completely eliminated. Market intelligence is now grounded in official agricultural market sources:
1. **AGMARKNET** (Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare, Government of India).
2. **e-NAM** (National Agriculture Market, Small Farmers' Agri-Business Consortium).
3. **CACP / DA&FW** (Commission for Agricultural Costs & Prices, Department of Agriculture & Farmers Welfare) official Minimum Support Price (MSP) statutory policy notifications.

---

## 2. Authorized Data Sources & Provenance

| Data Source | Managing Authority | Nature of Data | Update Frequency | Quality Tier |
|:---|:---|:---|:---|:---:|
| **AGMARKNET** | Directorate of Marketing & Inspection (DMI), MoA&FW | Daily APMC mandi closing price bulletins (min, modal, max, arrivals) | Daily (EOD closing bulletin) | Primary / Authoritative |
| **e-NAM** | Small Farmers' Agri-Business Consortium (SFAC) | Electronic auction transaction lots, traded quantities, and auction prices | Intraday / Post-session | High (Auction Verified) |
| **CACP / DA&FW MSP** | Cabinet Committee on Economic Affairs (CCEA), GoI | Statutory procurement price benchmarks for mandated crops | Biannual (RMS & KMS gazette notifications) | Statutory Benchmark |
| **State APMC Boards** | State Agricultural Marketing Boards (e.g. Mandi Parishad UP, PSAMB Punjab) | Physical mandi arrivals, weighing slips, and market cess records | Daily | Regional Benchmark |

---

## 3. Reference Master Catalogs

AgriGuardian maintains standardized, validated reference master catalogs in [market_catalog.py](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/repositories/market_catalog.py):

### 3.1 Commodity Catalog (12 Standardized Commodities)
- `CMD-WHEAT`: Wheat (*Triticum aestivum*) — गेहूं
- `CMD-PADDY`: Paddy (*Oryza sativa* Unhusked) — धान
- `CMD-RICE`: Rice (*Oryza sativa* Milled) — चावल
- `CMD-TOMATO`: Tomato (*Solanum lycopersicum*) — टमाटर
- `CMD-POTATO`: Potato (*Solanum tuberosum*) — आलू
- `CMD-ONION`: Onion (*Allium cepa*) — प्याज
- `CMD-COTTON`: Cotton (*Gossypium hirsutum*) — कपास
- `CMD-MAIZE`: Maize (*Zea mays*) — मक्का
- `CMD-MUSTARD`: Mustard (*Brassica juncea*) — सरसों
- `CMD-SOYBEAN`: Soybean (*Glycine max*) — सोयाबीन
- `CMD-GRAM`: Chickpea / Gram (*Cicer arietinum*) — चना
- `CMD-SUGARCANE`: Sugarcane (*Saccharum officinarum*) — गन्ना

### 3.2 APMC Mandi Catalog (25+ Mandis Across 12 States with Verified GPS Coordinates)
All mandis carry verified geographic centroids for straight-line geodesic distance calculation:
- **Uttar Pradesh (5):** Meerut (28.9845, 77.7064), Varanasi/Rohania (25.2678, 82.9344), Agra (27.1767, 78.0081), Lucknow/Dubagga (26.8622, 80.8653), Bareilly (28.3670, 79.4304).
- **Punjab (2):** Ludhiana Grain Market (30.9010, 75.8573), Khanna Asia's Largest Grain Mandi (30.7072, 76.2163).
- **Haryana (2):** Karnal Grain Mandi (29.6857, 76.9905), Kurukshetra/Thanesar (29.9695, 76.8783).
- **Maharashtra (4):** Lasalgaon Premier Onion Mandi (20.1472, 74.2256), Pimpalgaon Baswant (20.1697, 73.9856), Nashik (19.9975, 73.7898), Nagpur Cotton Market (21.1458, 79.0882).
- **Gujarat (2):** Rajkot APMC (22.3039, 70.8022), Gondal APMC (21.9619, 70.7923).
- **Madhya Pradesh (2):** Indore Choithram Mandi (22.7196, 75.8577), Ujjain Madhav Nagar Mandi (23.1765, 75.7885).
- **Karnataka (2):** Kolar Tomato Mandi (13.1367, 78.1292), Yeshwanthpur Bangalore (13.0280, 77.5408).
- **Tamil Nadu (1):** Madurai Mattuthavani Central Market (9.9252, 78.1198).
- **Rajasthan (2):** Kota Bhamashah Mandi (25.2138, 75.8648), Jaipur Muhana Terminal Market (26.9124, 75.7873).
- **Andhra Pradesh (1):** Guntur Mirchi & Grain Yard (16.3067, 80.4365).
- **Telangana (1):** Warangal Enamamula Grain Market (17.9689, 79.5941).
- **West Bengal (1):** Burdwan APMC (23.2324, 87.8615).
- **Bihar (1):** Gulabbagh Purnea (25.7771, 87.4753).
- **Kerala (1):** Kochi Vyttila APMC (9.9674, 76.3180).

---

## 4. Strict Agricultural Integrity Distinctions

### 4.1 Paddy (Unhusked) vs. Milled Rice Distinction
- **Agronomic & Market Reality:** Paddy (*Dhan*, rough unhusked grain) is transacted directly by farmers at APMC yards with 14–17% moisture content and is traded at ₹2,100–₹2,500/quintal.
- **Milled Rice (*Chawal*):** Transacted post-milling by rice millers/processors with bran and husk removed (approx. 67% milling recovery). Milled rice trades at ₹3,200–₹4,500/quintal.
- **MSP Enforcement:** Statutory MSP announced by CCEA applies strictly to Paddy (Common: ₹2,300/qtl; Grade A: ₹2,320/qtl for KMS 2024-25). Rice has no statutory MSP.
- **AgriGuardian Rule:** The system strictly treats `CMD-PADDY` and `CMD-RICE` as distinct commodities with different price bands. If a farmer asks for "Rice MSP", AgriGuardian clarifies that MSP applies to unhusked Paddy and explains the difference.

### 4.2 MSP Statutory Boundary
- AgriGuardian never presents MSP as a guaranteed private transaction price.
- Every MSP presentation carries the mandatory regulatory notice:
  > *"MSP is an indicative government procurement support price benchmark for Fair Average Quality (FAQ) produce at designated state procurement centers (e.g. FCI, NAFED), NOT a guaranteed statutory minimum price enforced at private APMC auctions."*

---

## 5. Data Ingestion, Validation & Normalization Rules

Implemented in [market_data_validator.py](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/integrations/market/market_data_validator.py) and [market_normalizer.py](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/backend/app/integrations/market/market_normalizer.py):

1. **Unit Normalization:**
   - Standard unit: `₹/quintal` (100 kg).
   - Retail per-kg rates: $\times 100.0$.
   - Bulk metric tonne rates: $/ 10.0$.
2. **Logical Price Bounds:**
   - Strict condition: $0 < \text{min\_price} \le \text{modal\_price} \le \text{max\_price}$.
   - Biological plausibility bounds: min price $\ge ₹50$/qtl, max price $\le ₹200,000$/qtl.
3. **Temporal Validity:**
   - Reporting date format: ISO `YYYY-MM-DD`.
   - Distant future dates ($> \text{today}$) are strictly rejected.
4. **Missing Data Policy:**
   - If market data is unavailable for a requested crop in a region, the system explicitly returns `is_available=False` and `current_price=None`. No synthetic fallbacks are permitted.
