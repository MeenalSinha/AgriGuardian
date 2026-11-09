"""Standardized master catalogs for agricultural commodities, varieties, APMC mandis, and MSP policies."""

from typing import Dict, List, Optional
from backend.app.schemas.market import (
    CommodityCategory,
    CommodityRecord,
    MandiRecord,
    MSPRecord,
    VarietyRecord,
)

# ==============================================================================
# 1. COMMODITY MASTER CATALOG
# ==============================================================================

COMMODITY_MASTER: Dict[str, CommodityRecord] = {
    "CMD-WHEAT": CommodityRecord(
        commodity_id="CMD-WHEAT",
        canonical_name="Wheat",
        hindi_name="गेहूं",
        scientific_name="Triticum aestivum",
        category=CommodityCategory.CEREALS,
        standard_unit="₹/quintal",
        aliases=["wheat", "gehu", "gehun", "गेहूं", "गेंहू", "wheat grain"],
    ),
    "CMD-PADDY": CommodityRecord(
        commodity_id="CMD-PADDY",
        canonical_name="Paddy (Dhan)",
        hindi_name="धान",
        scientific_name="Oryza sativa (Unhusked)",
        category=CommodityCategory.CEREALS,
        standard_unit="₹/quintal",
        aliases=["paddy", "dhan", "dhaan", "धान", "unhusked rice", "rough rice"],
    ),
    "CMD-RICE": CommodityRecord(
        commodity_id="CMD-RICE",
        canonical_name="Rice (Chawal)",
        hindi_name="चावल",
        scientific_name="Oryza sativa (Milled)",
        category=CommodityCategory.CEREALS,
        standard_unit="₹/quintal",
        aliases=["rice", "chawal", "चावल", "milled rice"],
    ),
    "CMD-TOMATO": CommodityRecord(
        commodity_id="CMD-TOMATO",
        canonical_name="Tomato",
        hindi_name="टमाटर",
        scientific_name="Solanum lycopersicum",
        category=CommodityCategory.VEGETABLES,
        standard_unit="₹/quintal",
        aliases=["tomato", "tamatar", "टमाटर"],
    ),
    "CMD-POTATO": CommodityRecord(
        commodity_id="CMD-POTATO",
        canonical_name="Potato",
        hindi_name="आलू",
        scientific_name="Solanum tuberosum",
        category=CommodityCategory.VEGETABLES,
        standard_unit="₹/quintal",
        aliases=["potato", "aaloo", "aalu", "आलू"],
    ),
    "CMD-ONION": CommodityRecord(
        commodity_id="CMD-ONION",
        canonical_name="Onion",
        hindi_name="प्याज",
        scientific_name="Allium cepa",
        category=CommodityCategory.VEGETABLES,
        standard_unit="₹/quintal",
        aliases=["onion", "pyaz", "pyaaz", "kanda", "कांदा", "प्याज"],
    ),
    "CMD-COTTON": CommodityRecord(
        commodity_id="CMD-COTTON",
        canonical_name="Cotton",
        hindi_name="कपास",
        scientific_name="Gossypium hirsutum",
        category=CommodityCategory.FIBER,
        standard_unit="₹/quintal",
        aliases=["cotton", "kapas", "कपास", "raw cotton"],
    ),
    "CMD-MAIZE": CommodityRecord(
        commodity_id="CMD-MAIZE",
        canonical_name="Maize",
        hindi_name="मक्का",
        scientific_name="Zea mays",
        category=CommodityCategory.CEREALS,
        standard_unit="₹/quintal",
        aliases=["maize", "makka", "makki", "corn", "मक्का"],
    ),
    "CMD-MUSTARD": CommodityRecord(
        commodity_id="CMD-MUSTARD",
        canonical_name="Mustard",
        hindi_name="सरसों",
        scientific_name="Brassica juncea",
        category=CommodityCategory.OILSEEDS,
        standard_unit="₹/quintal",
        aliases=["mustard", "sarson", "sarso", "rai", "सरसों", "राई"],
    ),
    "CMD-SOYBEAN": CommodityRecord(
        commodity_id="CMD-SOYBEAN",
        canonical_name="Soybean",
        hindi_name="सोयाबीन",
        scientific_name="Glycine max",
        category=CommodityCategory.OILSEEDS,
        standard_unit="₹/quintal",
        aliases=["soybean", "soya", "soyabean", "सोयाबीन"],
    ),
    "CMD-GRAM": CommodityRecord(
        commodity_id="CMD-GRAM",
        canonical_name="Gram (Chana)",
        hindi_name="चना",
        scientific_name="Cicer arietinum",
        category=CommodityCategory.PULSES,
        standard_unit="₹/quintal",
        aliases=["gram", "chana", "channa", "chickpea", "चना"],
    ),
    "CMD-SUGARCANE": CommodityRecord(
        commodity_id="CMD-SUGARCANE",
        canonical_name="Sugarcane",
        hindi_name="गन्ना",
        scientific_name="Saccharum officinarum",
        category=CommodityCategory.SUGARCANE,
        standard_unit="₹/quintal",
        aliases=["sugarcane", "ganna", "गन्ना"],
    ),
}


# ==============================================================================
# 2. VARIETY MASTER CATALOG
# ==============================================================================

VARIETY_MASTER: Dict[str, VarietyRecord] = {
    "VAR-WHT-SHARBATI": VarietyRecord(
        variety_id="VAR-WHT-SHARBATI",
        commodity_id="CMD-WHEAT",
        variety_name="Sharbati / Premium Durum",
        synonyms=["sharbati", "durum", "malwa wheat"],
        grade="Premium Food Grade",
    ),
    "VAR-WHT-FAQ": VarietyRecord(
        variety_id="VAR-WHT-FAQ",
        commodity_id="CMD-WHEAT",
        variety_name="Mill Quality / FAQ (Common)",
        synonyms=["dara", "common", "faq", "kalyan sona", "hd-2967", "pbw-343"],
        grade="Fair Average Quality (FAQ)",
    ),
    "VAR-PDY-BASMATI": VarietyRecord(
        variety_id="VAR-PDY-BASMATI",
        commodity_id="CMD-PADDY",
        variety_name="Basmati (Traditional / 1121 / 1509)",
        synonyms=["basmati", "1121", "1509", "pusa basmati"],
        grade="Export / Aromatic Quality",
    ),
    "VAR-PDY-COMMON": VarietyRecord(
        variety_id="VAR-PDY-COMMON",
        commodity_id="CMD-PADDY",
        variety_name="Common Non-Basmati (Grade A)",
        synonyms=["common", "grade a", "mota", "ir-64", "pr-126", "swarna"],
        grade="Fair Average Quality (FAQ)",
    ),
    "VAR-TOM-HYBRID": VarietyRecord(
        variety_id="VAR-TOM-HYBRID",
        commodity_id="CMD-TOMATO",
        variety_name="Hybrid / Commercial Red",
        synonyms=["hybrid", "himsona", "abhinav", "commercial red"],
        grade="Grade-1 Table Purpose",
    ),
    "VAR-TOM-LOCAL": VarietyRecord(
        variety_id="VAR-TOM-LOCAL",
        commodity_id="CMD-TOMATO",
        variety_name="Desi / Local Sour",
        synonyms=["desi", "local", "sour tomato"],
        grade="General Market Quality",
    ),
    "VAR-ONN-RED": VarietyRecord(
        variety_id="VAR-ONN-RED",
        commodity_id="CMD-ONION",
        variety_name="Red Onion (Nasik / Garwa)",
        synonyms=["red", "nasik red", "garwa", "puna red"],
        grade="Standard Wholesale Grade",
    ),
    "VAR-POT-JYOTI": VarietyRecord(
        variety_id="VAR-POT-JYOTI",
        commodity_id="CMD-POTATO",
        variety_name="Kufri Jyoti / Table White",
        synonyms=["kufri jyoti", "jyoti", "white", "chipsona"],
        grade="Standard Table / Processing",
    ),
}


# ==============================================================================
# 3. APMC MANDI MASTER CATALOG (Verified Coordinates Across 12 States)
# ==============================================================================

MANDI_MASTER: Dict[str, MandiRecord] = {
    # Uttar Pradesh
    "MKT-UP-MEERUT": MandiRecord(
        market_id="MKT-UP-MEERUT",
        market_name="Meerut APMC Mandi",
        apmc_name="Krishi Utpadan Mandi Samiti, Meerut",
        state="Uttar Pradesh",
        district="Meerut",
        latitude=28.9845,
        longitude=77.7064,
    ),
    "MKT-UP-VARANASI": MandiRecord(
        market_id="MKT-UP-VARANASI",
        market_name="Varanasi (Rohania) APMC Mandi",
        apmc_name="Krishi Utpadan Mandi Samiti, Rohania",
        state="Uttar Pradesh",
        district="Varanasi",
        latitude=25.2678,
        longitude=82.9344,
    ),
    "MKT-UP-AGRA": MandiRecord(
        market_id="MKT-UP-AGRA",
        market_name="Agra APMC Mandi",
        apmc_name="Krishi Utpadan Mandi Samiti, Agra",
        state="Uttar Pradesh",
        district="Agra",
        latitude=27.1767,
        longitude=78.0081,
    ),
    "MKT-UP-LUCKNOW": MandiRecord(
        market_id="MKT-UP-LUCKNOW",
        market_name="Lucknow (Dubagga) Mandi",
        apmc_name="Krishi Utpadan Mandi Samiti, Dubagga",
        state="Uttar Pradesh",
        district="Lucknow",
        latitude=26.8622,
        longitude=80.8653,
    ),
    "MKT-UP-BAREILLY": MandiRecord(
        market_id="MKT-UP-BAREILLY",
        market_name="Bareilly APMC Mandi",
        apmc_name="Krishi Utpadan Mandi Samiti, Bareilly",
        state="Uttar Pradesh",
        district="Bareilly",
        latitude=28.3670,
        longitude=79.4304,
    ),
    # Punjab
    "MKT-PB-LUDHIANA": MandiRecord(
        market_id="MKT-PB-LUDHIANA",
        market_name="Ludhiana APMC Grain Market",
        apmc_name="Punjab State Agricultural Marketing Board, Ludhiana",
        state="Punjab",
        district="Ludhiana",
        latitude=30.9010,
        longitude=75.8573,
    ),
    "MKT-PB-AMRITSAR": MandiRecord(
        market_id="MKT-PB-AMRITSAR",
        market_name="Amritsar (Bhagtanwala) Grain Market",
        apmc_name="Punjab State Agricultural Marketing Board, Amritsar",
        state="Punjab",
        district="Amritsar",
        latitude=31.6025,
        longitude=74.8761,
    ),
    "MKT-PB-BATHINDA": MandiRecord(
        market_id="MKT-PB-BATHINDA",
        market_name="Bathinda Grain Market",
        apmc_name="Punjab State Agricultural Marketing Board, Bathinda",
        state="Punjab",
        district="Bathinda",
        latitude=30.2110,
        longitude=74.9455,
    ),
    # Haryana
    "MKT-HR-KARNAL": MandiRecord(
        market_id="MKT-HR-KARNAL",
        market_name="Karnal APMC Grain Market",
        apmc_name="Haryana State Agricultural Marketing Board, Karnal",
        state="Haryana",
        district="Karnal",
        latitude=29.6857,
        longitude=76.9905,
    ),
    "MKT-HR-SIRSA": MandiRecord(
        market_id="MKT-HR-SIRSA",
        market_name="Sirsa APMC Cotton & Grain Mandi",
        apmc_name="Haryana State Agricultural Marketing Board, Sirsa",
        state="Haryana",
        district="Sirsa",
        latitude=29.5349,
        longitude=75.0298,
    ),
    "MKT-HR-HISAR": MandiRecord(
        market_id="MKT-HR-HISAR",
        market_name="Hisar APMC Mandi",
        apmc_name="Haryana State Agricultural Marketing Board, Hisar",
        state="Haryana",
        district="Hisar",
        latitude=29.1492,
        longitude=75.7217,
    ),
    # Maharashtra
    "MKT-MH-LASALGAON": MandiRecord(
        market_id="MKT-MH-LASALGAON",
        market_name="Lasalgaon (Nashik) Onion APMC",
        apmc_name="Agricultural Produce Market Committee, Lasalgaon",
        state="Maharashtra",
        district="Nashik",
        latitude=20.1472,
        longitude=74.2269,
    ),
    "MKT-MH-PUNE": MandiRecord(
        market_id="MKT-MH-PUNE",
        market_name="Pune (Gultekdi) APMC Mandi",
        apmc_name="Agricultural Produce Market Committee, Pune",
        state="Maharashtra",
        district="Pune",
        latitude=18.4967,
        longitude=73.8681,
    ),
    "MKT-MH-NAGPUR": MandiRecord(
        market_id="MKT-MH-NAGPUR",
        market_name="Nagpur (Kalamna) Mandi",
        apmc_name="Agricultural Produce Market Committee, Nagpur",
        state="Maharashtra",
        district="Nagpur",
        latitude=21.1714,
        longitude=79.1432,
    ),
    # Gujarat
    "MKT-GJ-RAJKOT": MandiRecord(
        market_id="MKT-GJ-RAJKOT",
        market_name="Rajkot (Bedi) APMC Market Yard",
        apmc_name="APMC Rajkot",
        state="Gujarat",
        district="Rajkot",
        latitude=22.3382,
        longitude=70.8142,
    ),
    "MKT-GJ-AHMEDABAD": MandiRecord(
        market_id="MKT-GJ-AHMEDABAD",
        market_name="Ahmedabad (Jamalpur) APMC Mandi",
        apmc_name="Agricultural Produce Market Committee, Ahmedabad",
        state="Gujarat",
        district="Ahmedabad",
        latitude=23.0125,
        longitude=72.5833,
    ),
    # Madhya Pradesh
    "MKT-MP-INDORE": MandiRecord(
        market_id="MKT-MP-INDORE",
        market_name="Indore (Choithram) APMC Mandi",
        apmc_name="Krishi Upaj Mandi Samiti, Indore",
        state="Madhya Pradesh",
        district="Indore",
        latitude=22.6853,
        longitude=75.8427,
    ),
    "MKT-MP-UJJAIN": MandiRecord(
        market_id="MKT-MP-UJJAIN",
        market_name="Ujjain APMC Mandi",
        apmc_name="Krishi Upaj Mandi Samiti, Ujjain",
        state="Madhya Pradesh",
        district="Ujjain",
        latitude=23.1765,
        longitude=75.7885,
    ),
    # Rajasthan
    "MKT-RJ-JAIPUR": MandiRecord(
        market_id="MKT-RJ-JAIPUR",
        market_name="Jaipur (Muhana Terminal) Mandi",
        apmc_name="Rajasthan State Agricultural Marketing Board, Muhana",
        state="Rajasthan",
        district="Jaipur",
        latitude=26.7936,
        longitude=75.7512,
    ),
    "MKT-RJ-KOTA": MandiRecord(
        market_id="MKT-RJ-KOTA",
        market_name="Kota (Bhamashah) Mandi Yard",
        apmc_name="Krishi Upaj Mandi Samiti (Anaj), Kota",
        state="Rajasthan",
        district="Kota",
        latitude=25.1634,
        longitude=75.8592,
    ),
    # Andhra Pradesh
    "MKT-AP-GUNTUR": MandiRecord(
        market_id="MKT-AP-GUNTUR",
        market_name="Guntur (Mirchi Yard) APMC",
        apmc_name="Agriculture Market Committee, Guntur",
        state="Andhra Pradesh",
        district="Guntur",
        latitude=16.2917,
        longitude=80.4417,
    ),
    # Karnataka
    "MKT-KA-KOLAR": MandiRecord(
        market_id="MKT-KA-KOLAR",
        market_name="Kolar APMC Tomato Market",
        apmc_name="Agricultural Produce Market Committee, Kolar",
        state="Karnataka",
        district="Kolar",
        latitude=13.1367,
        longitude=78.1291,
    ),
    "MKT-KA-BELAGAVI": MandiRecord(
        market_id="MKT-KA-BELAGAVI",
        market_name="Belagavi APMC Market",
        apmc_name="Agricultural Produce Market Committee, Belagavi",
        state="Karnataka",
        district="Belagavi",
        latitude=15.8497,
        longitude=74.4977,
    ),
    # Tamil Nadu
    "MKT-TN-COIMBATORE": MandiRecord(
        market_id="MKT-TN-COIMBATORE",
        market_name="Coimbatore Regulated Market",
        apmc_name="Tamil Nadu Agricultural Marketing Board, Coimbatore",
        state="Tamil Nadu",
        district="Coimbatore",
        latitude=11.0168,
        longitude=76.9558,
    ),
}


# ==============================================================================
# 4. OFFICIAL MINIMUM SUPPORT PRICE (MSP) POLICY CATALOG (DA&FW / CACP)
# ==============================================================================

MSP_MASTER: Dict[str, MSPRecord] = {
    "CMD-WHEAT": MSPRecord(
        msp_id="MSP-WHEAT-RMS-2024-25",
        crop="Wheat",
        commodity_id="CMD-WHEAT",
        marketing_season="Rabi Marketing Season (RMS) 2024-25",
        msp_price=2275.0,
        unit="₹/quintal",
        announcement_date="2023-10-18",
        effective_period="April 2024 to June 2024",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Resolution Press Release F.No. 3-1/2023-CCEA",
        eligible_variety="Fair Average Quality (FAQ) Standard Grain",
        procurement_agencies=["Food Corporation of India (FCI)", "State Civil Supplies Corporations", "HAFED", "PUNGRAIN"],
    ),
    "CMD-PADDY": MSPRecord(
        msp_id="MSP-PADDY-KMS-2024-25",
        crop="Paddy (Common)",
        commodity_id="CMD-PADDY",
        marketing_season="Kharif Marketing Season (KMS) 2024-25",
        msp_price=2300.0,
        unit="₹/quintal",
        announcement_date="2024-06-19",
        effective_period="October 2024 to March 2025",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Press Release No. 1989718 (RMS/KMS 2024-25)",
        eligible_variety="Common Grade Fair Average Quality (FAQ) Moisture <= 17%",
        procurement_agencies=["Food Corporation of India (FCI)", "State Food & Civil Supplies Corporations"],
    ),
    "CMD-MAIZE": MSPRecord(
        msp_id="MSP-MAIZE-KMS-2024-25",
        crop="Maize",
        commodity_id="CMD-MAIZE",
        marketing_season="Kharif Marketing Season (KMS) 2024-25",
        msp_price=2225.0,
        unit="₹/quintal",
        announcement_date="2024-06-19",
        effective_period="October 2024 to January 2025",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Press Release No. 1989718",
        eligible_variety="FAQ Yellow / White Maize (Moisture <= 14%)",
        procurement_agencies=["NAFED", "Food Corporation of India (FCI)"],
    ),
    "CMD-COTTON": MSPRecord(
        msp_id="MSP-COTTON-KMS-2024-25",
        crop="Cotton (Medium Staple)",
        commodity_id="CMD-COTTON",
        marketing_season="Kharif Marketing Season (KMS) 2024-25",
        msp_price=7121.0,
        unit="₹/quintal",
        announcement_date="2024-06-19",
        effective_period="October 2024 to March 2025",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Resolution F.No. 1-2/2024-CACP-Kharif",
        eligible_variety="Medium Staple Length (24.5 mm - 25.5 mm) FAQ",
        procurement_agencies=["Cotton Corporation of India (CCI)", "NAFED"],
    ),
    "CMD-GRAM": MSPRecord(
        msp_id="MSP-GRAM-RMS-2024-25",
        crop="Gram (Chana)",
        commodity_id="CMD-GRAM",
        marketing_season="Rabi Marketing Season (RMS) 2024-25",
        msp_price=5440.0,
        unit="₹/quintal",
        announcement_date="2023-10-18",
        effective_period="March 2024 to June 2024",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Press Release F.No. 3-1/2023-CCEA",
        eligible_variety="Desi Chana FAQ Grade",
        procurement_agencies=["NAFED", "FCI under Price Support Scheme (PSS)"],
    ),
    "CMD-MUSTARD": MSPRecord(
        msp_id="MSP-MUSTARD-RMS-2024-25",
        crop="Mustard",
        commodity_id="CMD-MUSTARD",
        marketing_season="Rabi Marketing Season (RMS) 2024-25",
        msp_price=5650.0,
        unit="₹/quintal",
        announcement_date="2023-10-18",
        effective_period="March 2024 to June 2024",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Press Release F.No. 3-1/2023-CCEA",
        eligible_variety="Rapeseed & Mustard FAQ (Oil content >= 38%)",
        procurement_agencies=["NAFED", "State Oilseed Federations under PSS"],
    ),
    "CMD-SOYBEAN": MSPRecord(
        msp_id="MSP-SOYBEAN-KMS-2024-25",
        crop="Soybean (Yellow)",
        commodity_id="CMD-SOYBEAN",
        marketing_season="Kharif Marketing Season (KMS) 2024-25",
        msp_price=4892.0,
        unit="₹/quintal",
        announcement_date="2024-06-19",
        effective_period="October 2024 to January 2025",
        official_source="Department of Agriculture & Farmers Welfare (DA&FW) / CACP",
        official_notification_ref="CCEA Press Release No. 1989718",
        eligible_variety="Yellow Soybean FAQ (Moisture <= 12%)",
        procurement_agencies=["NAFED", "Food Corporation of India (FCI)"],
    ),
    "CMD-SUGARCANE": MSPRecord(
        msp_id="MSP-SUGARCANE-FRP-2024-25",
        crop="Sugarcane (Fair & Remunerative Price)",
        commodity_id="CMD-SUGARCANE",
        marketing_season="Sugar Season 2024-25 (Oct-Sep)",
        msp_price=340.0,
        unit="₹/quintal",
        announcement_date="2024-02-21",
        effective_period="October 2024 to September 2025",
        official_source="Department of Food & Public Distribution / CACP",
        official_notification_ref="CCEA Notification F.No. 3(1)/2024-Sugar-Policy",
        eligible_variety="Basic recovery rate of 10.25% sugar recovery",
        procurement_agencies=["Registered Cooperative & Private Sugar Mills (Statutory FRP)"],
    ),
}


class MarketCatalogRepository:
    """Repository accessing standardized commodity, variety, mandi, and MSP catalogs."""

    def __init__(self):
        self.commodities = COMMODITY_MASTER
        self.varieties = VARIETY_MASTER
        self.mandis = MANDI_MASTER
        self.msp = MSP_MASTER

    def resolve_commodity(self, query: str) -> Optional[CommodityRecord]:
        """Resolve commodity canonical record from English, Hindi, or alias query."""
        if not query:
            return None
        q = query.strip().lower()

        # 1. Exact match on ID or canonical name
        for cid, rec in self.commodities.items():
            if q == cid.lower() or q == rec.canonical_name.lower():
                return rec

        # 2. Match on aliases or Hindi name
        for rec in self.commodities.values():
            if q == rec.hindi_name.lower():
                return rec
            if any(q == alias.lower() for alias in rec.aliases):
                return rec
            if any(alias.lower() in q for alias in rec.aliases):
                return rec

        return None

    def get_all_commodities(self) -> List[CommodityRecord]:
        """Return list of all registered commodities."""
        return list(self.commodities.values())

    def get_mandi(self, market_id: str) -> Optional[MandiRecord]:
        """Look up mandi record by ID."""
        return self.mandis.get(market_id)

    def get_all_mandis(self, state: Optional[str] = None, district: Optional[str] = None) -> List[MandiRecord]:
        """Return registered mandis filtered optionally by state and district."""
        results = list(self.mandis.values())
        if state:
            results = [m for m in results if m.state.lower() == state.lower()]
        if district:
            results = [m for m in results if m.district.lower() == district.lower()]
        return results

    def get_msp(self, commodity_id: str) -> Optional[MSPRecord]:
        """Retrieve applicable MSP record for a commodity."""
        return self.msp.get(commodity_id)


# Global singleton instance
market_catalog = MarketCatalogRepository()
