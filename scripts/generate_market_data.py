"""Generate synthetic calibration fixtures for offline testing and algorithm benchmarks.

WARNING: This script produces synthetic calibration fixtures for testing and demo purposes ONLY.
Synthetic fixtures must NEVER be presented as actual official market transactions in production.
Authoritative official bulletins reside in data/market/official_bulletins.json.
"""

import json
import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

# Add repository root to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.repositories.market_catalog import (
    COMMODITY_MASTER,
    MANDI_MASTER,
    VARIETY_MASTER,
)
from backend.app.schemas.market import MarketDataSource, MarketPriceObservation

# Seed for reproducible, deterministic market time series
random.seed(42)

# Benchmark baseline pricing profiles (₹/quintal) and standard varieties
COMMODITY_PROFILES = {
    "CMD-WHEAT": {
        "base_price": 2320.0,
        "price_volatility": 15.0,
        "min_margin": 80.0,
        "max_margin": 110.0,
        "base_arrivals": 250.0,
        "trend_drift": 0.35,  # slight upward seasonal drift
        "variety_id": "VAR-WHT-FAQ",
        "variety_name": "Mill Quality / FAQ (Common)",
        "mandis": ["MKT-UP-MEERUT", "MKT-PB-LUDHIANA", "MKT-HR-KARNAL", "MKT-RJ-KOTA", "MKT-UP-AGRA", "MKT-UP-VARANASI"],
    },
    "CMD-PADDY": {
        "base_price": 2280.0,
        "price_volatility": 18.0,
        "min_margin": 90.0,
        "max_margin": 120.0,
        "base_arrivals": 320.0,
        "trend_drift": 0.20,
        "variety_id": "VAR-PDY-COMMON",
        "variety_name": "Common Non-Basmati (Grade A)",
        "mandis": ["MKT-HR-KARNAL", "MKT-PB-KHANNA", "MKT-WB-BURDWAN", "MKT-TS-WARANGAL", "MKT-PB-LUDHIANA", "MKT-UP-VARANASI"],
    },
    "CMD-RICE": {
        "base_price": 3650.0,
        "price_volatility": 25.0,
        "min_margin": 120.0,
        "max_margin": 180.0,
        "base_arrivals": 140.0,
        "trend_drift": 0.40,
        "variety_id": "VAR-PDY-COMMON",
        "variety_name": "Common Milled Rice",
        "mandis": ["MKT-UP-MEERUT", "MKT-HR-KARNAL", "MKT-WB-BURDWAN", "MKT-UP-LUCKNOW"],
    },
    "CMD-TOMATO": {
        "base_price": 1950.0,
        "price_volatility": 85.0,  # Highly perishable & volatile
        "min_margin": 180.0,
        "max_margin": 240.0,
        "base_arrivals": 180.0,
        "trend_drift": 0.0,
        "seasonal_amplitude": 450.0,
        "variety_id": "VAR-TOM-HYBRID",
        "variety_name": "Hybrid / Commercial Red",
        "mandis": ["MKT-KA-KOLAR", "MKT-TN-MADURAI", "MKT-MH-NASHIK", "MKT-UP-MEERUT", "MKT-KA-BANGALORE"],
    },
    "CMD-POTATO": {
        "base_price": 1420.0,
        "price_volatility": 20.0,
        "min_margin": 70.0,
        "max_margin": 95.0,
        "base_arrivals": 450.0,
        "trend_drift": 0.15,
        "variety_id": "VAR-POT-JYOTI",
        "variety_name": "Kufri Jyoti / Table White",
        "mandis": ["MKT-UP-AGRA", "MKT-UP-MEERUT", "MKT-MP-INDORE", "MKT-UP-BAREILLY"],
    },
    "CMD-ONION": {
        "base_price": 2450.0,
        "price_volatility": 65.0,  # Sensitive to storage & arrivals
        "min_margin": 150.0,
        "max_margin": 210.0,
        "base_arrivals": 380.0,
        "trend_drift": 0.25,
        "seasonal_amplitude": 320.0,
        "variety_id": "VAR-ONN-RED",
        "variety_name": "Red Onion (Nasik / Garwa)",
        "mandis": ["MKT-MH-LASALGAON", "MKT-MH-PIMPALGAON", "MKT-MH-NASHIK", "MKT-MP-INDORE", "MKT-GJ-GONDAL"],
    },
    "CMD-COTTON": {
        "base_price": 7180.0,
        "price_volatility": 45.0,
        "min_margin": 200.0,
        "max_margin": 260.0,
        "base_arrivals": 190.0,
        "trend_drift": 0.50,
        "variety_id": "VAR-COT-MEDIUM",
        "variety_name": "Medium Staple FAQ",
        "mandis": ["MKT-GJ-RAJKOT", "MKT-GJ-GONDAL", "MKT-TS-WARANGAL", "MKT-MH-NAGPUR"],
    },
    "CMD-MAIZE": {
        "base_price": 2180.0,
        "price_volatility": 18.0,
        "min_margin": 75.0,
        "max_margin": 105.0,
        "base_arrivals": 210.0,
        "trend_drift": 0.20,
        "variety_id": "VAR-MAIZE-YELLOW",
        "variety_name": "Yellow FAQ Maize",
        "mandis": ["MKT-BR-GULABBAGH", "MKT-MP-INDORE", "MKT-KA-KOLAR", "MKT-UP-BAREILLY"],
    },
    "CMD-MUSTARD": {
        "base_price": 5580.0,
        "price_volatility": 35.0,
        "min_margin": 160.0,
        "max_margin": 220.0,
        "base_arrivals": 160.0,
        "trend_drift": 0.30,
        "variety_id": "VAR-MUST-BLACK",
        "variety_name": "Mustard Bold Black FAQ",
        "mandis": ["MKT-RJ-JAIPUR", "MKT-RJ-KOTA", "MKT-UP-AGRA", "MKT-HR-KURUKSHETRA"],
    },
    "CMD-SOYBEAN": {
        "base_price": 4720.0,
        "price_volatility": 30.0,
        "min_margin": 140.0,
        "max_margin": 190.0,
        "base_arrivals": 280.0,
        "trend_drift": 0.10,
        "variety_id": "VAR-SOY-YELLOW",
        "variety_name": "Yellow Soybean FAQ",
        "mandis": ["MKT-MP-INDORE", "MKT-MP-UJJAIN", "MKT-MH-NAGPUR", "MKT-MH-NASHIK"],
    },
    "CMD-GRAM": {
        "base_price": 5750.0,
        "price_volatility": 28.0,
        "min_margin": 150.0,
        "max_margin": 200.0,
        "base_arrivals": 120.0,
        "trend_drift": 0.40,
        "variety_id": "VAR-GRAM-DESI",
        "variety_name": "Desi Chana FAQ",
        "mandis": ["MKT-MP-INDORE", "MKT-GJ-RAJKOT", "MKT-RJ-JAIPUR", "MKT-MP-UJJAIN"],
    },
    "CMD-SUGARCANE": {
        "base_price": 350.0,
        "price_volatility": 2.0,
        "min_margin": 10.0,
        "max_margin": 15.0,
        "base_arrivals": 850.0,
        "trend_drift": 0.05,
        "variety_id": "VAR-SUG-CO0238",
        "variety_name": "Early Maturing Co-0238 (10.25% Recovery)",
        "mandis": ["MKT-UP-MEERUT", "MKT-UP-BAREILLY", "MKT-MH-NASHIK"],
    },
}


def generate_market_dataset(days_back: int = 120, output_path: str = "data/fixtures/market/synthetic_demo_prices.json"):
    """Generate calibration/demo time-series fixtures covering `days_back` days."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    observations = []
    
    for cmd_id, profile in COMMODITY_PROFILES.items():
        cmd_rec = COMMODITY_MASTER.get(cmd_id)
        if not cmd_rec:
            continue
        
        for market_id in profile["mandis"]:
            mandi_rec = MANDI_MASTER.get(market_id)
            if not mandi_rec:
                continue
            
            # Mandi-specific fixed price differential (basis spread)
            random_offset = (hash(market_id + cmd_id) % 21 - 10) * 4.0
            current_modal = profile["base_price"] + random_offset
            
            for day_offset in range(days_back + 1):
                obs_date = start_date + timedelta(days=day_offset)
                
                # Exclude Sundays (APMCs typically closed or skeletal trade)
                if obs_date.weekday() == 6:
                    continue
                
                date_str = obs_date.isoformat()
                
                # Daily walk with mean-reversion and seasonal component
                seasonal = 0.0
                if "seasonal_amplitude" in profile:
                    seasonal = profile["seasonal_amplitude"] * math.sin(2 * math.pi * day_offset / 60.0)
                
                shock = random.gauss(0, profile["price_volatility"])
                drift = profile.get("trend_drift", 0.0)
                
                current_modal += drift + shock * 0.4
                current_modal = max(profile["base_price"] * 0.4, min(profile["base_price"] * 2.5, current_modal))
                
                effective_modal = round(current_modal + seasonal, 2)
                
                # Spread min and max around modal
                min_p = round(effective_modal - random.uniform(profile["min_margin"] * 0.7, profile["min_margin"] * 1.3), 2)
                max_p = round(effective_modal + random.uniform(profile["max_margin"] * 0.7, profile["max_margin"] * 1.3), 2)
                min_p = max(10.0, min_p)
                max_p = max(min_p, max_p)
                
                # Arrivals with inverse volume-price relationship
                arrivals_noise = random.gauss(0, profile["base_arrivals"] * 0.15)
                daily_arrivals = max(5.0, round(profile["base_arrivals"] + arrivals_noise, 1))
                daily_traded = round(daily_arrivals * random.uniform(0.82, 0.98), 1)
                
                obs_id = f"FIXTURE-{mandi_rec.market_id}-{cmd_id}-{date_str}"
                
                obs = MarketPriceObservation(
                    observation_id=obs_id,
                    commodity_id=cmd_id,
                    commodity_name=cmd_rec.canonical_name,
                    variety_id=profile.get("variety_id"),
                    variety_name=profile.get("variety_name", "Standard / FAQ"),
                    market_id=mandi_rec.market_id,
                    market_name=mandi_rec.market_name,
                    state=mandi_rec.state,
                    district=mandi_rec.district,
                    reporting_date=date_str,
                    min_price=min_p,
                    max_price=max_p,
                    modal_price=effective_modal,
                    unit="₹/quintal",
                    arrivals_tonnes=daily_arrivals,
                    traded_quantity_tonnes=daily_traded,
                    source="SYNTHETIC_SIMULATION_FIXTURE (Calibration / Demo Benchmark)",
                    retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                    is_live=False,
                    data_quality_status="synthetic_demo_fixture",
                )
                observations.append(obs.model_dump())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(observations, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully generated {len(observations)} synthetic calibration fixture records -> {output_path}")


if __name__ == "__main__":
    generate_market_dataset()
