"""Tests for heterogenous AGMARKNET and e-NAM record normalizer."""

from backend.app.integrations.market.market_normalizer import MarketNormalizer
from backend.app.schemas.market import MarketDataSource, MarketPriceObservation


def test_normalize_agmarknet_record():
    """Verify standard AGMARKNET raw record normalization."""
    raw = {
        "Commodity": "Wheat",
        "Market": "Meerut APMC Mandi",
        "State": "Uttar Pradesh",
        "District": "Meerut",
        "Arrival_Date": "2025-02-15",
        "Min_Price": "2200",
        "Max_Price": "2400",
        "Modal_Price": "2310",
        "Unit": "Quintal",
        "Arrivals": "250.5",
    }

    obs = MarketNormalizer.normalize_record(raw, source=MarketDataSource.AGMARKNET)
    assert obs is not None
    assert isinstance(obs, MarketPriceObservation)
    assert obs.commodity_id == "CMD-WHEAT"
    assert obs.commodity_name == "Wheat"
    assert obs.market_id == "MKT-UP-MEERUT"
    assert obs.modal_price == 2310.0
    assert obs.min_price == 2200.0
    assert obs.max_price == 2400.0
    assert obs.unit == "₹/quintal"
    assert obs.arrivals_tonnes == 250.5


def test_normalize_enam_record_per_kg():
    """Verify e-NAM record reporting in per-kg is automatically scaled to ₹/quintal."""
    raw = {
        "commodity": "Tomato",
        "market": "Kolar APMC Market",
        "state": "Karnataka",
        "district": "Kolar",
        "reporting_date": "2025-02-15",
        "min_price": 18.0,
        "max_price": 24.0,
        "modal_price": 21.0,
        "unit": "Kg",
    }

    obs = MarketNormalizer.normalize_record(raw, source=MarketDataSource.ENAM)
    assert obs is not None
    assert obs.commodity_id == "CMD-TOMATO"
    assert obs.modal_price == 2100.0
    assert obs.min_price == 1800.0
    assert obs.max_price == 2400.0
    assert obs.unit == "₹/quintal"


def test_normalize_unrecognized_commodity_returns_none():
    """Verify unresolvable agricultural commodity returns None without crashing."""
    raw = {
        "Commodity": "Unobtainium Crop",
        "Market": "Meerut",
        "Min_Price": "100",
        "Modal_Price": "200",
        "Max_Price": "300",
    }
    obs = MarketNormalizer.normalize_record(raw)
    assert obs is None
