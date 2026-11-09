"""Tests for official CACP / DA&FW Minimum Support Price (MSP) policy provider."""

import pytest
from backend.app.integrations.market.msp_provider import MSPProvider
from backend.app.repositories.market_catalog import market_catalog
from backend.app.schemas.market import MSPRecord


@pytest.fixture
def msp_provider():
    return MSPProvider()


def test_msp_retrieval_wheat(msp_provider):
    """Verify official RMS 2024-25 MSP benchmark for wheat."""
    rec = msp_provider.get_msp("wheat")
    assert rec is not None
    assert isinstance(rec, MSPRecord)
    assert rec.commodity_id == "CMD-WHEAT"
    assert rec.msp_price == 2275.0
    assert rec.unit == "₹/quintal"
    assert "Food Corporation of India" in str(rec.procurement_agencies)


def test_msp_strict_paddy_vs_rice_boundary(msp_provider):
    """Verify MSP applies strictly to unhusked Paddy, and Rice queries receive explanatory context."""
    # 1. Paddy query returns official KMS 2024-25 MSP
    paddy_res = msp_provider.compare_price_with_msp("Paddy", current_modal_price=2320.0)
    assert paddy_res["has_msp"] is True
    assert paddy_res["msp_price"] == 2300.0
    assert paddy_res["status"] == "ABOVE_MSP"

    # 2. Milled Rice query explains MSP does not apply directly to milled commercial rice
    rice_res = msp_provider.compare_price_with_msp("Rice", current_modal_price=3650.0)
    assert rice_res["has_msp"] is False
    assert "unhusked Paddy" in rice_res["message"]
    assert rice_res["related_paddy_msp"] == 2300.0


def test_msp_comparison_above_and_below(msp_provider):
    """Verify above-MSP and below-MSP advisory evaluation."""
    # Above MSP
    above_res = msp_provider.compare_price_with_msp("Wheat", current_modal_price=2450.0)
    assert above_res["has_msp"] is True
    assert above_res["status"] == "ABOVE_MSP"
    assert above_res["difference_inr"] > 0
    assert "ABOVE" in above_res["advisory"]

    # Below MSP
    below_res = msp_provider.compare_price_with_msp("Wheat", current_modal_price=2150.0)
    assert below_res["has_msp"] is True
    assert below_res["status"] == "BELOW_MSP"
    assert below_res["difference_inr"] < 0
    assert "BELOW" in below_res["advisory"]
    assert "procurement" in below_res["advisory"].lower()


def test_msp_non_covered_horticultural_crop(msp_provider):
    """Verify crops without statutory MSP (e.g. Tomato, Potato) return explicit non-MSP status."""
    tom_res = msp_provider.compare_price_with_msp("Tomato", current_modal_price=2200.0)
    assert tom_res["has_msp"] is False
    assert "does not fall under statutory" in tom_res["message"]
