"""Unit tests for the Dynamic Rule Engine."""

import pytest
from backend.app.services.agriculture.rule_engine import DynamicRuleEngine


def test_rule_engine_initialization(test_rule_engine):
    """Verify rule engine initializes with active rules."""
    assert len(test_rule_engine.rules) >= 4
    rule_ids = [r["id"] for r in test_rule_engine.rules]
    assert "FUNGAL_001" in rule_ids
    assert "WATER_001" in rule_ids


def test_fungal_risk_evaluation(test_rule_engine):
    """Verify that yellowing symptoms with recent rain triggers FUNGAL_001 high risk."""
    symptoms = "The leaves are yellowing and drooping after recent heavy rains."
    weather = {"rain_recent": True, "temp_high": False}

    result = test_rule_engine.evaluate(symptoms, weather)
    assert result["risk"] == "High"
    assert result["rule_count"] >= 1
    triggered_ids = [r["id"] for r in result["triggered_rules"]]
    assert "FUNGAL_001" in triggered_ids


def test_water_stress_evaluation(test_rule_engine):
    """Verify wilting with high temperature triggers WATER_001 high risk."""
    symptoms = "Plants are wilting severely under the scorching sun."
    weather = {"rain_recent": False, "temp_high": True}

    result = test_rule_engine.evaluate(symptoms, weather)
    assert result["risk"] == "High"
    triggered_ids = [r["id"] for r in result["triggered_rules"]]
    assert "WATER_001" in triggered_ids


def test_pest_evaluation(test_rule_engine):
    """Verify pest symptoms trigger PEST_001 rule."""
    symptoms = "Small insects are eating holes in the crop leaves."
    weather = {"rain_recent": False, "temp_high": False}

    result = test_rule_engine.evaluate(symptoms, weather)
    assert result["risk"] in ("Medium", "High")
    triggered_ids = [r["id"] for r in result["triggered_rules"]]
    assert "PEST_001" in triggered_ids


def test_healthy_symptoms_low_risk(test_rule_engine):
    """Verify normal symptoms return Low risk when no rules trigger."""
    symptoms = "Healthy green crop growth, vigorous tillering, good soil conditions."
    weather = {"rain_recent": False, "temp_high": False}

    result = test_rule_engine.evaluate(symptoms, weather)
    assert result["risk"] == "Low"
    assert result["rule_count"] == 0
