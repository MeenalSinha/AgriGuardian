"""Unit tests for metric definitions catalog and metric builder."""

import pytest
from backend.app.schemas.analytics import MetricItem, MetricStatus
from backend.app.schemas.farm import UserRole
from backend.app.services.analytics.metric_definitions import (
    METRIC_CATALOG,
    build_metric_item,
    get_metrics_for_role,
)


def test_metric_catalog_completeness():
    """Verify all mandatory metrics are defined across Farmer, Officer, and Admin roles."""
    assert len(METRIC_CATALOG) >= 16

    # Verify key Farmer metrics
    assert "farmer_active_farms" in METRIC_CATALOG
    assert "farmer_total_acres" in METRIC_CATALOG
    assert "farmer_active_crops" in METRIC_CATALOG
    assert "farmer_due_tasks" in METRIC_CATALOG
    assert "farmer_weather_risk_index" in METRIC_CATALOG
    assert "farmer_market_price_bench" in METRIC_CATALOG

    # Verify key Officer metrics
    assert "officer_assigned_districts" in METRIC_CATALOG
    assert "officer_registered_cases" in METRIC_CATALOG
    assert "officer_high_risk_cases" in METRIC_CATALOG
    assert "officer_field_visits_due" in METRIC_CATALOG
    assert "officer_resolution_rate" in METRIC_CATALOG

    # Verify key Admin metrics
    assert "admin_total_users" in METRIC_CATALOG
    assert "admin_active_users_30d" in METRIC_CATALOG
    assert "admin_platform_farms" in METRIC_CATALOG
    assert "admin_system_uptime" in METRIC_CATALOG
    assert "admin_api_error_rate" in METRIC_CATALOG
    assert "admin_data_quality_score" in METRIC_CATALOG


def test_metric_catalog_attributes():
    """Verify each catalog entry contains required operational metadata."""
    for m_id, m_def in METRIC_CATALOG.items():
        assert len(m_def.metric_id) > 0
        assert len(m_def.name) > 0
        assert len(m_def.formula) > 0
        assert len(m_def.data_source) > 0
        assert len(m_def.allowed_roles) > 0
        assert len(m_def.missing_data_behavior) > 0


def test_build_metric_item_normal():
    """Verify standard metric item building with formatting and provenance."""
    item = build_metric_item(
        metric_id="farmer_total_acres",
        value=14.5,
        unit="Acres",
        display_value="14.5 Acres",
        status=MetricStatus.OPTIMAL,
    )
    assert item.metric_id == "farmer_total_acres"
    assert item.value == 14.5
    assert item.unit == "Acres"
    assert item.display_value == "14.5 Acres"
    assert item.status == MetricStatus.OPTIMAL
    assert item.is_available is True
    assert item.missing_reason is None


def test_build_metric_item_missing_data():
    """Verify missing data handling sets unavailable status and honest reason."""
    item = build_metric_item(
        metric_id="farmer_weather_risk_index",
        value=0.0,
        unit="/100",
        display_value="N/A",
        status=MetricStatus.UNAVAILABLE,
        is_available=False,
        missing_reason="Farm has no GPS coordinates recorded.",
    )
    assert item.is_available is False
    assert item.status == MetricStatus.UNAVAILABLE
    assert item.missing_reason == "Farm has no GPS coordinates recorded."
    assert item.display_value == "N/A"


def test_get_metrics_for_role():
    """Verify role filtering returns only authorized metric IDs."""
    farmer_metrics = get_metrics_for_role(UserRole.FARMER)
    assert "farmer_active_farms" in farmer_metrics
    assert "admin_total_users" not in farmer_metrics

    officer_metrics = get_metrics_for_role(UserRole.FIELD_OFFICER)
    assert "officer_registered_cases" in officer_metrics
    assert "admin_api_error_rate" not in officer_metrics

    admin_metrics = get_metrics_for_role(UserRole.ADMINISTRATOR)
    assert "admin_total_users" in admin_metrics
    assert "farmer_active_farms" in admin_metrics
