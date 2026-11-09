"""Unit tests for DataManager persistence and CSV records."""

import os
import pytest
from backend.app.repositories.data_manager import DataManager


def test_data_manager_creates_csv(test_data_manager):
    """Verify CSV is initialized with expected headers."""
    assert os.path.exists(test_data_manager.csv_path)
    df = test_data_manager.get_advisories()
    assert "timestamp" in df.columns
    assert "farmer_name" in df.columns
    assert "risk" in df.columns


def test_save_and_retrieve_advisory(test_data_manager):
    """Verify saving advisory record and retrieving it."""
    sample_advisory = {
        "farmer_name": "सुरेश पटेल",
        "crop": "Cotton",
        "location": "Nagpur",
        "symptoms": "Wilting",
        "risk": "High",
        "advisory_hi": "सिंचाई करें।",
        "advisory_en": "Irrigate the field.",
        "three_step_plan": ["Step 1", "Step 2", "Step 3"],
        "weather_summary": "40°C",
        "market_summary": "₹6500",
        "field_visit_required": True,
        "language": "hi",
    }

    assert test_data_manager.save_advisory(sample_advisory) is True

    df = test_data_manager.get_advisories()
    assert len(df) == 1
    assert df.iloc[0]["farmer_name"] == "सुरेश पटेल"
    assert df.iloc[0]["risk"] == "High"
    assert df.iloc[0]["crop"] == "Cotton"


def test_update_status(test_data_manager):
    """Verify updating advisory review status and feedback."""
    sample = {"farmer_name": "Test", "crop": "Wheat", "risk": "Medium"}
    test_data_manager.save_advisory(sample)

    assert test_data_manager.update_status(0, "completed", "Reviewed by Officer") is True
    df = test_data_manager.get_advisories()
    assert df.iloc[0]["status"] == "completed"
    assert df.iloc[0]["feedback"] == "Reviewed by Officer"

