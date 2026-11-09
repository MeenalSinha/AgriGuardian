"""Unit tests for FastAPI REST API endpoints."""

import pytest
from backend.app.main import (
    root,
    health_check,
    get_languages,
    get_crop_knowledge,
    get_dashboard_metrics,
    get_disease_classes,
    get_model_info,
    generate_advisory,
    process_conversation_turn,
    AdvisoryGenerationRequest,
    ConversationTurnRequest,
)


def test_api_root():
    """Verify API root metadata endpoint."""
    data = root()
    assert data["app_name"] == "AgriGuardian"
    assert data["status"] == "online"
    assert "gnani_models" in data


def test_api_health():
    """Verify health check endpoint."""
    data = health_check()
    assert data["status"] == "healthy"
    assert "supported_languages_count" in data


def test_api_languages():
    """Verify supported language catalog endpoint."""
    data = get_languages()
    assert "languages" in data
    assert "hi-IN" in data["languages"]
    assert "en-IN" in data["languages"]
    assert "voices" in data


def test_api_knowledge_valid_crop():
    """Verify ICAR crop knowledge retrieval endpoint."""
    data = get_crop_knowledge("wheat")
    assert "Wheat" in data["name"]
    assert data["season"] == "Rabi"
    assert "critical_irrigation_stages" in data


def test_api_knowledge_invalid_crop():
    """Verify 404 response for unknown crop."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        get_crop_knowledge("nonexistent_crop_xyz")
    assert exc_info.value.status_code == 404


def test_api_dashboard_metrics():
    """Verify field officer analytics endpoint."""
    data = get_dashboard_metrics()
    assert "risk_distribution" in data
    assert "crop_wise_issues" in data


def test_api_advisory_generation():
    """Verify complete advisory generation via API function."""
    req = AdvisoryGenerationRequest(
        farmer_name="सुरेश कुमार",
        crop="Wheat",
        location="Meerut, UP",
        symptoms="Yellowing leaves after rain",
        language="hi",
    )
    data = generate_advisory(req)
    assert data["farmer_name"] == "सुरेश कुमार"
    assert data["crop"] == "Wheat"
    assert data["risk"] in ("Low", "Medium", "High")
    assert len(data["three_step_plan"]) >= 1


def test_api_conversation_turn():
    """Verify conversational turn processing via API function."""
    req = ConversationTurnRequest(
        farmer_name="अनिता",
        user_query="टमाटर में कीड़ा लगा है",
        crop="Tomato",
        language_code="hi-IN",
        input_type="text",
        synthesize_voice=False,
    )
    data = process_conversation_turn(req)
    assert "session_id" in data
    assert data["turn"]["role"] == "user"
    assert data["turn"]["intent"] == "disease"
    assert data["turn"]["response_text"] is not None


def test_api_disease_classes():
    """Verify endpoint listing 10 supported tomato disease classes."""
    data = get_disease_classes()
    assert data["total_classes"] == 10
    assert len(data["classes"]) == 10
    class_ids = [c["id"] for c in data["classes"]]
    assert "Tomato___Bacterial_spot" in class_ids
    assert "Tomato___healthy" in class_ids


def test_api_disease_model_info():
    """Verify endpoint returning model manifest and checksum."""
    data = get_model_info()
    assert data["architecture"] == "MobileNetV3-Small"
    assert "file_metadata" in data
    assert "sha256_checksum" in data["file_metadata"]
