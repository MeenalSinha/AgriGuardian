"""End-to-end integration tests for weather intelligence across AgriGuardian services."""

from unittest.mock import MagicMock, patch
import pytest
from backend.app.schemas.conversation import ConversationSession
from backend.app.schemas.recommendation import PersonalizedRecommendation
from backend.app.services.advisory.orchestrator import AgriGuardianSystem
from backend.app.services.conversation.engine import ConversationEngine


@pytest.fixture
def mock_open_meteo_hot():
    """Mock weather response simulating 37.5°C heat in Meerut."""
    return {
        "current": {
            "time": "2026-11-09T12:00:00Z",
            "temperature_2m": 37.5,
            "relative_humidity_2m": 45.0,
            "dew_point_2m": 22.0,
            "apparent_temperature": 40.0,
            "precipitation": 0.0,
            "weather_code": 0,
            "wind_speed_10m": 12.0,
        },
        "daily": {
            "time": ["2026-11-09", "2026-11-10", "2026-11-11"],
            "temperature_2m_max": [38.0, 39.0, 37.5],
            "temperature_2m_min": [24.0, 25.0, 24.5],
            "precipitation_sum": [0.0, 0.0, 0.0],
            "weather_code": [0, 0, 0],
            "wind_speed_10m_max": [14.0, 15.0, 12.0],
            "et0_fao_evapotranspiration": [5.5, 5.8, 5.4],
        },
    }


def test_system_process_request_with_weather_intelligence(test_settings, mock_open_meteo_hot):
    """Verify AgriGuardianSystem.process_request generates weather report, risks, and alerts."""
    system = AgriGuardianSystem(settings=test_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_open_meteo_hot

    with patch("requests.get", return_value=mock_resp):
        res = system.process_request(
            farmer_name="Suresh Verma",
            crop="tomato",
            location="Meerut, Uttar Pradesh",
            symptoms="flowers dropping in hot weather",
            lat=28.9845,
            lon=77.7064,
        )

        assert "weather_report" in res
        assert "weather_risk_assessment" in res
        assert "weather_alerts" in res

        # Check that heat stress alert was generated
        alerts = res["weather_alerts"]
        assert len(alerts) >= 1
        assert any(a["hazard_type"] == "heat_stress" for a in alerts)

        # Check recommendation precautions received the weather alert warning
        rec = res["recommendation"]
        assert isinstance(rec, dict)
        precautions = rec.get("precautions", [])
        assert any("weather" in p.lower() or "afternoon" in p.lower() or "spray" in p.lower() for p in precautions)


def test_conversation_engine_weather_turn(test_settings, mock_open_meteo_hot):
    """Verify ConversationEngine handles weather queries with embedded weather alerts and context."""
    system = AgriGuardianSystem(settings=test_settings)
    engine = ConversationEngine(system=system, settings=test_settings)
    session = ConversationSession(session_id="SESS-WEATHER-01", farmer_name="Ramesh", language="hi")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_open_meteo_hot

    with patch("requests.get", return_value=mock_resp):
        turn, _ = engine.process_turn(
            session=session,
            user_query="क्या कल बहुत गर्मी होगी? क्या मुझे खेत में पानी देना चाहिए?",
            crop="Tomato",
            location="Meerut",
            synthesize_voice=False,
        )

        assert turn.intent == "weather"
        assert "weather_alerts" in turn.context_used
        assert turn.context_used["weather_risk_level"] in ("high", "extreme", "moderate")
        assert len(session.turns) == 1

