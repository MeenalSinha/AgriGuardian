"""Unit tests for the multi-turn Voice Conversation Engine."""

import pytest
from backend.app.schemas.conversation import ConversationSession
from backend.app.services.conversation.engine import ConversationEngine


def test_intent_classification(test_settings):
    """Verify intent routing for agricultural domain queries."""
    engine = ConversationEngine(settings=test_settings)

    assert engine.classify_intent("Will it rain tomorrow? Should I irrigate?") == "weather"
    assert engine.classify_intent("कल का मौसम कैसा रहेगा?") == "weather"

    assert engine.classify_intent("What is the current market price of wheat?") == "market"
    assert engine.classify_intent("आज मंडी में आलू का क्या भाव है?") == "market"

    assert engine.classify_intent("There are yellow spots on the tomato leaves.") == "disease"
    assert engine.classify_intent("पत्तियों पर कीड़ा लगा हुआ है") == "disease"

    assert engine.classify_intent("When is the best time for sowing?") == "general_agronomy"


def test_process_turn_maintains_history(test_settings):
    """Verify that processing turns accumulates session history."""
    engine = ConversationEngine(settings=test_settings)
    session = ConversationSession(
        farmer_name="सुरेश",
        crop="Cotton",
        language="hi-IN",
    )

    # First turn
    turn1, _ = engine.process_turn(
        session=session,
        user_query="मेरी कपास की फसल मुरझा रही है",
        input_type="text",
        synthesize_voice=False,
    )

    assert len(session.turns) == 1
    assert turn1.confirmed_text == "मेरी कपास की फसल मुरझा रही है"
    assert turn1.response_text is not None

    # Follow-up turn
    turn2, _ = engine.process_turn(
        session=session,
        user_query="क्या मुझे शाम को पानी देना चाहिए?",
        input_type="text",
        synthesize_voice=False,
    )

    assert len(session.turns) == 2
    assert session.turns[0].turn_id != session.turns[1].turn_id
    history = session.get_conversation_history()
    assert len(history) == 4  # 2 user turns + 2 assistant responses

