"""Integration tests for multimodal voice assistant with image leaf diagnosis."""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest

from backend.app.schemas.conversation import ConversationSession
from backend.app.services.conversation.engine import ConversationEngine


@pytest.fixture
def test_leaf_photo():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    arr = np.zeros((250, 250, 3), dtype=np.uint8)
    arr[:, :] = [40, 160, 45]  # Green
    arr[70:120, 70:120] = [80, 45, 20]  # Lesion
    img = Image.fromarray(arr)
    img.save(path, format="JPEG")

    yield path

    if os.path.exists(path):
        os.remove(path)


def test_multimodal_turn_with_image(test_leaf_photo):
    """Ensure conversation engine incorporates leaf photo diagnosis into dialogue turn."""
    engine = ConversationEngine()
    session = ConversationSession(
        session_id="test_multimodal_sess_1",
        farmer_name="Ramesh",
        crop="Tomato",
        language="hi-IN",
    )

    turn, tts_result = engine.process_turn(
        session=session,
        user_query="मेरी टमाटर की फसल के पत्तों पर यह धब्बा क्या है?",
        input_type="voice",
        image_path=test_leaf_photo,
        synthesize_voice=False,
    )

    assert turn.role == "user"
    assert turn.intent == "disease"
    assert "image_diagnosis" in turn.context_used
    diag = turn.context_used["image_diagnosis"]
    assert diag is not None
    assert diag["is_simulated"] is False
    assert len(turn.response_text) > 20

