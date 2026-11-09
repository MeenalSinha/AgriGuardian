"""Schemas for multi-turn voice and text conversations."""

import uuid
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """Single turn in a farmer-AgriGuardian conversation."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "assistant"]
    input_type: Literal["voice", "text"] = "text"
    original_transcript: Optional[str] = None
    confirmed_text: str
    response_text: Optional[str] = None
    language: str = "hi-IN"
    intent: Optional[str] = None
    speech_provider: Optional[str] = None
    speech_model: Optional[str] = None
    audio_available: bool = False
    audio_file_path: Optional[str] = None
    context_used: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConversationSession(BaseModel):
    """Session container maintaining multi-turn agricultural context."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    farmer_name: str = "Farmer"
    crop: Optional[str] = None
    location: Optional[str] = None
    language: str = "hi-IN"
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a new turn and update session timestamp."""
        self.turns.append(turn)
        self.updated_at = datetime.now().isoformat()

    def get_conversation_history(self, limit: int = 6) -> List[Dict[str, str]]:
        """Get formatted conversation history for LLM prompting."""
        history = []
        for turn in self.turns[-limit:]:
            if turn.confirmed_text:
                history.append({"role": "user", "content": turn.confirmed_text})
            if turn.response_text:
                history.append({"role": "assistant", "content": turn.response_text})
        return history
