"""Pydantic schemas package for AgriGuardian."""

from backend.app.schemas.speech import (
    AudioValidationResult,
    TranscriptionRequest,
    TranscriptionResult,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from backend.app.schemas.conversation import ConversationTurn, ConversationSession
from backend.app.schemas.advisory import (
    WeatherContext,
    MarketContext,
    RuleEvaluationResult,
    ImageAnalysisResult,
    AdvisoryResult,
)
from backend.app.schemas.disease import (
    ImageValidationResult,
    QualityAssessmentResult,
    ClassPrediction,
    SeverityAssessmentResult,
    ManagementGuidelines,
    DiagnosisResult,
)

__all__ = [
    "AudioValidationResult",
    "TranscriptionRequest",
    "TranscriptionResult",
    "SpeechSynthesisRequest",
    "SpeechSynthesisResult",
    "ConversationTurn",
    "ConversationSession",
    "WeatherContext",
    "MarketContext",
    "RuleEvaluationResult",
    "ImageAnalysisResult",
    "AdvisoryResult",
    "ImageValidationResult",
    "QualityAssessmentResult",
    "ClassPrediction",
    "SeverityAssessmentResult",
    "ManagementGuidelines",
    "DiagnosisResult",
]

