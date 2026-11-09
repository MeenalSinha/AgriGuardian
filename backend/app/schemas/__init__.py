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
from backend.app.schemas.source import SourceRecord
from backend.app.schemas.evidence import EvidenceExcerpt
from backend.app.schemas.knowledge import AtomicFact
from backend.app.schemas.recommendation import ActionItem, PersonalizedRecommendation

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
    "SourceRecord",
    "EvidenceExcerpt",
    "AtomicFact",
    "ActionItem",
    "PersonalizedRecommendation",
]

