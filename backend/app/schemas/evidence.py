"""Schemas for traceable evidence excerpts and citation linkages."""

from typing import Optional
from pydantic import BaseModel, Field


class EvidenceExcerpt(BaseModel):
    """Schema for an atomic, traceable text excerpt from an approved agricultural source."""
    evidence_id: str = Field(..., description="Unique evidence excerpt identifier (e.g., 'EVD-TOM-001')")
    source_id: str = Field(..., description="Foreign key reference to registered SourceRecord")
    source_title: str = Field(..., description="Display title of the supporting authoritative source")
    organization: str = Field(..., description="Issuing authority (e.g., 'ICAR - Indian Agricultural Research Institute')")
    page: Optional[int] = Field(None, description="Page number in original document if available")
    section: Optional[str] = Field(None, description="Section heading or chapter title")
    quote_or_excerpt: str = Field(..., description="Verbatim or faithfully summarized factual excerpt")
    confidence_score: float = Field(1.0, description="Confidence in excerpt factual integrity (0.0 to 1.0)")
    verification_status: str = Field("approved", description="'approved', 'pending_review', 'flagged'")

    @property
    def excerpt_text(self) -> str:
        """Alias for quote_or_excerpt."""
        return self.quote_or_excerpt
