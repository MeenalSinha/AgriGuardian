"""Schemas for agricultural knowledge source registration and provenance tracking."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    """Schema for verified authoritative agricultural sources."""
    source_id: str = Field(..., description="Unique alphanumeric source identifier (e.g., 'SRC-ICAR-TOMATO-2024')")
    organization: str = Field(..., description="Authoritative issuing organization (e.g., 'ICAR-IIHR Bangalore', 'CIBRC')")
    title: str = Field(..., description="Full official title of publication, gazette, or package of practices")
    canonical_url: Optional[str] = Field(None, description="Official government or institutional repository URL")
    document_type: str = Field(
        ...,
        description="Type of document: 'official_guideline', 'package_of_practices', 'regulatory_gazette', 'extension_bulletin'"
    )
    publication_date: Optional[str] = Field(None, description="Date of original publication (YYYY-MM-DD)")
    last_updated: Optional[str] = Field(None, description="Date of last revision (YYYY-MM-DD)")
    accessed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: Optional[str] = Field("1.0", description="Document edition or bulletin version")
    jurisdiction: str = Field("India", description="Applicable regulatory jurisdiction")
    state_scope: List[str] = Field(default_factory=list, description="Applicable Indian states (empty = national)")
    crop_scope: List[str] = Field(default_factory=list, description="Target crops covered by this document")
    disease_scope: List[str] = Field(default_factory=list, description="Target foliar pathogens and pests covered")
    language: str = Field("en", description="Source document publication language")
    license: Optional[str] = Field("Open Government Data (OGD) India / Public Domain", description="Document copyright license")
    document_hash: str = Field(..., description="SHA-256 cryptographic digest of raw document text")
    verification_status: str = Field(
        "approved",
        description="Governance status: 'approved', 'pending_review', 'retired', 'rejected'"
    )
    authority_tier: str = Field(
        "Tier-1 (National Statutory/ICAR)",
        description="Tier-1 (ICAR/CIBRC/IMD), Tier-2 (State Agr. University), Tier-3 (KVK Extension)"
    )
    reviewed_by: Optional[str] = Field(None, description="Agronomist or specialist reviewer ID")
    reviewed_at: Optional[str] = Field(None, description="Timestamp of peer review approval")
    next_review_date: Optional[str] = Field(None, description="Scheduled expiration / sunset review date")

    @property
    def source_name(self) -> str:
        """Alias for title."""
        return self.title

    @property
    def valid_until(self) -> Optional[str]:
        """Alias for next_review_date."""
        return self.next_review_date

    @property
    def applicable_crops(self) -> List[str]:
        """Alias for crop_scope."""
        return self.crop_scope

    @property
    def reliability_score(self) -> float:
        """Normalized reliability rating based on authority tier."""
        return 1.0 if "Tier-1" in str(self.authority_tier) else 0.85

    @property
    def peer_reviewed(self) -> bool:
        """Check if source is peer reviewed or statutory guideline."""
        return self.document_type in ("official_guideline", "package_of_practices", "regulatory_gazette")

    @property
    def is_active(self) -> bool:
        """Check if source is approved and active."""
        return self.verification_status == "approved"

    @property
    def license_notes(self) -> str:
        """Alias for license."""
        return self.license or "Public Domain"

    @property
    def tier_int(self) -> int:
        """Numeric authority tier (1, 2, or 3)."""
        tier_str = str(self.authority_tier)
        if "1" in tier_str:
            return 1
        if "2" in tier_str:
            return 2
        if "3" in tier_str:
            return 3
        return 1

    def requires_review(self) -> bool:
        """Check if scheduled review date has elapsed."""
        if not self.next_review_date:
            return False
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(self.next_review_date.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > dt
        except Exception:
            return False
