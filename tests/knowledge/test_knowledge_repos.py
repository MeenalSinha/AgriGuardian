"""Tests for SourceRepository, FactRepository, and SourceValidationService."""

import pytest
from datetime import date, timedelta
from backend.app.schemas.source import SourceRecord
from backend.app.schemas.evidence import EvidenceExcerpt
from backend.app.schemas.knowledge import AtomicFact
from backend.app.repositories.source_repository import SourceRepository
from backend.app.repositories.fact_repository import FactRepository
from backend.app.services.agriculture.source_validation_service import SourceValidationService


def test_source_repository_loads_registered_sources():
    """Verify that SourceRepository loads authoritative sources correctly."""
    repo = SourceRepository.get_instance()
    sources = repo.get_all()
    assert len(sources) >= 5

    # Check key Tier 1 sources exist
    cibrc = repo.get_by_id("SRC-CIBRC-FUNGICIDES-2024")
    assert cibrc is not None
    assert "Tier-1" in cibrc.authority_tier or cibrc.authority_tier == 1
    assert "CIB" in cibrc.organization
    assert cibrc.reliability_score == 1.0

    icar_iihr = repo.get_by_id("SRC-ICAR-TOMATO-2024")
    assert icar_iihr is not None
    assert "Tier-1" in icar_iihr.authority_tier or icar_iihr.authority_tier == 1
    assert "tomato" in icar_iihr.applicable_crops


def test_source_repository_filtering():
    """Verify filtering by tier, crop, and active status."""
    repo = SourceRepository.get_instance()

    tier1_sources = repo.get_by_tier(1)
    assert len(tier1_sources) >= 3
    assert all(s.tier_int == 1 or "Tier-1" in s.authority_tier for s in tier1_sources)

    tomato_sources = repo.get_by_crop("tomato")
    assert len(tomato_sources) >= 2
    assert all("tomato" in s.applicable_crops or "all" in s.applicable_crops for s in tomato_sources)


def test_fact_repository_loads_facts():
    """Verify FactRepository loads atomic agricultural facts."""
    repo = FactRepository.get_instance()
    facts = repo.get_all()
    assert len(facts) >= 10

    # Test filtering by crop
    tomato_facts = repo.get_by_crop("tomato")
    assert len(tomato_facts) >= 5
    assert all(f.crop.lower() == "tomato" for f in tomato_facts)

    # Test filtering by predicate
    cultural_facts = repo.get_by_predicate("managed_by_cultural_practice")
    assert len(cultural_facts) >= 1
    assert all(f.predicate == "managed_by_cultural_practice" for f in cultural_facts)


def test_fact_evidence_excerpts_grounding():
    """Verify that atomic facts contain verified evidence excerpts with valid source IDs."""
    fact_repo = FactRepository.get_instance()
    source_repo = SourceRepository.get_instance()

    facts = fact_repo.get_all()
    for fact in facts:
        assert len(fact.evidence) > 0
        for ev in fact.evidence:
            assert ev.evidence_id.startswith("EV")
            assert ev.source_id.startswith("SRC-")
            # Verify source actually exists in SourceRepository
            src = source_repo.get_by_id(ev.source_id)
            assert src is not None, f"Source {ev.source_id} referenced by fact {fact.fact_id} not in registry!"


def test_source_validation_service():
    """Verify source validation logic for tiers, expiry, and crop/state scope."""
    validator = SourceValidationService()
    source_repo = SourceRepository.get_instance()

    valid_src = source_repo.get_by_id("SRC-ICAR-TOMATO-2024")
    assert valid_src is not None

    is_valid, reasons = validator.validate_source(valid_src, crop="tomato", state="Karnataka")
    assert is_valid is True
    assert len(reasons) == 0

    # Test crop mismatch
    is_valid, reasons = validator.validate_source(valid_src, crop="sugarcane")
    assert is_valid is False
    assert any("not applicable" in r.lower() for r in reasons)

    # Test expired source
    expired_src = SourceRecord(
        source_id="SRC-TEST-EXPIRED",
        title="Expired Guide",
        organization="Test Org",
        document_type="extension_bulletin",
        authority_tier="Tier-2 (State Agricultural University)",
        document_hash="dummyhash12345678",
        publication_date="2020-01-01",
        next_review_date="2021-01-01",
        crop_scope=["tomato"],
    )
    is_valid, reasons = validator.validate_source(expired_src, crop="tomato")
    assert is_valid is False
    assert any("expired" in r.lower() or "passed" in r.lower() for r in reasons)
