"""Tests for RetrievalService with hybrid lexical BM25 scoring and multilingual Indic expansion."""

import pytest
from backend.app.services.agriculture.retrieval_service import RetrievalService


def test_retrieval_english_query():
    """Verify retrieval returns relevant facts for English tomato early blight query."""
    service = RetrievalService()
    results = service.retrieve(
        query="tomato early blight leaf spots",
        crop="tomato",
        top_k=3,
    )

    assert len(results) > 0
    top_fact, score = results[0]
    assert score > 0
    assert "early blight" in top_fact.subject.lower() or "alternaria" in top_fact.value.lower() or "fungal" in top_fact.value.lower()


def test_retrieval_indic_hindi_synonym_expansion():
    """Verify Hindi query 'झुलसा' (blight) expands and retrieves blight facts."""
    service = RetrievalService()
    results = service.retrieve(
        query="टमाटर में झुलसा रोग और पत्तों पर धब्बा",
        crop="tomato",
        top_k=3,
    )

    assert len(results) > 0
    top_fact, score = results[0]
    assert score > 0
    # Fact should be related to blight or leaf spot
    assert any(term in top_fact.value.lower() or term in top_fact.subject.lower() for term in ["blight", "spot", "leaf", "alternaria"])


def test_retrieval_crop_filtering():
    """Verify crop-specific filtering: wheat query retrieves wheat irrigation fact, not tomato."""
    service = RetrievalService()
    results = service.retrieve(
        query="critical irrigation stage",
        crop="wheat",
        top_k=3,
    )

    assert len(results) > 0
    top_fact, score = results[0]
    assert top_fact.crop.lower() == "wheat"
    assert "crown root initiation" in top_fact.value.lower() or "irrigation" in top_fact.predicate.lower()


def test_retrieval_ranking_favors_tier1():
    """Verify that authority weighting elevates Tier 1 statutory facts."""
    service = RetrievalService()
    results = service.retrieve(
        query="fungal disease leaf sanitation copper oxychloride",
        crop="tomato",
        top_k=5,
    )

    assert len(results) > 0
    # High ranked facts should have substantial authority weight (>= 1.2)
    top_fact, _ = results[0]
    assert top_fact.authority_weight >= 1.2
