"""Evidence-grounded agricultural knowledge retrieval service with hybrid scoring and provenance retention."""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from backend.app.core.logging import get_logger
from backend.app.repositories.fact_repository import FactRepository
from backend.app.repositories.source_repository import SourceRepository
from backend.app.schemas.evidence import EvidenceExcerpt
from backend.app.schemas.knowledge import AtomicFact

logger = get_logger("RetrievalService")

# Multilingual agricultural synonym mapping (Hindi / Hinglish / English)
AGRI_SYNONYMS: Dict[str, List[str]] = {
    "tomato": ["टमाटर", "tamatar", "solanum lycopersicum"],
    "wheat": ["गेहूं", "gehun", "kanak", "triticum"],
    "rice": ["धान", "चावल", "paddy", "oryza"],
    "blight": ["झुलसा", "jhulsa", "early blight", "late blight", "alternaria", "phytophthora"],
    "spot": ["धब्बा", "धब्बे", "dhabba", "spots", "leaf spot", "septoria", "bacterial spot"],
    "rust": ["गेरुआ", "रतुआ", "tula", "puccinia", "yellow rust", "stripe rust", "brown rust"],
    "irrigation": ["सिंचाई", "पानी", "water", "watering", "irrigate", "drip", "moisture"],
    "crown root": ["सीआरआई", "cri", "crown root initiation", "tiller", "tillering"],
    "yellowing": ["पीला", "पीली", "पीले", "chlorosis", "yellow"],
    "wilting": ["मुरझाना", "सूखना", "wilt", "droop"],
    "pest": ["कीड़ा", "कीड़े", "कीट", "insect", "mite", "whitefly", "caterpillar"],
}


class RetrievalService:
    """Retrieves verified agricultural facts and evidence excerpts with provenance tracking."""

    def __init__(
        self,
        fact_repo: Optional[FactRepository] = None,
        source_repo: Optional[SourceRepository] = None,
    ):
        self.fact_repo = fact_repo or FactRepository.get_instance()
        self.source_repo = source_repo or SourceRepository.get_instance()

    def _normalize_tokens(self, text: str) -> Set[str]:
        """Tokenize and expand agricultural synonyms across English and Indic terms."""
        cleaned = re.sub(r"[^\w\s\u0900-\u097F]", " ", text.lower())
        tokens = set(cleaned.split())

        expanded = set(tokens)
        for canon, synonyms in AGRI_SYNONYMS.items():
            if canon in tokens or any(syn in tokens for syn in synonyms):
                expanded.add(canon)
                for syn in synonyms:
                    expanded.add(syn)

        return expanded

    def retrieve(
        self,
        query: str,
        crop: Optional[str] = None,
        category: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> List[Tuple[AtomicFact, float]]:
        """Retrieve approved atomic facts ranked by domain relevance and source authority.

        Args:
            query: Farmer's natural language question in Hindi, English, etc.
            crop: Filter by target crop entity (e.g. 'tomato', 'wheat').
            category: Filter by agricultural domain ('disease_management', 'irrigation', etc.).
            state: Filter by state jurisdiction.
            top_k: Maximum facts to return.
            min_score: Minimum relevance threshold.

        Returns:
            List of (AtomicFact, relevance_score) sorted by relevance descending.
        """
        approved_facts = self.fact_repo.get_all_facts(status="approved")
        if not approved_facts:
            logger.warning("No approved facts found in repository.")
            return []

        query_tokens = self._normalize_tokens(query)
        if crop:
            query_tokens.update(self._normalize_tokens(crop))

        scored_candidates: List[Tuple[AtomicFact, float]] = []

        for fact in approved_facts:
            # 1. Scope and category filtering
            if crop and fact.subject.lower() != "general":
                if fact.subject.lower() != crop.strip().lower():
                    continue

            if category and fact.category.lower() != category.strip().lower():
                continue

            if state and fact.geographic_scope.get("states"):
                states = [s.lower() for s in fact.geographic_scope["states"]]
                if state.lower() not in states:
                    continue

            # 2. Compute BM25-style term frequency overlap
            cond_text = " ".join([f"{k} {v}" for k, v in fact.conditions.items()])
            fact_text = f"{fact.fact_id} {fact.subject} {fact.category} {fact.predicate} {fact.value} {fact.scientific_name or ''} {cond_text}"
            evidence_text = " ".join([f"{e.source_title} {getattr(e, 'quote_or_excerpt', '')}" for e in fact.evidence])
            doc_text = f"{fact_text} {evidence_text}"

            doc_tokens = self._normalize_tokens(doc_text)
            overlap = query_tokens.intersection(doc_tokens)
            if not overlap:
                continue

            # Raw score based on token overlap proportion and document length normalization
            score = len(overlap) / (math.log(len(doc_tokens) + 1) + 1.0)

            # 3. Boost for exact subject match
            if crop and fact.subject.lower() == crop.strip().lower():
                score *= 1.4

            # 4. Boost for source authority tier
            authority_multiplier = 1.0
            for ev in fact.evidence:
                src = self.source_repo.get_source(ev.source_id)
                if src:
                    if "Tier-1" in src.authority_tier:
                        authority_multiplier = max(authority_multiplier, 1.3)
                    elif "Tier-2" in src.authority_tier:
                        authority_multiplier = max(authority_multiplier, 1.15)

            final_score = round(score * authority_multiplier, 4)

            if final_score >= min_score:
                scored_candidates.append((fact, final_score))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        results = scored_candidates[:top_k]

        logger.info(f"Retrieval for query '{query[:40]}' (Crop: {crop}): Returned {len(results)} facts.")
        return results

    def extract_evidence(self, facts: List[AtomicFact]) -> List[EvidenceExcerpt]:
        """Deduplicate and extract all traceable evidence excerpts from retrieved facts."""
        seen_ids = set()
        evidence_list: List[EvidenceExcerpt] = []

        for fact in facts:
            for ev in fact.evidence:
                if ev.evidence_id not in seen_ids:
                    seen_ids.add(ev.evidence_id)
                    evidence_list.append(ev)

        return evidence_list

    def rebuild_index(self) -> None:
        """Reload underlying repositories and reset any cached indices."""
        self.fact_repo.reload()
        self.source_repo.reload()
        logger.info("RetrievalService reloaded repositories and rebuilt index.")
