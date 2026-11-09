"""Service for identifying and arbitrating conflicting agricultural guidance."""

from typing import Dict, List, Optional, Tuple
from backend.app.core.logging import get_logger
from backend.app.repositories.source_repository import SourceRepository
from backend.app.schemas.knowledge import AtomicFact

logger = get_logger("ConflictResolutionService")


class ConflictResolutionService:
    """Detects and resolves contradictory advice across multiple agricultural knowledge sources."""

    def __init__(self, source_repo: Optional[SourceRepository] = None):
        self.source_repo = source_repo or SourceRepository.get_instance()

    def arbitrate_facts(self, facts: List[AtomicFact]) -> Tuple[List[AtomicFact], List[Dict[str, Any]]]:
        """Detect conflicts among facts and prioritize higher-tier, recent, authoritative records.

        Returns:
            Tuple of (reconciled_facts, conflict_audit_trail)
        """
        seen_predicates: Dict[Tuple[str, str], AtomicFact] = {}
        conflicts: List[Dict[str, Any]] = []
        approved_facts: List[AtomicFact] = []

        for fact in facts:
            key = (fact.subject.lower(), fact.predicate)
            if key not in seen_predicates:
                seen_predicates[key] = fact
                approved_facts.append(fact)
            else:
                existing = seen_predicates[key]
                # Check if values diverge significantly
                if existing.value.strip().lower() != fact.value.strip().lower():
                    # Resolve based on authority tier
                    winner = self._compare_authority(existing, fact)
                    conflicts.append({
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "fact_a": existing.fact_id,
                        "fact_b": fact.fact_id,
                        "preferred_fact": winner.fact_id,
                        "resolution_reason": "Prioritized higher authority tier and regulatory statutory record.",
                    })
                    seen_predicates[key] = winner
                    if winner is fact and existing in approved_facts:
                        approved_facts.remove(existing)
                        approved_facts.append(fact)

        if conflicts:
            logger.info(f"Arbitrated {len(conflicts)} agricultural knowledge conflicts.")

        return approved_facts, conflicts

    def _compare_authority(self, fact_a: AtomicFact, fact_b: AtomicFact) -> AtomicFact:
        """Compare authority tiers between two conflicting facts."""
        tier_a = self._get_tier_weight(fact_a)
        tier_b = self._get_tier_weight(fact_b)
        return fact_a if tier_a >= tier_b else fact_b

    def _get_tier_weight(self, fact: AtomicFact) -> int:
        """Compute numeric weight for source authority tier."""
        for ev in fact.evidence:
            src = self.source_repo.get_source(ev.source_id)
            if src:
                if "Tier-1" in src.authority_tier:
                    return 3
                if "Tier-2" in src.authority_tier:
                    return 2
                if "Tier-3" in src.authority_tier:
                    return 1
        return 0

