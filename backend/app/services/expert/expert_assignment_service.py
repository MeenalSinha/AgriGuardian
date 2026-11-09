"""Matching, ranking, and workload balancing service for assigning cases to agricultural experts."""

from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.expert import (
    ExpertAccountStatus,
    ExpertProfile,
    ExpertSpecializationDomain,
)

logger = get_logger("ExpertAssignmentService")


class ExpertAssignmentService:
    """Specialization-aware, workload-balanced expert matching engine."""

    def __init__(self, repo: FarmDatabaseRepository):
        self.repo = repo

    def find_best_expert_match(
        self,
        crop: str,
        state: str,
        language: str = "hi-IN",
        domain: ExpertSpecializationDomain = ExpertSpecializationDomain.PLANT_PATHOLOGY,
        exclude_expert_ids: Optional[List[str]] = None,
    ) -> Optional[ExpertProfile]:
        """Find the most qualified, available verified expert based on crop, region, language, and caseload."""
        exclude_ids = set(exclude_expert_ids or [])
        verified_experts = self.repo.list_expert_profiles(status=ExpertAccountStatus.ACTIVE_VERIFIED.value)
        
        candidates = [e for e in verified_experts if e.expert_id not in exclude_ids]
        if not candidates:
            return None

        scored_candidates: List[Tuple[float, ExpertProfile]] = []
        for exp in candidates:
            score = self._score_expert_suitability(exp, crop, state, language, domain)
            scored_candidates.append((score, exp))

        # Sort descending by suitability score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1] if scored_candidates else None

    def find_secondary_reviewer(
        self,
        primary_expert_id: str,
        crop: str,
        state: str,
        language: str = "hi-IN",
    ) -> Optional[ExpertProfile]:
        """Find an independent verified expert to conduct secondary / dual review."""
        return self.find_best_expert_match(
            crop=crop,
            state=state,
            language=language,
            domain=ExpertSpecializationDomain.PLANT_PATHOLOGY,
            exclude_expert_ids=[primary_expert_id],
        )

    def _score_expert_suitability(
        self,
        expert: ExpertProfile,
        crop: str,
        state: str,
        language: str,
        domain: ExpertSpecializationDomain,
    ) -> float:
        """Calculate match score from specialization, geographic familiarity, language, and caseload."""
        score = 0.0
        specs = expert.specializations

        # Crop matching (+35 points)
        crop_lower = crop.lower()
        if any(c.lower() in crop_lower or crop_lower in c.lower() for c in specs.crop_types):
            score += 35.0

        # Domain matching (+25 points)
        if domain in specs.domains or ExpertSpecializationDomain.GENERAL_AGRONOMY in specs.domains:
            score += 25.0

        # State / Region matching (+20 points)
        if any(s.lower() == state.lower() for s in specs.states):
            score += 20.0

        # Language matching (+10 points)
        if any(l.lower().startswith(language.lower()[:2]) for l in specs.languages):
            score += 10.0

        # Experience bonus (+0.5 point per year, max 10 points)
        score += min(expert.experience_years * 0.5, 10.0)

        # Workload penalty (-2 points per active case to balance load)
        score -= expert.active_caseload * 2.0

        # Accuracy rating multiplier
        score *= (expert.accuracy_rating / 5.0)

        return max(score, 0.0)

