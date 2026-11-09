"""Repository for atomic, verified agricultural facts with lifecycle and provenance tracking."""

import json
import os
from typing import Any, Dict, List, Optional
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.knowledge import AtomicFact

logger = get_logger("FactRepository")


class FactRepository:
    """Manages the persistence, lifecycle, and query filtering of atomic agricultural facts."""

    _instance: Optional["FactRepository"] = None

    def __init__(self, data_dir: Optional[str] = None):
        settings = get_settings()
        self.data_dir = data_dir or os.path.join(settings.data_dir, "facts")
        self.facts_file = os.path.join(self.data_dir, "facts.json")
        self.facts: Dict[str, AtomicFact] = {}
        self._ensure_storage()
        self.load_facts()

    @classmethod
    def get_instance(cls) -> "FactRepository":
        """Get or initialize singleton repository instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_storage(self) -> None:
        """Ensure storage directory and initial facts file exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.facts_file):
            with open(self.facts_file, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0.0", "facts": []}, f, indent=2)

    def load_facts(self) -> None:
        """Load atomic facts from disk."""
        if not os.path.exists(self.facts_file):
            self.facts = {}
            return

        try:
            with open(self.facts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("facts", [])
                self.facts = {
                    item["fact_id"]: AtomicFact.model_validate(item)
                    for item in records
                }
            logger.info(f"Loaded {len(self.facts)} atomic facts from {self.facts_file}")
        except Exception as e:
            logger.error(f"Failed to load facts repository: {e}", exc_info=True)
            self.facts = {}

    def save_facts(self) -> None:
        """Persist facts dictionary back to disk."""
        payload = {
            "version": "1.0.0",
            "facts": [f.model_dump() for f in self.facts.values()],
        }
        with open(self.facts_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def add_fact(self, fact: AtomicFact) -> AtomicFact:
        """Add or update an atomic agricultural fact."""
        self.facts[fact.fact_id] = fact
        self.save_facts()
        logger.info(f"Stored fact: {fact.fact_id} ({fact.subject} - {fact.predicate})")
        return fact

    def get_fact(self, fact_id: str) -> Optional[AtomicFact]:
        """Retrieve fact by unique identifier."""
        return self.facts.get(fact_id)

    def get_all_facts(self, status: Optional[str] = "approved") -> List[AtomicFact]:
        """List facts filtered by verification status."""
        if status is None:
            return list(self.facts.values())
        return [f for f in self.facts.values() if f.verification_status == status]

    def filter_facts(
        self,
        subject: Optional[str] = None,
        category: Optional[str] = None,
        predicate: Optional[str] = None,
        status: str = "approved",
    ) -> List[AtomicFact]:
        """Filter facts by subject entity, domain category, and predicate."""
        results: List[AtomicFact] = []
        for f in self.facts.values():
            if status and f.verification_status != status:
                continue
            if subject and f.subject.lower() != subject.lower():
                continue
            if category and f.category.lower() != category.lower():
                continue
            if predicate and f.predicate.lower() != predicate.lower():
                continue
            results.append(f)
        return results

    def detect_conflicts(self, new_fact: AtomicFact) -> List[AtomicFact]:
        """Identify potentially contradictory facts sharing subject and predicate."""
        conflicts = []
        for existing in self.facts.values():
            if (
                existing.fact_id != new_fact.fact_id
                and existing.subject.lower() == new_fact.subject.lower()
                and existing.predicate == new_fact.predicate
                and existing.verification_status == "approved"
                and existing.value.strip().lower() != new_fact.value.strip().lower()
            ):
                conflicts.append(existing)
        return conflicts

    # --------------------------------------------------------------------------
    # CONVENIENCE METHODS & ALIASES
    # --------------------------------------------------------------------------

    def get_all(self) -> List[AtomicFact]:
        """Alias returning all facts regardless of status."""
        return self.get_all_facts(status=None)

    def get_by_id(self, fact_id: str) -> Optional[AtomicFact]:
        """Alias for get_fact."""
        return self.get_fact(fact_id)

    def get_by_crop(self, crop: str) -> List[AtomicFact]:
        """Alias filtering facts by crop entity (subject)."""
        return self.filter_facts(subject=crop, status=None)

    def get_by_subject(self, subject: str) -> List[AtomicFact]:
        """Alias filtering facts by subject."""
        return self.filter_facts(subject=subject, status=None)

    def get_by_predicate(self, predicate: str) -> List[AtomicFact]:
        """Alias filtering facts by predicate relation."""
        return self.filter_facts(predicate=predicate, status=None)

    def reload(self) -> None:
        """Reload facts repository from disk."""
        self.load_facts()
