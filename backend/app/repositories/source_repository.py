"""Repository for registering, validating, and retrieving authoritative agricultural sources."""

import json
import os
from typing import Any, Dict, List, Optional
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.source import SourceRecord

logger = get_logger("SourceRepository")


class SourceRepository:
    """Manages the persistence, metadata validation, and retrieval of agricultural sources."""

    _instance: Optional["SourceRepository"] = None

    def __init__(self, data_dir: Optional[str] = None):
        settings = get_settings()
        self.data_dir = data_dir or os.path.join(settings.data_dir, "sources")
        self.registry_file = os.path.join(self.data_dir, "registry.json")
        self.sources: Dict[str, SourceRecord] = {}
        self._ensure_storage()
        self.load_registry()

    @classmethod
    def get_instance(cls) -> "SourceRepository":
        """Get or initialize singleton repository instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_storage(self) -> None:
        """Ensure storage directory and initial registry exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0.0", "sources": []}, f, indent=2)

    def load_registry(self) -> None:
        """Load registered sources from disk."""
        if not os.path.exists(self.registry_file):
            self.sources = {}
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data.get("sources", [])
                self.sources = {
                    item["source_id"]: SourceRecord.model_validate(item)
                    for item in records
                }
            logger.info(f"Loaded {len(self.sources)} sources from {self.registry_file}")
        except Exception as e:
            logger.error(f"Failed to load source registry: {e}", exc_info=True)
            self.sources = {}

    def save_registry(self) -> None:
        """Persist sources dictionary back to disk."""
        payload = {
            "version": "1.0.0",
            "sources": [s.model_dump() for s in self.sources.values()],
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def register_source(self, record: SourceRecord) -> SourceRecord:
        """Register or update an authoritative agricultural source."""
        self.sources[record.source_id] = record
        self.save_registry()
        logger.info(f"Registered source: {record.source_id} ('{record.title}')")
        return record

    def get_source(self, source_id: str) -> Optional[SourceRecord]:
        """Retrieve source by unique identifier."""
        return self.sources.get(source_id)

    def get_all_sources(self, status: Optional[str] = "approved") -> List[SourceRecord]:
        """List sources filtered by verification status."""
        if status is None:
            return list(self.sources.values())
        return [s for s in self.sources.values() if s.verification_status == status]

    def filter_sources(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        status: str = "approved",
    ) -> List[SourceRecord]:
        """Find sources applicable to specific crop and geographic state."""
        matches: List[SourceRecord] = []
        for s in self.sources.values():
            if status and s.verification_status != status:
                continue
            if crop and s.crop_scope:
                if not any(crop.lower() == c.lower() for c in s.crop_scope):
                    continue
            if state and s.state_scope:
                if not any(state.lower() == st.lower() for st in s.state_scope):
                    continue
            matches.append(s)
        return matches

    def retire_source(self, source_id: str, reason: str = "Superseded by updated edition") -> bool:
        """Mark a source as retired so its facts are excluded from authoritative retrieval."""
        src = self.get_source(source_id)
        if not src:
            return False
        src.verification_status = "retired"
        self.sources[source_id] = src
        self.save_registry()
        logger.info(f"Retired source {source_id}: {reason}")
        return True

    # --------------------------------------------------------------------------
    # CONVENIENCE METHODS & ALIASES
    # --------------------------------------------------------------------------

    def get_all(self) -> List[SourceRecord]:
        """Alias returning all sources regardless of status."""
        return self.get_all_sources(status=None)

    def get_by_id(self, source_id: str) -> Optional[SourceRecord]:
        """Alias for get_source."""
        return self.get_source(source_id)

    def get_by_tier(self, tier: Any) -> List[SourceRecord]:
        """Find sources matching authority tier (int or string)."""
        target = str(tier)
        return [
            s for s in self.sources.values()
            if target in str(s.authority_tier) or (hasattr(s, "tier_int") and str(s.tier_int) == target)
        ]

    def get_by_crop(self, crop: str) -> List[SourceRecord]:
        """Find sources covering a specific crop."""
        c_low = crop.lower()
        return [
            s for s in self.sources.values()
            if any(c_low == c.lower() or c.lower() == "all" for c in s.crop_scope)
        ]

    def reload(self) -> None:
        """Reload source registry from disk."""
        self.load_registry()
