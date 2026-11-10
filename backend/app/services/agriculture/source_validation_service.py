"""Service for validating agricultural source authority, currency, and geographic scope."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.logging import get_logger
from backend.app.repositories.source_repository import SourceRepository
from backend.app.schemas.source import SourceRecord

logger = get_logger("SourceValidationService")


class SourceValidationService:
    """Evaluates the authority, validity period, and jurisdictional applicability of sources."""

    def __init__(self, source_repo: Optional[SourceRepository] = None):
        self.source_repo = source_repo or SourceRepository.get_instance()

    def validate_source(
        self,
        source: Any,
        target_crop: Optional[str] = None,
        target_state: Optional[str] = None,
        crop: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate whether a source is approved, current, and applicable.

        Returns:
            Tuple of (is_valid, warnings)
        """
        c_filter = crop or target_crop
        s_filter = state or target_state

        if isinstance(source, SourceRecord):
            src_obj = source
            source_id = src_obj.source_id
        else:
            source_id = str(source)
            src_obj = self.source_repo.get_source(source_id)

        if not src_obj:
            return False, [f"Source '{source_id}' is not registered in the authoritative registry."]

        warnings: List[str] = []

        # 1. Lifecycle check
        if src_obj.verification_status != "approved":
            warnings.append(f"Source '{source_id}' has non-approved status: '{src_obj.verification_status}'.")

        # 2. Expiration check
        expiry_val = src_obj.valid_until or src_obj.next_review_date
        if expiry_val:
            try:
                # Handle YYYY-MM-DD or ISO
                if len(expiry_val) == 10:
                    exp_dt = datetime.fromisoformat(f"{expiry_val}T23:59:59+00:00")
                else:
                    exp_dt = datetime.fromisoformat(expiry_val.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp_dt:
                    warnings.append(f"Source '{source_id}' validity/review date ({expiry_val}) has passed; source expired.")
            except Exception:
                pass

        # 3. Crop scope check
        if c_filter and src_obj.crop_scope:
            crop_lower = c_filter.strip().lower()
            if not any(crop_lower == c.lower() or c.lower() == "all" for c in src_obj.crop_scope):
                warnings.append(f"Target crop '{c_filter}' is not applicable to source ({src_obj.crop_scope}).")

        # 4. State / Geographic scope check
        if s_filter and src_obj.state_scope:
            state_lower = s_filter.strip().lower()
            if not any(state_lower in s.lower() for s in src_obj.state_scope):
                warnings.append(f"Target state '{s_filter}' is outside regional scope of source ({src_obj.state_scope}).")

        is_valid = len(warnings) == 0
        return is_valid, warnings
