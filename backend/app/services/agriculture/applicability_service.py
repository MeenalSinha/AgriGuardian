"""Service for evaluating agricultural context applicability and prerequisite constraints."""

from typing import Any, Dict, List, Tuple
from backend.app.core.logging import get_logger
from backend.app.schemas.knowledge import AtomicFact

logger = get_logger("ApplicabilityService")


class ApplicabilityService:
    """Evaluates whether retrieved facts match the farmer's operational context."""

    @staticmethod
    def evaluate_applicability(
        fact: AtomicFact,
        farmer_context: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[str]]:
        """Determine if a fact applies to the farmer's crop, growth stage, soil, and region.

        Args:
            fact: The candidate AtomicFact record.
            farmer_context: Dictionary with keys:
                - crop (str)
                - crop_stage (str, optional)
                - days_after_sowing (int, optional)
                - state (str, optional)
                - weather (dict, optional)
                - severity_stage (str, optional)

        Returns:
            Tuple of (is_applicable, match_reasons, missing_prerequisites)
        """
        match_reasons: List[str] = []
        missing_prereqs: List[str] = []

        # 1. Crop Match
        target_crop = farmer_context.get("crop", "").strip().lower()
        if fact.subject.lower() != "general":
            if target_crop and fact.subject.lower() != target_crop:
                return False, [], [f"Crop mismatch: Fact is for '{fact.subject}', farmer crop is '{target_crop}'."]
            match_reasons.append(f"Direct crop match for {fact.subject}")

        # 2. Growth Stage Condition Check
        conds = fact.conditions or {}
        required_stage = conds.get("crop_stage")
        farmer_stage = farmer_context.get("crop_stage")
        farmer_das = farmer_context.get("days_after_sowing")

        if required_stage:
            if farmer_stage:
                if required_stage.lower() in farmer_stage.lower():
                    match_reasons.append(f"Matches crop growth stage '{required_stage}'")
                else:
                    return False, [], [f"Growth stage mismatch: Fact applies to '{required_stage}', crop is currently at '{farmer_stage}'."]
            elif farmer_das is not None and conds.get("days_after_sowing"):
                das_min, das_max = conds["days_after_sowing"]
                if das_min <= farmer_das <= das_max:
                    match_reasons.append(f"Crop age ({farmer_das} DAS) matches target window [{das_min}-{das_max} DAS]")
                else:
                    return False, [], [f"Crop age ({farmer_das} DAS) outside applicable range [{das_min}-{das_max} DAS]."]
            else:
                missing_prereqs.append(f"Growth stage '{required_stage}' not confirmed by farmer")

        # 3. Geographic State Applicability
        target_state = farmer_context.get("state")
        fact_states = fact.geographic_scope.get("states", [])
        if target_state and fact_states:
            if any(target_state.lower() in s.lower() for s in fact_states):
                match_reasons.append(f"Geographically validated for state '{target_state}'")
            else:
                return False, [], [f"Geographic mismatch: Fact validated for {fact_states}, farm is in '{target_state}'."]

        # 4. Disease Severity Level Applicability
        required_severity = conds.get("severity_stage")
        farmer_severity = farmer_context.get("severity_stage")
        if required_severity and farmer_severity:
            if required_severity.lower() in farmer_severity.lower():
                match_reasons.append(f"Matches clinical severity stage '{farmer_severity}'")
            elif "severe" in required_severity.lower() and farmer_severity.lower() == "mild":
                return False, [], ["Chemical intervention reserved for moderate/severe stage; current foliage damage is mild."]

        return True, match_reasons, missing_prereqs
