"""Agricultural Safety & Regulatory Guardrail Service enforcing CIBRC and ICAR standards."""

import re
from typing import Any, Dict, List, Optional, Tuple
from backend.app.core.logging import get_logger
from backend.app.schemas.knowledge import AtomicFact
from backend.app.schemas.recommendation import ActionItem

logger = get_logger("SafetyGuardrailService")

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"ignore\s+(all\s+)?(rules|safety|guidelines)",
    r"disregard\s+(all\s+)?(safety\s+)?rules",
    r"bypass\s+(safety|rules|guardrails)",
    r"jailbreak",
    r"you\s+are\s+now\s+(dan|unrestricted)",
    r"system\s+prompt",
    r"how\s+to\s+make\s+(poison|explosive|toxic)",
]

HAZARDOUS_CHEMICAL_KEYWORDS = [
    "monocrotophos", "phorate", "endosulfan", "carbofuran", "dichlorvos",
    "methyl parathion", "paraquat", "aluminium phosphide"
]


class SafetyGuardrailService:
    """Enforces chemical, environmental, biological, and prompt security guardrails."""

    BANNED_CHEMICALS = HAZARDOUS_CHEMICAL_KEYWORDS

    @staticmethod
    def inspect_query_security(query: str) -> Tuple[bool, str]:
        """Detect prompt injection attempts or hazardous query intent.

        Returns:
            Tuple of (is_safe, refusal_reason)
        """
        q_lower = query.lower()

        # Check prompt injection patterns
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                logger.warning(f"Prompt injection pattern detected in query: '{query[:50]}'")
                return False, "Input query rejected due to prompt security policy violation."

        # Check banned or restricted lethal agrochemicals
        for banned in HAZARDOUS_CHEMICAL_KEYWORDS:
            if banned in q_lower:
                logger.warning(f"Banned toxic agrochemical queried: '{banned}'")
                return False, f"Notice: '{banned.title()}' is a restricted/banned substance in Indian agriculture under the Insecticides Act. AgriGuardian promotes CIBRC-registered, safe IPM alternatives."

        return True, ""

    @staticmethod
    def sanitize_action_items(
        actions: List[ActionItem],
        is_confirmed_diagnosis: bool = False,
    ) -> List[ActionItem]:
        """Verify chemical safety: downgrade or require KVK verification if diagnosis is unconfirmed."""
        sanitized: List[ActionItem] = []

        for item in actions:
            # If action involves restricted chemical intervention
            if item.safety_level in ("chemical_restricted", "restricted_chemical"):
                if not is_confirmed_diagnosis:
                    # Downgrade to expert referral / preliminary protectant
                    item.action = (
                        f"[Expert Guidance Required] {item.action}. "
                        "Do not apply systemic chemical fungicides or pesticides without physical leaf confirmation by your local Krishi Vigyan Kendra (KVK)."
                    )
                    item.safety_level = "expert_referral"
                    item.regulatory_warning = (
                        "Statutory Safety Notice: Wear protective mask and gloves. "
                        "Observe Central Insecticides Board (CIBRC) approved dosage and pre-harvest intervals (PHI)."
                    )
                else:
                    item.regulatory_warning = (
                        "Statutory Safety Notice: Wear protective mask and gloves. "
                        "Observe Central Insecticides Board (CIBRC) approved dosage and pre-harvest intervals (PHI)."
                    )

            sanitized.append(item)

        return sanitized

    @staticmethod
    def evaluate_escalation_triggers(
        severity_stage: str,
        confidence: float,
        is_uncertain: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether the case mandates physical Krishi Vigyan Kendra (KVK) escalation."""
        if is_uncertain or confidence < 0.40:
            return True, (
                "Diagnostic confidence is low or foliar symptoms are ambiguous. "
                "Physical leaf sample examination at your district Krishi Vigyan Kendra (KVK) is recommended before applying any chemical interventions."
            )

        if severity_stage == "Severe":
            return True, (
                "Severe foliar damage detected (>30% visible leaf tissue affected). "
                "Immediate agronomist field consultation at your local KVK or Department of Agriculture is strongly advised to prevent widespread defoliation."
            )

        return False, None
