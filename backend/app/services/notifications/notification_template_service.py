"""Bilingual notification template rendering service."""

import re
from typing import Any, Dict, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.notification import NotificationCategory, NotificationPriority, NotificationTemplate

logger = get_logger("NotificationTemplateService")


class NotificationTemplateService:
    """Renders localized bilingual notification content from database templates."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def render(
        self,
        event_type: str,
        context: Dict[str, Any],
        language: str = "hi-IN",
    ) -> Tuple[str, str, str, str, Optional[str], NotificationPriority, NotificationCategory]:
        """
        Render title and body in English and Hindi for a given event type.
        Returns: (title_en, title_hi, body_en, body_hi, action_url, default_priority, category)
        """
        template = self.repo.get_notification_template(event_type)
        if not template:
            # Fallback generic template
            category = context.get("category", NotificationCategory.SYSTEM_GOVERNANCE)
            if isinstance(category, str):
                category = NotificationCategory(category)
            title_en = str(context.get("title", f"AgriGuardian Notice: {event_type}"))
            title_hi = str(context.get("title_hi", f"एग्रीगार्डियन सूचना: {event_type}"))
            body_en = str(context.get("body", "Important agricultural update for your farm."))
            body_hi = str(context.get("body_hi", "आपके खेत के लिए महत्वपूर्ण कृषि सूचना।"))
            action_url = context.get("action_url")
            priority = context.get("priority", NotificationPriority.MEDIUM)
            if isinstance(priority, str):
                priority = NotificationPriority(priority)
            return title_en, title_hi, body_en, body_hi, action_url, priority, category

        # Interpolate variables safely
        title_en = self._interpolate(template.title_template_en, context)
        title_hi = self._interpolate(template.title_template_hi, context)
        body_en = self._interpolate(template.body_template_en, context)
        body_hi = self._interpolate(template.body_template_hi, context)

        action_url = None
        if template.action_url_template:
            action_url = self._interpolate(template.action_url_template, context)

        return (
            title_en,
            title_hi,
            body_en,
            body_hi,
            action_url,
            template.default_priority,
            template.category,
        )

    def _interpolate(self, text: str, context: Dict[str, Any]) -> str:
        """Replace {{var_name}} placeholders with values from context."""
        if not text:
            return ""

        def replacer(match):
            key = match.group(1).strip()
            val = context.get(key)
            if val is not None:
                return str(val)
            return f"[{key}]"

        return re.sub(r"\{\{([a-zA-Z0-9_]+)\}\}", replacer, text)
