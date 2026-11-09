"""In-App Notification Provider adapter."""

import time
from typing import Any, Dict, Optional

from backend.app.integrations.notifications.base_provider import BaseNotificationProvider, ProviderDeliveryResult
from backend.app.schemas.notification import NotificationChannel


class InAppNotificationProvider(BaseNotificationProvider):
    """Native in-app delivery provider. Always configured as notifications are stored locally."""

    def __init__(self):
        super().__init__(channel=NotificationChannel.IN_APP, provider_name="In-App Store")

    def is_configured(self) -> bool:
        return True

    def send(
        self,
        recipient_id: str,
        recipient_phone: Optional[str],
        title: str,
        body: str,
        language: str = "hi-IN",
        action_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProviderDeliveryResult:
        start = time.perf_counter()
        # In-App messages are persisted in the database; provider dispatch verifies delivery
        latency = (time.perf_counter() - start) * 1000.0
        return ProviderDeliveryResult(
            success=True,
            channel=self.channel,
            provider_name=self.provider_name,
            latency_ms=round(latency, 2),
            response_code="200_IN_APP_QUEUED",
            details=f"In-app notification ready for farmer {recipient_id}",
        )
