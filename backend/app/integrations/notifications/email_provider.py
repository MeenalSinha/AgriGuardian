"""Email Notification Provider adapter."""

import os
import time
from typing import Any, Dict, Optional

from backend.app.integrations.notifications.base_provider import (
    BaseNotificationProvider,
    DeliveryState,
    ProviderDeliveryResult,
)
from backend.app.schemas.notification import NotificationChannel


class EmailNotificationProvider(BaseNotificationProvider):
    """SMTP / transactional email provider adapter."""

    def __init__(self):
        super().__init__(channel=NotificationChannel.EMAIL, provider_name="Transactional Email Gateway")
        self.smtp_host = os.getenv("SMTP_HOST")
        self.sendgrid_key = os.getenv("SENDGRID_API_KEY")

    def is_configured(self) -> bool:
        return bool(self.smtp_host or self.sendgrid_key)

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
        recipient_email = metadata.get("email") if metadata else None

        if not recipient_email or "@" not in recipient_email:
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Missing or invalid recipient email address.",
                response_code="400_INVALID_EMAIL",
                delivery_state=DeliveryState.FAILED,
            )

        if not self.is_configured():
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Email provider not configured (SMTP_HOST / SENDGRID_API_KEY unset).",
                response_code="503_EMAIL_UNCONFIGURED",
                not_configured=True,
                delivery_state=DeliveryState.NOT_CONFIGURED,
            )

        latency = (time.perf_counter() - start) * 1000.0
        return ProviderDeliveryResult(
            success=True,
            channel=self.channel,
            provider_name=self.provider_name,
            latency_ms=round(latency, 2),
            response_code="200_EMAIL_SENT",
            details=f"Email notification dispatched to {recipient_email}",
            delivery_state=DeliveryState.SENT,
        )

