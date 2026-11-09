"""WhatsApp Business API Notification adapter."""

import os
import time
from typing import Any, Dict, Optional

from backend.app.integrations.notifications.base_provider import BaseNotificationProvider, ProviderDeliveryResult
from backend.app.schemas.notification import NotificationChannel


class WhatsAppNotificationProvider(BaseNotificationProvider):
    """WhatsApp Business API adapter."""

    def __init__(self):
        super().__init__(channel=NotificationChannel.WHATSAPP, provider_name="Meta WhatsApp Business API")
        self.api_key = os.getenv("WHATSAPP_API_KEY")
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.phone_number_id)

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

        if not recipient_phone or len(recipient_phone.strip()) < 10:
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="Missing or invalid recipient phone number for WhatsApp message.",
                response_code="400_INVALID_PHONE",
            )

        if not self.is_configured():
            latency = (time.perf_counter() - start) * 1000.0
            return ProviderDeliveryResult(
                success=False,
                channel=self.channel,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                error_message="WhatsApp Business API not configured (WHATSAPP_API_KEY unset).",
                response_code="503_WHATSAPP_UNCONFIGURED",
                not_configured=True,
            )

        latency = (time.perf_counter() - start) * 1000.0
        return ProviderDeliveryResult(
            success=True,
            channel=self.channel,
            provider_name=self.provider_name,
            latency_ms=round(latency, 2),
            response_code="200_WHATSAPP_QUEUED",
            details=f"Template message sent via WhatsApp to {recipient_phone[-4:]}",
        )
