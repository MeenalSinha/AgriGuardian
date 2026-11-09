"""Registry managing channel notification providers and operational status checks."""

from typing import Dict, List

from backend.app.integrations.notifications.base_provider import BaseNotificationProvider
from backend.app.integrations.notifications.email_provider import EmailNotificationProvider
from backend.app.integrations.notifications.in_app_provider import InAppNotificationProvider
from backend.app.integrations.notifications.sms_provider import SmsNotificationProvider
from backend.app.integrations.notifications.voice_provider import VoiceCallNotificationProvider
from backend.app.integrations.notifications.whatsapp_provider import WhatsAppNotificationProvider
from backend.app.schemas.notification import ChannelHealthInfo, NotificationChannel


class NotificationProviderRegistry:
    """Manages active multi-channel providers and inspects configuration readiness."""

    def __init__(self):
        self._providers: Dict[NotificationChannel, BaseNotificationProvider] = {
            NotificationChannel.IN_APP: InAppNotificationProvider(),
            NotificationChannel.SMS: SmsNotificationProvider(),
            NotificationChannel.WHATSAPP: WhatsAppNotificationProvider(),
            NotificationChannel.EMAIL: EmailNotificationProvider(),
            NotificationChannel.VOICE_CALL: VoiceCallNotificationProvider(),
        }

    def get_provider(self, channel: NotificationChannel) -> BaseNotificationProvider:
        """Retrieve provider instance for a channel."""
        if channel not in self._providers:
            raise ValueError(f"No notification provider registered for channel '{channel.value}'.")
        return self._providers[channel]

    def get_all_providers(self) -> Dict[NotificationChannel, BaseNotificationProvider]:
        """Return all registered providers."""
        return self._providers

    def get_channels_health(self) -> List[ChannelHealthInfo]:
        """Truthfully report live configuration and readiness status across all notification channels."""
        health_list = []
        for channel, provider in self._providers.items():
            is_ready = provider.is_configured()
            if channel == NotificationChannel.IN_APP:
                details = "Native SQLite transactional store; always operational"
            elif is_ready:
                details = f"Active configuration detected for {provider.provider_name}"
            else:
                details = f"Provider credentials unconfigured ({provider.provider_name})"

            health_list.append(
                ChannelHealthInfo(
                    channel=channel,
                    configured=is_ready,
                    provider_name=provider.provider_name,
                    details=details,
                )
            )
        return health_list

