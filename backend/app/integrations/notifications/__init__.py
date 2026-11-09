"""Multi-channel notification provider adapters and registry."""

from backend.app.integrations.notifications.base_provider import BaseNotificationProvider, ProviderDeliveryResult
from backend.app.integrations.notifications.email_provider import EmailNotificationProvider
from backend.app.integrations.notifications.in_app_provider import InAppNotificationProvider
from backend.app.integrations.notifications.provider_registry import NotificationProviderRegistry
from backend.app.integrations.notifications.sms_provider import SmsNotificationProvider
from backend.app.integrations.notifications.voice_provider import VoiceCallNotificationProvider
from backend.app.integrations.notifications.whatsapp_provider import WhatsAppNotificationProvider

__all__ = [
    "BaseNotificationProvider",
    "ProviderDeliveryResult",
    "InAppNotificationProvider",
    "SmsNotificationProvider",
    "WhatsAppNotificationProvider",
    "EmailNotificationProvider",
    "VoiceCallNotificationProvider",
    "NotificationProviderRegistry",
]
