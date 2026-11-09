"""Base provider interface and result dataclasses for multi-channel notification dispatch."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from backend.app.schemas.notification import NotificationChannel


class DeliveryState(str, Enum):
    """Truthful provider dispatch and delivery status."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    REQUEST_ACCEPTED = "REQUEST_ACCEPTED"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    READ = "READ"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ProviderDeliveryResult:
    """Outcome of a physical or simulated provider dispatch."""
    success: bool
    channel: NotificationChannel
    provider_name: str
    latency_ms: float
    error_message: Optional[str] = None
    response_code: Optional[str] = None
    details: Optional[str] = None
    not_configured: bool = False
    delivery_state: DeliveryState = DeliveryState.UNKNOWN


class BaseNotificationProvider(ABC):
    """Abstract interface implemented by all channel providers."""

    def __init__(self, channel: NotificationChannel, provider_name: str):
        self.channel = channel
        self.provider_name = provider_name

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if environment credentials or gateways are active."""
        pass

    @abstractmethod
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
        """Deliver the message to recipient."""
        pass

