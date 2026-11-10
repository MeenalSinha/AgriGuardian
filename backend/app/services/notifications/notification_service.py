"""Unified high-level facade for Notification & Reminder Orchestration."""

from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.notifications.provider_registry import NotificationProviderRegistry
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.notification import (
    ChannelHealthInfo,
    FarmReminderCreate,
    FarmReminderRecord,
    FarmReminderUpdate,
    NotificationPreference,
    NotificationRecord,
    OutboxProcessResult,
)
from backend.app.services.notifications.notification_event_bus import NotificationEventBus
from backend.app.services.notifications.notification_outbox_service import NotificationOutboxService
from backend.app.services.notifications.notification_policy_engine import NotificationPolicyEngine
from backend.app.services.notifications.notification_template_service import NotificationTemplateService
from backend.app.services.notifications.reminder_service import ReminderService

logger = get_logger("NotificationService")

_notification_service_singleton: Optional["NotificationService"] = None


class NotificationService:
    """Master service orchestrating events, templates, outbox dispatch, and smart reminders."""

    @classmethod
    def get_instance(cls, repo: Optional[FarmDatabaseRepository] = None) -> "NotificationService":
        """Return global singleton instance, initializing if needed."""
        global _notification_service_singleton
        if _notification_service_singleton is None or repo is not None:
            _notification_service_singleton = cls(repo=repo)
        return _notification_service_singleton

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        global _notification_service_singleton
        self.repo = repo or get_farm_repo()
        self.registry = NotificationProviderRegistry()
        self.policy_engine = NotificationPolicyEngine(self.repo)
        self.template_service = NotificationTemplateService(self.repo)
        self.event_bus = NotificationEventBus(
            repo=self.repo,
            policy_engine=self.policy_engine,
            template_service=self.template_service,
        )
        self.outbox_service = NotificationOutboxService(
            repo=self.repo,
            provider_registry=self.registry,
        )
        self.reminder_service = ReminderService(
            repo=self.repo,
            event_bus=self.event_bus,
        )
        if _notification_service_singleton is None:
            _notification_service_singleton = self

    # In-App Notifications
    def list_notifications(
        self,
        farmer_id: str,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        unread_only: Optional[bool] = None,
    ) -> List[NotificationRecord]:
        """Fetch notifications for farmer inbox."""
        effective_status = "unread" if unread_only else status
        return self.repo.list_notifications_for_farmer(
            farmer_id=farmer_id,
            status=effective_status,
            category=category,
            limit=limit,
            offset=offset,
        )

    def get_unread_count(self, farmer_id: str) -> int:
        """Count unread notifications for badge icon."""
        return self.repo.get_unread_notification_count(farmer_id)

    def mark_read(self, notification_id: str, farmer_id: Optional[str] = None) -> bool:
        """Mark single notification as read."""
        return self.repo.mark_notification_as_read(notification_id=notification_id, farmer_id=farmer_id)

    def mark_all_read(self, farmer_id: str) -> int:
        """Mark all unread notifications as read."""
        return self.repo.mark_all_notifications_as_read(farmer_id)

    def dismiss(self, notification_id: str, farmer_id: Optional[str] = None) -> bool:
        """Dismiss notification."""
        return self.repo.dismiss_notification(notification_id=notification_id, farmer_id=farmer_id)

    # Preferences & Consents
    def get_preferences(self, farmer_id: str) -> NotificationPreference:
        """Get notification preferences."""
        return self.repo.get_notification_preference(farmer_id)

    def update_preferences(self, pref: NotificationPreference) -> NotificationPreference:
        """Update notification preferences."""
        self.repo.save_notification_preference(pref)
        return self.get_preferences(pref.farmer_id)

    def get_channel_consents(self, farmer_id: str):
        """Get farmer's channel-specific consents."""
        return self.repo.get_channel_consents(farmer_id)

    def save_channel_consent(self, consent):
        """Save farmer channel consent."""
        return self.repo.save_channel_consent(consent)

    # Smart Reminders
    def create_reminder(self, farmer_id: str, req: FarmReminderCreate) -> FarmReminderRecord:
        """Create custom reminder."""
        return self.reminder_service.create_reminder(farmer_id, req)

    def list_reminders(
        self,
        farmer_id: str,
        status: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[FarmReminderRecord]:
        """List reminders for farmer."""
        return self.reminder_service.list_reminders(farmer_id, status, from_date, to_date)

    def complete_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> bool:
        """Mark reminder completed."""
        return self.reminder_service.complete_reminder(reminder_id, farmer_id)

    def snooze_reminder(
        self,
        reminder_id: str,
        snooze_days: int = 1,
        custom_date: Optional[str] = None,
        farmer_id: Optional[str] = None,
    ) -> bool:
        """Snooze reminder."""
        return self.reminder_service.snooze_reminder(reminder_id, snooze_days, custom_date, farmer_id)

    def get_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> Optional[FarmReminderRecord]:
        """Fetch reminder by ID with optional tenant scope."""
        return self.reminder_service.get_reminder(reminder_id, farmer_id=farmer_id)

    def update_reminder(
        self,
        reminder_id: str,
        update_req: FarmReminderUpdate,
        farmer_id: Optional[str] = None,
    ) -> Optional[FarmReminderRecord]:
        """Update reminder and re-evaluate weather risk."""
        return self.reminder_service.update_reminder(reminder_id, update_req, farmer_id=farmer_id)

    def delete_reminder(self, reminder_id: str, farmer_id: Optional[str] = None) -> bool:
        """Permanently delete reminder."""
        return self.reminder_service.delete_reminder(reminder_id, farmer_id=farmer_id)

    def generate_crop_reminders(self, farmer_id: str) -> List[FarmReminderRecord]:
        """Generate smart reminders from active crop cycle activities."""
        return self.reminder_service.generate_reminders_from_crop_cycles(farmer_id)

    # Outbox & Administration
    def process_outbox(self, batch_size: int = 20) -> OutboxProcessResult:
        """Trigger worker execution cycle on outbox queue."""
        return self.outbox_service.process_outbox_batch(batch_size=batch_size)

    def get_outbox_metrics(self) -> Dict[str, int]:
        """Get outbox counts by state."""
        return self.repo.get_outbox_metrics()

    def get_outbox_telemetry(self) -> Dict[str, Any]:
        """Get outbox telemetry and queue counts."""
        metrics = self.get_outbox_metrics()
        return {
            "status_counts": metrics,
            "total_enqueued": sum(metrics.values()),
        }

    def list_outbox_items(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List outbox queue items."""
        return self.repo.list_outbox_items(status=status, limit=limit)

    def retry_outbox_message(self, outbox_id: str) -> bool:
        """Retry a message."""
        return self.outbox_service.retry_outbox_item(outbox_id)

    def get_channel_health(self) -> List[ChannelHealthInfo]:
        """Return truthful channel readiness and delivery statistics."""
        return self.outbox_service.get_channel_health_report()

    def get_channels_telemetry(self) -> Dict[str, Any]:
        """Return readiness and gateway status for all dispatch channels."""
        health_list = self.get_channel_health()
        res = {}
        for h in health_list:
            res[h.channel.value] = {
                "provider": h.provider_name,
                "configured": h.configured,
                "environment_mode": "Production" if h.configured else "Local Sandbox / Simulated",
                "note": h.details or "Operational",
            }
        return res

