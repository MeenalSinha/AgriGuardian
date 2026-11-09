"""Event ingestion bus and factory for proactive agricultural alerts and reminders."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.notification import (
    NotificationCategory,
    NotificationChannel,
    NotificationEvent,
    NotificationOutboxItem,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    OutboxStatus,
)
from backend.app.services.notifications.notification_policy_engine import NotificationPolicyEngine
from backend.app.services.notifications.notification_template_service import NotificationTemplateService

logger = get_logger("NotificationEventBus")


class NotificationEventBus:
    """Central event broker for producing and dispatching agricultural alerts."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        policy_engine: Optional[NotificationPolicyEngine] = None,
        template_service: Optional[NotificationTemplateService] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.policy_engine = policy_engine or NotificationPolicyEngine(self.repo)
        self.template_service = template_service or NotificationTemplateService(self.repo)

    def publish_event(self, event: NotificationEvent) -> Dict[str, Any]:
        """Process event: persist event log, evaluate policy, render bilingual copy, and enqueue outbox jobs."""
        # 1. Persist raw event log
        self.repo.save_notification_event(event)

        # 2. Evaluate policy (dedup, quiet hours, preferences, consents)
        policy_result = self.policy_engine.evaluate(event)
        if not policy_result.allowed:
            logger.info(f"Event {event.event_id} suppressed by policy: {policy_result.suppression_reason}")
            return {
                "event_id": event.event_id,
                "status": "suppressed",
                "reason": policy_result.suppression_reason,
            }

        # 3. Render bilingual content
        pref = self.repo.get_notification_preference(event.farmer_id)
        (
            title_en,
            title_hi,
            body_en,
            body_hi,
            action_url,
            default_priority,
            category,
        ) = self.template_service.render(
            event_type=event.event_type,
            context={**event.payload, "farmer_id": event.farmer_id},
            language=pref.preferred_language,
        )

        effective_priority = event.priority if event.priority != NotificationPriority.MEDIUM else default_priority
        notif_id = f"notif_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()

        # 4. Save to in-app notification center
        notif_record = NotificationRecord(
            notification_id=notif_id,
            farmer_id=event.farmer_id,
            category=category,
            priority=effective_priority,
            title_en=title_en,
            title_hi=title_hi,
            body_en=body_en,
            body_hi=body_hi,
            action_url=action_url,
            metadata={
                **event.payload,
                "event_id": event.event_id,
                "related_entity_id": event.related_entity_id,
            },
            status=NotificationStatus.UNREAD,
            created_at=now_str,
        )
        self.repo.create_notification(notif_record)

        # 5. Route to external channels via Transactional Outbox
        outbox_ids = []
        for channel in policy_result.target_channels:
            if channel == NotificationChannel.IN_APP:
                continue

            outbox_id = f"obx_{uuid.uuid4().hex[:12]}"
            outbox_item = NotificationOutboxItem(
                outbox_id=outbox_id,
                notification_id=notif_id,
                farmer_id=event.farmer_id,
                channel=channel,
                status=OutboxStatus.PENDING,
                retry_count=0,
                max_retries=3,
                next_retry_at=policy_result.defer_until,  # If deferred for quiet hours
                created_at=now_str,
                updated_at=now_str,
            )
            self.repo.enqueue_outbox_message(outbox_item)
            outbox_ids.append(outbox_id)

        logger.info(
            f"Event {event.event_id} processed into notification {notif_id} with {len(outbox_ids)} outbox jobs (channels: {[c.value for c in policy_result.target_channels]})"
        )

        return {
            "event_id": event.event_id,
            "notification_id": notif_id,
            "status": "dispatched",
            "channels": [c.value for c in policy_result.target_channels],
            "outbox_ids": outbox_ids,
            "quiet_hours_deferred": bool(policy_result.defer_until),
        }

    # =========================================================================
    # EVENT GENERATION FACTORIES
    # =========================================================================

    def emit_weather_alert_event(
        self,
        farmer_id: str,
        hazard_type: str,
        severity: str,
        summary: str,
        village_or_district: str,
        farm_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger an extreme weather alert event."""
        prio = NotificationPriority.EMERGENCY if severity.lower() in ("critical", "extreme", "danger") else NotificationPriority.HIGH
        dedup_key = f"weather_{farmer_id}_{hazard_type}_{village_or_district}_{datetime.now().strftime('%Y-%m-%d')}"
        event = NotificationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="weather_alert_extreme",
            category=NotificationCategory.WEATHER_ALERT,
            priority=prio,
            farmer_id=farmer_id,
            farm_id=farm_id,
            payload={
                "hazard_type": hazard_type,
                "severity": severity,
                "summary": summary,
                "village_or_district": village_or_district,
            },
            dedup_key=dedup_key,
        )
        return self.publish_event(event)

    def emit_spray_weather_conflict_event(
        self,
        farmer_id: str,
        farm_name: str,
        crop_name: str,
        chemical_name: str,
        expected_rain_mm: float,
        hours_window: int = 24,
        farm_id: Optional[str] = None,
        crop_cycle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger weather-spray conflict alert when rainfall is forecast during scheduled chemical application."""
        dedup_key = f"spray_rain_{farmer_id}_{crop_name}_{chemical_name}_{datetime.now().strftime('%Y-%m-%d')}"
        event = NotificationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="weather_risk_rain_during_spray",
            category=NotificationCategory.WEATHER_ALERT,
            priority=NotificationPriority.HIGH,
            farmer_id=farmer_id,
            farm_id=farm_id,
            crop_cycle_id=crop_cycle_id,
            payload={
                "farm_name": farm_name,
                "crop_name": crop_name,
                "chemical_name": chemical_name,
                "expected_rain_mm": expected_rain_mm,
                "hours_window": hours_window,
            },
            dedup_key=dedup_key,
        )
        return self.publish_event(event)

    def emit_market_price_event(
        self,
        farmer_id: str,
        crop_name: str,
        mandi_name: str,
        current_price: float,
        previous_price: float,
        price_change_pct: float,
        is_surge: bool = True,
    ) -> Dict[str, Any]:
        """Trigger market price benchmark alert (surge or sharp drop)."""
        event_type = "market_price_surge" if is_surge else "market_price_drop"
        prio = NotificationPriority.MEDIUM if abs(price_change_pct) < 15.0 else NotificationPriority.HIGH
        dedup_key = f"market_{farmer_id}_{crop_name}_{mandi_name}_{event_type}_{datetime.now().strftime('%Y-%m-%d')}"
        event = NotificationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            category=NotificationCategory.MARKET_INTELLIGENCE,
            priority=prio,
            farmer_id=farmer_id,
            payload={
                "crop_name": crop_name,
                "mandi_name": mandi_name,
                "current_price": current_price,
                "previous_price": previous_price,
                "price_change_pct": round(abs(price_change_pct), 1),
            },
            dedup_key=dedup_key,
        )
        return self.publish_event(event)

    def emit_expert_review_event(
        self,
        farmer_id: str,
        case_id: str,
        crop_name: str,
        expert_name: str,
        institution: str,
        is_secondary_approved: bool = False,
        phi_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trigger expert review resolution alert to farmer."""
        if is_secondary_approved:
            event_type = "expert_review_secondary_approved"
            payload = {
                "crop_name": crop_name,
                "case_id": case_id,
                "phi_days": phi_days or 14,
            }
        else:
            event_type = "expert_review_completed"
            payload = {
                "crop_name": crop_name,
                "case_id": case_id,
                "expert_name": expert_name,
                "institution": institution,
            }

        dedup_key = f"expert_review_{case_id}_{event_type}"
        event = NotificationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            category=NotificationCategory.EXPERT_REVIEW,
            priority=NotificationPriority.HIGH,
            farmer_id=farmer_id,
            related_entity_id=case_id,
            payload=payload,
            dedup_key=dedup_key,
        )
        return self.publish_event(event)

    def emit_crop_activity_reminder_event(
        self,
        farmer_id: str,
        farm_name: str,
        crop_name: str,
        activity_title: str,
        scheduled_date: str,
        reminder_id: str,
        weather_risk_reason: Optional[str] = None,
        farm_id: Optional[str] = None,
        crop_cycle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger farm reminder notification."""
        if weather_risk_reason:
            event_type = "crop_reminder_weather_warning"
            prio = NotificationPriority.HIGH
            payload = {
                "farm_name": farm_name,
                "crop_name": crop_name,
                "activity_title": activity_title,
                "scheduled_date": scheduled_date,
                "weather_risk_reason": weather_risk_reason,
            }
        else:
            event_type = "crop_reminder_activity_due"
            prio = NotificationPriority.MEDIUM
            payload = {
                "farm_name": farm_name,
                "crop_name": crop_name,
                "activity_title": activity_title,
                "scheduled_date": scheduled_date,
            }

        dedup_key = f"reminder_{reminder_id}_{scheduled_date}"
        event = NotificationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            category=NotificationCategory.CROP_REMINDER,
            priority=prio,
            farmer_id=farmer_id,
            farm_id=farm_id,
            crop_cycle_id=crop_cycle_id,
            related_entity_id=reminder_id,
            payload=payload,
            dedup_key=dedup_key,
        )
        return self.publish_event(event)
