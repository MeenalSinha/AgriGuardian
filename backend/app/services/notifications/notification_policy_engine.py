"""Policy and decision engine for notification filtering, deduplication, quiet hours, and channel routing."""

from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.notification import (
    NotificationCategory,
    NotificationChannel,
    NotificationEvent,
    NotificationPreference,
    NotificationPriority,
)

logger = get_logger("NotificationPolicyEngine")


@dataclass
class PolicyEvaluationResult:
    """Outcome of policy evaluation for an incoming event."""
    allowed: bool
    suppression_reason: Optional[str] = None
    target_channels: List[NotificationChannel] = None
    is_quiet_hours: bool = False
    bypass_quiet_hours: bool = False
    defer_until: Optional[str] = None


class NotificationPolicyEngine:
    """Enforces deduplication, quiet hours, consent checks, and channel selection."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def evaluate(
        self,
        event: NotificationEvent,
        now: Optional[datetime] = None,
    ) -> PolicyEvaluationResult:
        """Evaluate whether to notify farmer and over which channels."""
        current_time = now or datetime.now()

        # 1. Deduplication check
        if event.dedup_key:
            recent_dup = self.repo.find_recent_event_by_dedup_key(
                event.dedup_key,
                window_minutes=60,
                exclude_event_id=event.event_id,
            )
            if recent_dup and event.priority != NotificationPriority.EMERGENCY:
                logger.info(f"Notification event {event.event_id} suppressed: duplicate dedup_key '{event.dedup_key}' within 60m.")
                return PolicyEvaluationResult(
                    allowed=False,
                    suppression_reason=f"Duplicate notification within sliding window (key: {event.dedup_key})",
                    target_channels=[],
                )

        # 2. Farmer preferences & Category enablement
        pref = self.repo.get_notification_preference(event.farmer_id)
        if event.category not in pref.categories_enabled and event.priority != NotificationPriority.EMERGENCY:
            logger.info(f"Notification event {event.event_id} suppressed: category '{event.category.value}' disabled by farmer.")
            return PolicyEvaluationResult(
                allowed=False,
                suppression_reason=f"Category '{event.category.value}' opted out by farmer preference.",
                target_channels=[],
            )

        # 3. Quiet Hours evaluation
        is_quiet, defer_time = self._check_quiet_hours(pref, current_time)
        bypass_quiet = False
        if is_quiet:
            if event.priority == NotificationPriority.EMERGENCY and pref.emergency_override:
                bypass_quiet = True
                logger.info(f"Event {event.event_id} has EMERGENCY priority; overriding quiet hours for farmer {event.farmer_id}.")
            else:
                logger.info(f"Quiet hours active for farmer {event.farmer_id} ({pref.quiet_hours_start}-{pref.quiet_hours_end}). Restricting to in-app or deferring.")

        # 4. Channel Selection & Consent Cross-Check
        active_consents = {c.channel: c.is_granted for c in self.repo.get_channel_consents(event.farmer_id)}
        target_channels: List[NotificationChannel] = []

        for ch in pref.channels_enabled:
            # IN_APP is always permitted
            if ch == NotificationChannel.IN_APP:
                target_channels.append(ch)
                continue

            # If quiet hours are active and not bypassed, omit external noisy channels
            if is_quiet and not bypass_quiet:
                continue

            # External channels require explicit consent or verified opt-in
            # If consent record exists and is_granted is False, omit
            if ch in active_consents and not active_consents[ch]:
                continue

            target_channels.append(ch)

        # If priority is EMERGENCY and voice is not in channels, add VOICE_CALL or SMS if available
        if event.priority == NotificationPriority.EMERGENCY:
            if NotificationChannel.SMS not in target_channels:
                target_channels.append(NotificationChannel.SMS)

        if not target_channels:
            # Always ensure at least IN_APP if enabled in preferences
            target_channels = [NotificationChannel.IN_APP]

        return PolicyEvaluationResult(
            allowed=True,
            target_channels=target_channels,
            is_quiet_hours=is_quiet,
            bypass_quiet_hours=bypass_quiet,
            defer_until=defer_time.isoformat() if (is_quiet and not bypass_quiet and defer_time) else None,
        )

    def _check_quiet_hours(
        self,
        pref: NotificationPreference,
        current_time: datetime,
    ) -> Tuple[bool, Optional[datetime]]:
        """Determine if current time falls within farmer quiet hours window."""
        if not pref.quiet_hours_enabled:
            return False, None

        try:
            start_parts = [int(p) for p in pref.quiet_hours_start.split(":")]
            end_parts = [int(p) for p in pref.quiet_hours_end.split(":")]
            start_t = dtime(hour=start_parts[0], minute=start_parts[1])
            end_t = dtime(hour=end_parts[0], minute=end_parts[1])
        except Exception:
            start_t = dtime(21, 0)
            end_t = dtime(6, 0)

        curr_t = current_time.time()

        # Window crosses midnight (e.g. 21:00 to 06:00)
        if start_t > end_t:
            is_in_window = curr_t >= start_t or curr_t < end_t
            if is_in_window:
                # Calculate resume datetime (today if curr_t < end_t, or tomorrow if curr_t >= start_t)
                if curr_t < end_t:
                    defer_dt = current_time.replace(hour=end_t.hour, minute=end_t.minute, second=0, microsecond=0)
                else:
                    from datetime import timedelta
                    tomorrow = current_time + timedelta(days=1)
                    defer_dt = tomorrow.replace(hour=end_t.hour, minute=end_t.minute, second=0, microsecond=0)
                return True, defer_dt
        else:
            is_in_window = start_t <= curr_t < end_t
            if is_in_window:
                defer_dt = current_time.replace(hour=end_t.hour, minute=end_t.minute, second=0, microsecond=0)
                return True, defer_dt

        return False, None
