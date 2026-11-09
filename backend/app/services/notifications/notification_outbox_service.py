"""Transactional Outbox worker service executing lease-based delivery, retries, and dead-letter handling."""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.integrations.notifications.provider_registry import NotificationProviderRegistry
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.notification import (
    ChannelHealthInfo,
    NotificationChannel,
    NotificationDeliveryAttempt,
    OutboxProcessResult,
    OutboxStatus,
)

logger = get_logger("NotificationOutboxService")


class NotificationOutboxService:
    """Manages transactional outbox processing, atomic worker leases, exponential backoff, and dead-letter queues."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        provider_registry: Optional[NotificationProviderRegistry] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.registry = provider_registry or NotificationProviderRegistry()

    def process_outbox_batch(
        self,
        batch_size: int = 20,
        lease_duration_seconds: int = 60,
    ) -> OutboxProcessResult:
        """Atomically claim and dispatch ready outbox jobs."""
        start_time = time.perf_counter()
        claimed_items = self.repo.claim_pending_outbox_items(
            batch_size=batch_size,
            lease_duration_seconds=lease_duration_seconds,
        )

        delivered = 0
        failed = 0
        dead_letter = 0

        for item in claimed_items:
            outbox_id = item["outbox_id"]
            channel_str = item["channel"]
            channel = NotificationChannel(channel_str)
            retry_count = item["retry_count"]
            max_retries = item["max_retries"]
            phone = item.get("phone_number")
            pref_lang = item.get("preferred_language", "hi-IN")

            # Localize text based on farmer preference
            if pref_lang == "en-IN":
                title = item["title_en"]
                body = item["body_en"]
            else:
                title = item["title_hi"]
                body = item["body_hi"]

            provider = self.registry.get_provider(channel)
            attempt_start = time.perf_counter()
            result = provider.send(
                recipient_id=item["farmer_id"],
                recipient_phone=phone,
                title=title,
                body=body,
                language=pref_lang,
                action_url=item.get("action_url"),
                metadata=item.get("metadata", {}),
            )
            attempt_latency = (time.perf_counter() - attempt_start) * 1000.0

            # Record attempt log
            attempt_id = f"att_{outbox_id}_{retry_count}_{int(time.time())}"
            self.repo.record_delivery_attempt(
                NotificationDeliveryAttempt(
                    attempt_id=attempt_id,
                    outbox_id=outbox_id,
                    channel=channel,
                    provider_name=provider.provider_name,
                    status="success" if result.success else ("not_configured" if result.not_configured else "failure"),
                    latency_ms=round(attempt_latency, 2),
                    error_message=result.error_message,
                    provider_response_code=result.response_code,
                    attempted_at=datetime.now().isoformat(),
                )
            )

            if result.success:
                self.repo.update_outbox_status(outbox_id=outbox_id, status=OutboxStatus.DELIVERED.value)
                delivered += 1
                logger.info(f"Outbox {outbox_id} successfully delivered via {channel.value}.")
            else:
                new_retry = retry_count + 1
                if new_retry >= max_retries:
                    self.repo.update_outbox_status(
                        outbox_id=outbox_id,
                        status=OutboxStatus.DEAD_LETTER.value,
                        error_message=result.error_message,
                        increment_retry=True,
                    )
                    dead_letter += 1
                    logger.warning(
                        f"Outbox {outbox_id} moved to DEAD_LETTER after {new_retry} attempts. Error: {result.error_message}"
                    )
                else:
                    # Exponential backoff: 60s * 2^retry
                    backoff_seconds = 60 * (2 ** retry_count)
                    next_retry = datetime.now() + timedelta(seconds=backoff_seconds)
                    self.repo.update_outbox_status(
                        outbox_id=outbox_id,
                        status=OutboxStatus.FAILED.value,
                        next_retry_at=next_retry.isoformat(),
                        error_message=result.error_message,
                        increment_retry=True,
                    )
                    failed += 1
                    logger.info(
                        f"Outbox {outbox_id} failed attempt {new_retry}/{max_retries}. Next retry at {next_retry.isoformat()}."
                    )

        duration = (time.perf_counter() - start_time) * 1000.0
        return OutboxProcessResult(
            processed_count=len(claimed_items),
            delivered_count=delivered,
            failed_count=failed,
            dead_letter_count=dead_letter,
            duration_ms=round(duration, 2),
        )

    def get_channel_health_report(self) -> List[ChannelHealthInfo]:
        """Aggregate configuration readiness and actual delivery statistics."""
        base_health = self.registry.get_channels_health()
        telemetry = {t["channel"]: t for t in self.repo.get_channel_delivery_telemetry()}

        for h in base_health:
            ch_str = h.channel.value
            if ch_str in telemetry:
                h.total_dispatched = telemetry[ch_str]["successful"]
                h.total_failed = telemetry[ch_str]["failed"]

        return base_health

    def retry_outbox_item(self, outbox_id: str) -> bool:
        """Manually trigger retry for a failed or dead-letter message."""
        return self.repo.retry_outbox_item(outbox_id)
