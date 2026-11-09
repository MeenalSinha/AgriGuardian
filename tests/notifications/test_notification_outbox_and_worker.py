"""Unit and integration tests for Transactional Outbox, worker leasing, retry backoff, and dead-letter queues."""

import os
import tempfile
import time
from datetime import datetime, timedelta
import pytest

from backend.app.integrations.notifications.base_provider import BaseNotificationProvider, ProviderDeliveryResult
from backend.app.integrations.notifications.provider_registry import NotificationProviderRegistry
from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord
from backend.app.schemas.notification import (
    NotificationCategory,
    NotificationChannel,
    NotificationOutboxItem,
    NotificationPriority,
    NotificationRecord,
    NotificationStatus,
    OutboxStatus,
)
from backend.app.services.notifications.notification_outbox_service import NotificationOutboxService


class MockFailingProvider(BaseNotificationProvider):
    """Provider configured to simulate repeated physical network failures."""

    def __init__(self, channel: NotificationChannel):
        super().__init__(channel=channel, provider_name="Failing Gateway")

    def is_configured(self) -> bool:
        return True

    def send(self, *args, **kwargs) -> ProviderDeliveryResult:
        return ProviderDeliveryResult(
            success=False,
            channel=self.channel,
            provider_name=self.provider_name,
            latency_ms=15.0,
            error_message="Simulated connection timeout (504 Gateway Timeout)",
            response_code="504_TIMEOUT",
        )


@pytest.fixture
def outbox_fixture():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)

    now_str = datetime.now().isoformat()
    # Seed user, farmer and notification
    user = UserRecord(
        user_id="usr_outbox_01",
        username="vikram_choudhary",
        password_hash="fakehash",
        salt="fakesalt",
        role="farmer",
        created_at=now_str,
    )
    repo.create_user(user)

    farmer = FarmerProfile(
        farmer_id="fmr_outbox_01",
        user_id="usr_outbox_01",
        full_name="Vikram Choudhary",
        phone_number="9876543210",
        preferred_language="hi-IN",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at=now_str,
        updated_at=now_str,
    )
    repo.create_farmer_profile(farmer)

    notif = NotificationRecord(
        notification_id="notif_obx_01",
        farmer_id="fmr_outbox_01",
        category=NotificationCategory.WEATHER_ALERT,
        priority=NotificationPriority.HIGH,
        title_en="High Wind Warning",
        title_hi="तेज हवा की चेतावनी",
        body_en="High winds exceeding 45km/h expected.",
        body_hi="45 किमी/घंटा से अधिक की तेज हवाओं की संभावना।",
        created_at=datetime.now().isoformat(),
    )
    repo.create_notification(notif)

    registry = NotificationProviderRegistry()
    service = NotificationOutboxService(repo=repo, provider_registry=registry)

    yield {
        "repo": repo,
        "service": service,
        "registry": registry,
        "farmer": farmer,
        "notif": notif,
    }

    try:
        os.remove(path)
    except Exception:
        pass


def test_outbox_atomic_lease_claiming_prevents_double_processing(outbox_fixture):
    """Verify that claiming outbox jobs is atomic and locked under worker lease."""
    repo = outbox_fixture["repo"]

    # Enqueue outbox item
    item = NotificationOutboxItem(
        outbox_id="obx_claim_01",
        notification_id="notif_obx_01",
        farmer_id="fmr_outbox_01",
        channel=NotificationChannel.IN_APP,
        status=OutboxStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    repo.enqueue_outbox_message(item)

    # Worker 1 claims item
    claimed_1 = repo.claim_pending_outbox_items(batch_size=10, lease_duration_seconds=30)
    assert len(claimed_1) == 1
    assert claimed_1[0]["outbox_id"] == "obx_claim_01"

    # Worker 2 attempts concurrent claim - returns 0 items because lease is active
    claimed_2 = repo.claim_pending_outbox_items(batch_size=10, lease_duration_seconds=30)
    assert len(claimed_2) == 0


def test_expired_lease_worker_recovery(outbox_fixture):
    """Verify crashed worker leases (lease_expires_at < now) are reclaimed by next worker."""
    repo = outbox_fixture["repo"]

    # Enqueue outbox item with expired lease
    past_time = (datetime.now() - timedelta(minutes=5)).isoformat()
    item = NotificationOutboxItem(
        outbox_id="obx_expired_01",
        notification_id="notif_obx_01",
        farmer_id="fmr_outbox_01",
        channel=NotificationChannel.IN_APP,
        status=OutboxStatus.PROCESSING,
        lease_expires_at=past_time,
        created_at=past_time,
        updated_at=past_time,
    )
    repo.enqueue_outbox_message(item)

    # Worker claims and recovers the abandoned job
    recovered = repo.claim_pending_outbox_items(batch_size=10, lease_duration_seconds=30)
    assert len(recovered) == 1
    assert recovered[0]["outbox_id"] == "obx_expired_01"


def test_repeated_failure_routes_to_dead_letter_queue(outbox_fixture):
    """Verify message failure triggers exponential backoff and moves to DEAD_LETTER after 3 attempts."""
    repo = outbox_fixture["repo"]
    registry = outbox_fixture["registry"]

    # Replace SMS provider with simulated failing provider
    registry._providers[NotificationChannel.SMS] = MockFailingProvider(NotificationChannel.SMS)
    service = NotificationOutboxService(repo=repo, provider_registry=registry)

    item = NotificationOutboxItem(
        outbox_id="obx_fail_01",
        notification_id="notif_obx_01",
        farmer_id="fmr_outbox_01",
        channel=NotificationChannel.SMS,
        status=OutboxStatus.PENDING,
        retry_count=0,
        max_retries=3,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    repo.enqueue_outbox_message(item)

    # Attempt 1: Fails, sets status = failed, backoff scheduled
    res1 = service.process_outbox_batch(batch_size=1)
    assert res1.failed_count == 1
    row1 = repo.list_outbox_items()[0]
    assert row1["status"] == "failed"
    assert row1["retry_count"] == 1

    # Simulate retry time arrival
    repo.update_outbox_status("obx_fail_01", "pending", next_retry_at=None)

    # Attempt 2: Fails again
    res2 = service.process_outbox_batch(batch_size=1)
    assert res2.failed_count == 1
    row2 = repo.list_outbox_items()[0]
    assert row2["retry_count"] == 2

    # Simulate retry time arrival
    repo.update_outbox_status("obx_fail_01", "pending", next_retry_at=None)

    # Attempt 3: Max retries exhausted -> routes to DEAD_LETTER
    res3 = service.process_outbox_batch(batch_size=1)
    assert res3.dead_letter_count == 1
    row3 = repo.list_outbox_items()[0]
    assert row3["status"] == "dead_letter"
    assert "Simulated connection timeout" in row3["last_error"]


def test_manual_retry_from_dead_letter(outbox_fixture):
    """Verify administrator can reset a dead-letter message back to pending."""
    repo = outbox_fixture["repo"]
    service = outbox_fixture["service"]

    item = NotificationOutboxItem(
        outbox_id="obx_dead_01",
        notification_id="notif_obx_01",
        farmer_id="fmr_outbox_01",
        channel=NotificationChannel.SMS,
        status=OutboxStatus.DEAD_LETTER,
        retry_count=3,
        last_error="Provider offline",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
    repo.enqueue_outbox_message(item)

    # Administrator resets
    success = service.retry_outbox_item("obx_dead_01")
    assert success is True

    row = repo.list_outbox_items()[0]
    assert row["status"] == "pending"
    assert row["retry_count"] == 0
