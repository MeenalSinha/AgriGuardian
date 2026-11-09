"""Bootstrap and incremental pull synchronization service for Phase 10."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import (
    ChangeFeedItem,
    SyncBootstrapResponse,
    SyncPullRequest,
    SyncPullResponse,
)
from backend.app.services.sync.sync_device_service import SyncDeviceService

logger = get_logger("sync_pull_service")


class SyncPullService:
    """Provides initial offline bootstrap and incremental change feed subscriptions."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        device_service: Optional[SyncDeviceService] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.device_service = device_service or SyncDeviceService(self.repo)

    def bootstrap_farmer_data(
        self, user_id: str, farmer_id: str, device_id: str
    ) -> SyncBootstrapResponse:
        """Fetch complete authorized offline dataset for initial device sync."""
        self.device_service.validate_active_device(device_id, user_id)

        logger.info(f"Assembling offline bootstrap bundle for farmer {farmer_id} on device {device_id}")
        bundle = self.repo.get_bootstrap_bundle(farmer_id=farmer_id, user_id=user_id)

        # Update device cursor to current max sequence_id
        cursor_seq = bundle.get("sequence_cursor", 0)
        self.repo.update_sync_cursor(device_id, "global", cursor_seq)
        self.repo.update_device_sync_time(device_id)

        return SyncBootstrapResponse(
            farmer_profile=bundle.get("farmer_profile"),
            farms=bundle.get("farms", []),
            plots=bundle.get("plots", []),
            crop_cycles=bundle.get("crop_cycles", []),
            activities=bundle.get("activities", []),
            reminders=bundle.get("reminders", []),
            notifications=bundle.get("notifications", []),
            advisories=bundle.get("advisories", []),
            expenses=bundle.get("expenses", []),
            harvests=bundle.get("harvests", []),
            sales=bundle.get("sales", []),
            budgets=bundle.get("budgets", []),
            server_timestamp=bundle.get("server_timestamp", datetime.now(timezone.utc).isoformat()),
            sequence_cursor=cursor_seq,
        )

    def pull_incremental_changes(
        self, user_id: str, farmer_id: str, req: SyncPullRequest
    ) -> SyncPullResponse:
        """Fetch changes that occurred since device's last acknowledged sequence_id."""
        self.device_service.validate_active_device(req.device_id, user_id)

        logger.info(
            f"Pulling changes for farmer {farmer_id} on device {req.device_id} since seq {req.since_sequence_id} (limit {req.limit})"
        )
        raw_changes, max_seq, has_more = self.repo.get_change_feed(
            farmer_id=farmer_id,
            after_sequence_id=req.since_sequence_id,
            limit=req.limit,
        )

        items: List[ChangeFeedItem] = []
        for r in raw_changes:
            payload = None
            if r.get("payload_json"):
                try:
                    payload = json.loads(r["payload_json"])
                except Exception:
                    payload = None

            items.append(
                ChangeFeedItem(
                    sequence_id=r["sequence_id"],
                    resource_type=r["resource_type"],
                    resource_id=r["resource_id"],
                    farmer_id=r["farmer_id"],
                    operation=r["operation"],
                    version_number=r.get("version_number", 1),
                    payload=payload,
                    created_at=r["created_at"],
                )
            )

        if max_seq > req.since_sequence_id:
            self.repo.update_sync_cursor(req.device_id, "global", max_seq)

        self.repo.update_device_sync_time(req.device_id)

        now_str = datetime.now(timezone.utc).isoformat()
        return SyncPullResponse(
            changes=items,
            new_sequence_id=max_seq,
            has_more=has_more,
            server_timestamp=now_str,
        )
