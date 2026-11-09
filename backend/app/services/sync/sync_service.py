"""Master synchronization service facade for Phase 10 Offline-First Architecture."""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import (
    AttachmentMetadataResponse,
    AttachmentUploadResponse,
    ConflictItem,
    ConflictResolveRequest,
    ConflictResolveResponse,
    DeviceRegisterRequest,
    DeviceResponse,
    SyncBootstrapResponse,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
    SyncTelemetryReport,
)
from backend.app.services.sync.sync_attachment_service import SyncAttachmentService
from backend.app.services.sync.sync_conflict_service import SyncConflictService
from backend.app.services.sync.sync_device_service import SyncDeviceService
from backend.app.services.sync.sync_pull_service import SyncPullService
from backend.app.services.sync.sync_push_service import SyncPushService

logger = get_logger("sync_service")

_sync_service_instance: Optional["SyncService"] = None


class SyncService:
    """Unified offline synchronization coordinator."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()
        self.device_service = SyncDeviceService(self.repo)
        self.conflict_service = SyncConflictService(self.repo)
        self.pull_service = SyncPullService(self.repo, self.device_service)
        self.push_service = SyncPushService(self.repo, self.device_service, self.conflict_service)
        self.attachment_service = SyncAttachmentService(self.repo, self.device_service)

    @classmethod
    def get_instance(cls, repo: Optional[FarmDatabaseRepository] = None) -> "SyncService":
        global _sync_service_instance
        if _sync_service_instance is None or repo is not None:
            _sync_service_instance = cls(repo=repo)
        return _sync_service_instance

    # Device Operations
    def register_device(
        self, user_id: str, farmer_id: Optional[str], req: DeviceRegisterRequest
    ) -> DeviceResponse:
        return self.device_service.register_device(user_id=user_id, farmer_id=farmer_id, req=req)

    def list_user_devices(self, user_id: str) -> List[DeviceResponse]:
        return self.device_service.list_user_devices(user_id=user_id)

    def revoke_device(self, device_id: str, user_id: str) -> bool:
        return self.device_service.revoke_device(device_id=device_id, user_id=user_id)

    # Pull & Bootstrap
    def get_bootstrap(
        self, user_id: str, farmer_id: str, device_id: str
    ) -> SyncBootstrapResponse:
        start_t = time.time()
        res = self.pull_service.bootstrap_farmer_data(user_id=user_id, farmer_id=farmer_id, device_id=device_id)
        duration_ms = int((time.time() - start_t) * 1000)
        self.repo.record_sync_telemetry(
            session_id=f"tel_{device_id}_{uuid.uuid4().hex[:10]}",
            device_id=device_id,
            farmer_id=farmer_id,
            sync_type="bootstrap",
            duration_ms=duration_ms,
            pull_count=len(res.farms) + len(res.activities) + len(res.reminders),
            push_count=0,
            conflict_count=0,
            bytes_transferred=1024,
            status="success",
        )
        return res

    def pull_changes(
        self, user_id: str, farmer_id: str, req: SyncPullRequest
    ) -> SyncPullResponse:
        start_t = time.time()
        res = self.pull_service.pull_incremental_changes(user_id=user_id, farmer_id=farmer_id, req=req)
        duration_ms = int((time.time() - start_t) * 1000)
        self.repo.record_sync_telemetry(
            session_id=f"tel_{req.device_id}_{uuid.uuid4().hex[:10]}",
            device_id=req.device_id,
            farmer_id=farmer_id,
            sync_type="pull",
            duration_ms=duration_ms,
            pull_count=len(res.changes),
            push_count=0,
            conflict_count=0,
            bytes_transferred=512,
            status="success",
        )
        return res

    # Push Mutations
    def push_mutations(
        self, user_id: str, farmer_id: str, req: SyncPushRequest
    ) -> SyncPushResponse:
        start_t = time.time()
        res = self.push_service.process_push_mutations(user_id=user_id, farmer_id=farmer_id, req=req)
        duration_ms = int((time.time() - start_t) * 1000)
        self.repo.record_sync_telemetry(
            session_id=f"tel_{req.device_id}_{uuid.uuid4().hex[:10]}",
            device_id=req.device_id,
            farmer_id=farmer_id,
            sync_type="push",
            duration_ms=duration_ms,
            pull_count=0,
            push_count=res.processed_count,
            conflict_count=res.conflict_count,
            bytes_transferred=len(req.mutations) * 256,
            status="success" if res.conflict_count == 0 else "partial_conflict",
        )
        return res

    # Conflict Management
    def list_conflicts(self, farmer_id: str, status_filter: Optional[str] = "pending") -> List[ConflictItem]:
        return self.conflict_service.list_conflicts(farmer_id=farmer_id, status_filter=status_filter)

    def get_conflict(self, conflict_id: str, farmer_id: str) -> ConflictItem:
        return self.conflict_service.get_conflict(conflict_id=conflict_id, farmer_id=farmer_id)

    def resolve_conflict(
        self, conflict_id: str, farmer_id: str, req: ConflictResolveRequest
    ) -> ConflictResolveResponse:
        return self.conflict_service.resolve_conflict(conflict_id=conflict_id, farmer_id=farmer_id, req=req)

    # Attachments
    def upload_attachment(
        self,
        user_id: str,
        farmer_id: str,
        device_id: str,
        resource_type: str,
        resource_id: Optional[str],
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> AttachmentUploadResponse:
        return self.attachment_service.upload_attachment(
            user_id=user_id,
            farmer_id=farmer_id,
            device_id=device_id,
            resource_type=resource_type,
            resource_id=resource_id,
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )

    def get_attachment_metadata(self, attachment_id: str, farmer_id: str) -> AttachmentMetadataResponse:
        return self.attachment_service.get_attachment_metadata(attachment_id=attachment_id, farmer_id=farmer_id)

    # Health & Connectivity Ping
    def get_sync_status(self) -> Dict[str, Any]:
        """Lightweight endpoint used by dual online detection ping."""
        return {
            "status": "healthy",
            "service": "agri_guardian_sync",
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol_version": "1.10.0",
        }
