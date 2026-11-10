"""Offline photograph and attachment upload, SHA-256 verification, and draft linking service."""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import AttachmentMetadataResponse, AttachmentUploadResponse
from backend.app.services.sync.sync_device_service import SyncDeviceService

logger = get_logger("sync_attachment_service")

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024  # 10 MB


class SyncAttachmentService:
    """Manages offline image uploads, deduplication via SHA-256, and links to diagnostic cases."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        device_service: Optional[SyncDeviceService] = None,
        storage_dir: Optional[str] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.device_service = device_service or SyncDeviceService(self.repo)
        settings = get_settings()
        self.storage_dir = Path(storage_dir or getattr(settings, "attachment_dir", "data/attachments"))
        self.storage_dir.mkdir(parents=True, exist_ok=True)

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
        """Validate, store, and record an attachment from an offline client."""
        self.device_service.validate_active_device(device_id, user_id)

        # Validate file size
        byte_size = len(file_bytes)
        if byte_size > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=f"File exceeds maximum allowed size of {MAX_ATTACHMENT_BYTES // (1024*1024)}MB.",
            )

        # Validate MIME type
        normalized_mime = content_type.lower()
        if normalized_mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported media type '{content_type}'. Allowed types: {sorted(list(ALLOWED_MIME_TYPES))}",
            )

        # Validate magic byte signatures (Phase 12 File Security)
        is_valid_signature = False
        if len(file_bytes) >= 4:
            if normalized_mime in ("image/jpeg", "image/jpg") and file_bytes.startswith(b"\xff\xd8\xff"):
                is_valid_signature = True
            elif normalized_mime == "image/png" and file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                is_valid_signature = True
            elif normalized_mime == "image/webp" and file_bytes.startswith(b"RIFF") and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
                is_valid_signature = True

        if not is_valid_signature:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File signature (magic bytes) mismatch: content does not match declared MIME type '{content_type}'.",
            )

        # Sanitize filename against path traversal
        clean_filename = os.path.basename(filename)

        # Compute SHA-256 checksum
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        attachment_id = f"att_{uuid.uuid4().hex[:12]}"

        # Storage filename uses attachment_id and sanitized extension
        ext = Path(clean_filename).suffix.lower() or ".jpg"
        stored_filename = f"{attachment_id}_{sha256[:8]}{ext}"
        storage_path = self.storage_dir / stored_filename

        with open(storage_path, "wb") as f:
            f.write(file_bytes)

        logger.info(
            f"Saved sync attachment {attachment_id} ({byte_size} bytes, SHA-256: {sha256[:10]}...) to {storage_path}"
        )

        record = self.repo.save_sync_attachment(
            attachment_id=attachment_id,
            farmer_id=farmer_id,
            device_id=device_id,
            resource_type=resource_type,
            resource_id=resource_id,
            sha256_checksum=sha256,
            file_name=filename,
            mime_type=normalized_mime,
            byte_size=byte_size,
            storage_path=str(storage_path),
        )

        return AttachmentUploadResponse(
            attachment_id=record["attachment_id"],
            file_name=record["file_name"],
            mime_type=record["mime_type"],
            byte_size=record["byte_size"],
            sha256_checksum=record["sha256_checksum"],
            storage_path=record["storage_path"],
            created_at=record["created_at"],
        )

    def get_attachment_metadata(
        self, attachment_id: str, farmer_id: str
    ) -> AttachmentMetadataResponse:
        """Retrieve attachment metadata with ownership check."""
        record = self.repo.get_sync_attachment(attachment_id, farmer_id=farmer_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Attachment '{attachment_id}' not found.",
            )
        return AttachmentMetadataResponse(
            attachment_id=record["attachment_id"],
            farmer_id=record["farmer_id"],
            device_id=record["device_id"],
            resource_type=record["resource_type"],
            resource_id=record.get("resource_id"),
            sha256_checksum=record["sha256_checksum"],
            file_name=record["file_name"],
            mime_type=record["mime_type"],
            byte_size=record["byte_size"],
            storage_path=record["storage_path"],
            is_uploaded=bool(record.get("is_uploaded", 1)),
            created_at=record["created_at"],
        )
