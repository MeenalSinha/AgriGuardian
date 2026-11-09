"""Device registration, authentication validation, and revocation service for Phase 10."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import DeviceRegisterRequest, DeviceResponse

logger = get_logger("sync_device_service")


class SyncDeviceService:
    """Manages physical and PWA device lifecycles, association to farmers, and token validation."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def register_device(
        self, user_id: str, farmer_id: Optional[str], req: DeviceRegisterRequest
    ) -> DeviceResponse:
        """Register a new device or refresh metadata of existing device for an authenticated user."""
        logger.info(f"Registering sync device {req.device_id} for user {user_id} (farmer: {farmer_id})")
        dev_row = self.repo.register_device(
            device_id=req.device_id,
            user_id=user_id,
            farmer_id=farmer_id,
            device_name=req.device_name,
            device_type=req.device_type.value if hasattr(req.device_type, "value") else str(req.device_type),
            app_version=req.app_version,
        )
        return DeviceResponse(
            device_id=dev_row["device_id"],
            user_id=dev_row["user_id"],
            farmer_id=dev_row.get("farmer_id"),
            device_name=dev_row["device_name"],
            device_type=dev_row["device_type"],
            app_version=dev_row["app_version"],
            last_sync_at=dev_row.get("last_sync_at"),
            is_revoked=bool(dev_row.get("is_revoked", 0)),
            created_at=dev_row["created_at"],
            updated_at=dev_row["updated_at"],
        )

    def list_user_devices(self, user_id: str) -> List[DeviceResponse]:
        """List all devices registered to this user."""
        rows = self.repo.get_devices_for_user(user_id)
        return [
            DeviceResponse(
                device_id=r["device_id"],
                user_id=r["user_id"],
                farmer_id=r.get("farmer_id"),
                device_name=r["device_name"],
                device_type=r["device_type"],
                app_version=r["app_version"],
                last_sync_at=r.get("last_sync_at"),
                is_revoked=bool(r.get("is_revoked", 0)),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def revoke_device(self, device_id: str, user_id: str) -> bool:
        """Revoke a device. Revoked devices cannot perform future pulls or pushes."""
        logger.warning(f"User {user_id} requested revocation of device {device_id}")
        revoked = self.repo.revoke_device(device_id, user_id)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found or not owned by user {user_id}",
            )
        return True

    def validate_active_device(self, device_id: str, user_id: str) -> Dict[str, Any]:
        """Validate that device exists, belongs to user_id, and is NOT revoked."""
        device = self.repo.get_device(device_id)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unrecognized device '{device_id}'. Please register device first.",
            )
        if device["user_id"] != user_id:
            logger.error(f"Device {device_id} is registered to {device['user_id']}, not caller {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Device registration belongs to a different user account.",
            )
        if device.get("is_revoked", 0) == 1:
            logger.warning(f"Device {device_id} is revoked.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This device has been revoked and cannot synchronize. Please contact support or re-register.",
            )
        return device
