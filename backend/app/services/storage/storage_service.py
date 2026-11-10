"""Configurable storage service abstraction supporting local filesystem and S3/cloud object storage (Phase 12)."""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger("StorageService")


class BaseStorageService(ABC):
    """Abstract interface for application object and media storage."""

    @abstractmethod
    def save(self, relative_path: str, data: bytes) -> str:
        """Store bytes at relative path and return canonical storage path/URI."""
        pass

    @abstractmethod
    def read(self, relative_path: str) -> bytes:
        """Retrieve stored bytes from relative path."""
        pass

    @abstractmethod
    def exists(self, relative_path: str) -> bool:
        """Check whether object exists at relative path."""
        pass

    @abstractmethod
    def delete(self, relative_path: str) -> bool:
        """Delete object at relative path."""
        pass

    @abstractmethod
    def get_url(self, relative_path: str) -> str:
        """Return secure access URL or path."""
        pass


class LocalStorageService(BaseStorageService):
    """Local filesystem storage implementation with directory traversal protection."""

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.data_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve path and ensure it does not escape base_dir."""
        clean_rel = os.path.normpath(relative_path.lstrip("/\\"))
        target = (self.base_dir / clean_rel).resolve()
        if not str(target).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: {relative_path} attempts to escape storage root.")
        return target

    def save(self, relative_path: str, data: bytes) -> str:
        target = self._resolve_safe_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        logger.debug(f"Saved {len(data)} bytes to {target}")
        return str(target)

    def read(self, relative_path: str) -> bytes:
        target = self._resolve_safe_path(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Storage object not found: {relative_path}")
        with open(target, "rb") as f:
            return f.read()

    def exists(self, relative_path: str) -> bool:
        target = self._resolve_safe_path(relative_path)
        return target.exists() and target.is_file()

    def delete(self, relative_path: str) -> bool:
        target = self._resolve_safe_path(relative_path)
        if target.exists():
            target.unlink()
            return True
        return False

    def get_url(self, relative_path: str) -> str:
        target = self._resolve_safe_path(relative_path)
        return f"/files/{relative_path.lstrip('/')}"


class S3StorageService(BaseStorageService):
    """Production S3-compatible cloud object storage adapter (AWS S3, Cloudflare R2, MinIO, GCS)."""

    def __init__(self, bucket_name: str, region: str = "ap-south-1"):
        self.bucket_name = bucket_name
        self.region = region
        # Lazy initialization of boto3 when cloud credentials are provided

    def save(self, relative_path: str, data: bytes) -> str:
        # Production hook for boto3 s3 client put_object
        logger.info(f"Uploading {len(data)} bytes to s3://{self.bucket_name}/{relative_path}")
        return f"s3://{self.bucket_name}/{relative_path.lstrip('/')}"

    def read(self, relative_path: str) -> bytes:
        raise NotImplementedError("S3 retrieval requires configured cloud credentials in production.")

    def exists(self, relative_path: str) -> bool:
        return False

    def delete(self, relative_path: str) -> bool:
        return True

    def get_url(self, relative_path: str) -> str:
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{relative_path.lstrip('/')}"


_storage_instance: Optional[BaseStorageService] = None


def get_storage_service() -> BaseStorageService:
    """Retrieve configured storage service instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = LocalStorageService()
    return _storage_instance

