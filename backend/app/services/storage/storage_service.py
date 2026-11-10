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
        if not relative_path or os.path.isabs(relative_path) or relative_path.startswith("/") or relative_path.startswith("\\"):
            raise ValueError(f"Path traversal detected: {relative_path} is an absolute or invalid path.")

        clean_rel = os.path.normpath(relative_path)
        target = (self.base_dir / clean_rel).resolve()
        try:
            target.relative_to(self.base_dir)
        except ValueError:
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
        self._client = None
        try:
            import boto3
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                self._client = boto3.client("s3", region_name=region)
        except ImportError:
            pass

    def _ensure_configured(self) -> None:
        if not self._client:
            raise RuntimeError(
                "S3 cloud storage is not configured. Provide AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                "and ensure 'boto3' is installed to use S3StorageService."
            )

    def save(self, relative_path: str, data: bytes) -> str:
        self._ensure_configured()
        self._client.put_object(Bucket=self.bucket_name, Key=relative_path.lstrip("/"), Body=data)
        logger.info(f"Uploaded {len(data)} bytes to s3://{self.bucket_name}/{relative_path}")
        return f"s3://{self.bucket_name}/{relative_path.lstrip('/')}"

    def read(self, relative_path: str) -> bytes:
        self._ensure_configured()
        obj = self._client.get_object(Bucket=self.bucket_name, Key=relative_path.lstrip("/"))
        return obj["Body"].read()

    def exists(self, relative_path: str) -> bool:
        self._ensure_configured()
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=relative_path.lstrip("/"))
            return True
        except Exception:
            return False

    def delete(self, relative_path: str) -> bool:
        self._ensure_configured()
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=relative_path.lstrip("/"))
            return True
        except Exception:
            return False

    def get_url(self, relative_path: str) -> str:
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{relative_path.lstrip('/')}"


_storage_instance: Optional[BaseStorageService] = None


def get_storage_service() -> BaseStorageService:
    """Retrieve configured storage service instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = LocalStorageService()
    return _storage_instance

