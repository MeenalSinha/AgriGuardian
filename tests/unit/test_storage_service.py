"""Unit tests for storage service implementation and directory traversal protection."""

import os
import tempfile
import pytest
from pathlib import Path

from backend.app.services.storage.storage_service import (
    LocalStorageService,
    S3StorageService,
    get_storage_service,
)


@pytest.fixture
def temp_storage():
    """Create isolated temporary directory for local storage testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        service = LocalStorageService(base_dir=tmpdir)
        yield service, Path(tmpdir)


def test_local_storage_save_and_read(temp_storage):
    """Verify that file save and read work correctly."""
    service, base_path = temp_storage
    data = b"Hello AgriGuardian Storage"
    rel_path = "reports/2026/test.txt"

    saved_path = service.save(rel_path, data)
    assert Path(saved_path).exists()
    assert service.exists(rel_path) is True

    read_data = service.read(rel_path)
    assert read_data == data

    # Test URL generation
    url = service.get_url(rel_path)
    assert url == "/files/reports/2026/test.txt"


def test_local_storage_delete(temp_storage):
    """Verify that file deletion functions accurately."""
    service, _ = temp_storage
    rel_path = "delete_me.dat"
    service.save(rel_path, b"temporary data")
    assert service.exists(rel_path) is True

    deleted = service.delete(rel_path)
    assert deleted is True
    assert service.exists(rel_path) is False

    # Deleting non-existent file returns False
    assert service.delete(rel_path) is False


def test_local_storage_read_non_existent(temp_storage):
    """Verify that reading a non-existent file raises FileNotFoundError."""
    service, _ = temp_storage
    with pytest.raises(FileNotFoundError):
        service.read("non_existent_file.xyz")


def test_local_storage_path_traversal_protection(temp_storage):
    """Verify that directory traversal attempts are blocked."""
    service, base_path = temp_storage

    traversal_payloads = [
        "../outside.txt",
        "../../etc/passwd",
        "/etc/shadow",
        "..\\..\\windows\\win.ini",
        "nested/../../secret.key",
    ]

    for payload in traversal_payloads:
        with pytest.raises(ValueError, match="Path traversal detected"):
            service._resolve_safe_path(payload)


def test_s3_storage_unconfigured_raises_runtime_error():
    """Verify that unconfigured S3 storage raises RuntimeError instead of returning false success."""
    s3 = S3StorageService(bucket_name="my-farm-bucket")

    # In an environment without AWS credentials or boto3, operations must fail cleanly
    with pytest.raises(RuntimeError, match="S3 cloud storage is not configured"):
        s3.save("test.pdf", b"content")

    with pytest.raises(RuntimeError, match="S3 cloud storage is not configured"):
        s3.read("test.pdf")

    with pytest.raises(RuntimeError, match="S3 cloud storage is not configured"):
        s3.exists("test.pdf")

    with pytest.raises(RuntimeError, match="S3 cloud storage is not configured"):
        s3.delete("test.pdf")

