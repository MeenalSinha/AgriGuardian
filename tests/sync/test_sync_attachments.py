"""Tests for Phase 10 Offline Photograph Uploads, SHA-256 Checksums, and Draft Linking."""

import hashlib
import tempfile
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord, UserRole
from backend.app.schemas.sync import DeviceRegisterRequest, DeviceType
from backend.app.services.sync.sync_attachment_service import SyncAttachmentService
from backend.app.services.sync.sync_device_service import SyncDeviceService


@pytest.fixture
def attachment_env():
    """Create test repo, storage directory, and attachment service."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    storage_dir = tempfile.mkdtemp(prefix="sync_attach_")

    user1 = UserRecord(
        user_id="usr_attach_01",
        username="farmer_attach_01",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    user2 = UserRecord(
        user_id="usr_attach_02",
        username="farmer_attach_02",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user1)
    repo.create_user(user2)

    profile1 = FarmerProfile(
        farmer_id="fmr_attach_01",
        user_id="usr_attach_01",
        full_name="Photo Farmer",
        phone_number="+919876500001",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile1)

    dev_svc = SyncDeviceService(repo)
    dev_svc.register_device(
        user_id="usr_attach_01",
        farmer_id="fmr_attach_01",
        req=DeviceRegisterRequest(
            device_id="dev_cam_01",
            device_name="Field Camera Phone",
            device_type=DeviceType.ANDROID_CHROME,
            app_version="1.10.0",
        ),
    )

    att_svc = SyncAttachmentService(repo=repo, device_service=dev_svc, storage_dir=storage_dir)
    yield repo, att_svc, storage_dir
    try:
        import os
        import shutil
        os.unlink(tmp.name)
        shutil.rmtree(storage_dir, ignore_errors=True)
    except Exception:
        pass


def test_photo_upload_and_sha256(attachment_env):
    repo, att_svc, storage_dir = attachment_env

    # 1. Simulate JPEG photo bytes
    photo_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 200 + b"\xff\xd9"
    expected_sha256 = hashlib.sha256(photo_bytes).hexdigest()

    resp = att_svc.upload_attachment(
        user_id="usr_attach_01",
        farmer_id="fmr_attach_01",
        device_id="dev_cam_01",
        resource_type="crop_diagnosis",
        resource_id="diag_case_123",
        file_bytes=photo_bytes,
        filename="leaf_rust_field.jpg",
        content_type="image/jpeg",
    )

    assert resp.attachment_id.startswith("att_")
    assert resp.sha256_checksum == expected_sha256
    assert resp.byte_size == len(photo_bytes)
    assert resp.file_name == "leaf_rust_field.jpg"

    # Verify metadata retrieval
    meta = att_svc.get_attachment_metadata(resp.attachment_id, farmer_id="fmr_attach_01")
    assert meta.sha256_checksum == expected_sha256
    assert meta.resource_id == "diag_case_123"


def test_invalid_mime_type_rejection(attachment_env):
    repo, att_svc, storage_dir = attachment_env

    bad_bytes = b"MZ\x90\x00\x03\x00\x00\x00"

    with pytest.raises(HTTPException) as exc:
        att_svc.upload_attachment(
            user_id="usr_attach_01",
            farmer_id="fmr_attach_01",
            device_id="dev_cam_01",
            resource_type="crop_diagnosis",
            resource_id=None,
            file_bytes=bad_bytes,
            filename="malicious.exe",
            content_type="application/x-msdownload",
        )
    assert exc.value.status_code == 415
    assert "unsupported media type" in exc.value.detail.lower()


def test_attachment_size_limit_enforcement(attachment_env):
    repo, att_svc, storage_dir = attachment_env

    oversized_bytes = b"\x00" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit

    with pytest.raises(HTTPException) as exc:
        att_svc.upload_attachment(
            user_id="usr_attach_01",
            farmer_id="fmr_attach_01",
            device_id="dev_cam_01",
            resource_type="crop_diagnosis",
            resource_id=None,
            file_bytes=oversized_bytes,
            filename="huge_pano.jpg",
            content_type="image/jpeg",
        )
    assert exc.value.status_code == 413


def test_attachment_tenant_isolation(attachment_env):
    repo, att_svc, storage_dir = attachment_env

    photo_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x00\xff\xd9"
    resp = att_svc.upload_attachment(
        user_id="usr_attach_01",
        farmer_id="fmr_attach_01",
        device_id="dev_cam_01",
        resource_type="crop_diagnosis",
        resource_id=None,
        file_bytes=photo_bytes,
        filename="farmer1_leaf.jpg",
        content_type="image/jpeg",
    )

    # Farmer 2 cannot view or access Farmer 1's attachment
    with pytest.raises(HTTPException) as exc:
        att_svc.get_attachment_metadata(resp.attachment_id, farmer_id="fmr_attach_02")
    assert exc.value.status_code == 404
