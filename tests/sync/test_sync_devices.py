"""Tests for Phase 10 Device Registration, Lifecycle, and Revocation."""

import tempfile
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import FarmerProfile, LiteracyPreference, UserRecord, UserRole
from backend.app.schemas.sync import DeviceRegisterRequest, DeviceType
from backend.app.services.sync.sync_device_service import SyncDeviceService


@pytest.fixture
def sync_env():
    """Create isolated test repository and device service."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    user1 = UserRecord(
        user_id="usr_farmer_01",
        username="farmer_ramesh",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    user2 = UserRecord(
        user_id="usr_farmer_02",
        username="farmer_suresh",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user1)
    repo.create_user(user2)

    profile1 = FarmerProfile(
        farmer_id="fmr_01",
        user_id="usr_farmer_01",
        full_name="Ramesh Patel",
        phone_number="+919876543210",
        state="Madhya Pradesh",
        district="Sehore",
        primary_language="hi",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile1)

    svc = SyncDeviceService(repo=repo)
    yield repo, svc
    try:
        import os
        os.unlink(tmp.name)
    except Exception:
        pass


def test_device_registration_and_listing(sync_env):
    repo, svc = sync_env

    req = DeviceRegisterRequest(
        device_id="dev_redmi_10",
        device_name="Ramesh's Redmi Note 10",
        device_type=DeviceType.ANDROID_CHROME,
        app_version="1.10.0",
    )
    resp = svc.register_device(user_id="usr_farmer_01", farmer_id="fmr_01", req=req)

    assert resp.device_id == "dev_redmi_10"
    assert resp.user_id == "usr_farmer_01"
    assert resp.farmer_id == "fmr_01"
    assert resp.device_name == "Ramesh's Redmi Note 10"
    assert resp.is_revoked is False

    devices = svc.list_user_devices(user_id="usr_farmer_01")
    assert len(devices) == 1
    assert devices[0].device_id == "dev_redmi_10"


def test_device_registration_upsert(sync_env):
    repo, svc = sync_env

    req1 = DeviceRegisterRequest(
        device_id="dev_samsung_01",
        device_name="Samsung M31",
        device_type=DeviceType.PWA_MOBILE,
        app_version="1.9.0",
    )
    svc.register_device(user_id="usr_farmer_01", farmer_id="fmr_01", req=req1)

    # Re-register with upgraded version
    req2 = DeviceRegisterRequest(
        device_id="dev_samsung_01",
        device_name="Samsung M31 (Updated)",
        device_type=DeviceType.PWA_MOBILE,
        app_version="1.10.0",
    )
    resp2 = svc.register_device(user_id="usr_farmer_01", farmer_id="fmr_01", req=req2)

    assert resp2.app_version == "1.10.0"
    assert resp2.device_name == "Samsung M31 (Updated)"

    devices = svc.list_user_devices(user_id="usr_farmer_01")
    assert len(devices) == 1


def test_device_revocation_and_rejection(sync_env):
    repo, svc = sync_env

    req = DeviceRegisterRequest(
        device_id="dev_lost_phone",
        device_name="Lost Field Phone",
        device_type=DeviceType.ANDROID_CHROME,
        app_version="1.10.0",
    )
    svc.register_device(user_id="usr_farmer_01", farmer_id="fmr_01", req=req)

    # Valid before revocation
    dev = svc.validate_active_device(device_id="dev_lost_phone", user_id="usr_farmer_01")
    assert dev["device_id"] == "dev_lost_phone"

    # Revoke
    ok = svc.revoke_device(device_id="dev_lost_phone", user_id="usr_farmer_01")
    assert ok is True

    # After revocation, validation MUST raise HTTP 403 Forbidden
    with pytest.raises(HTTPException) as exc:
        svc.validate_active_device(device_id="dev_lost_phone", user_id="usr_farmer_01")
    assert exc.value.status_code == 403
    assert "revoked" in exc.value.detail.lower()


def test_cross_tenant_device_isolation(sync_env):
    repo, svc = sync_env

    req = DeviceRegisterRequest(
        device_id="dev_user1",
        device_name="User 1 Phone",
        device_type=DeviceType.PWA_MOBILE,
        app_version="1.10.0",
    )
    svc.register_device(user_id="usr_farmer_01", farmer_id="fmr_01", req=req)

    # User 2 cannot access or validate User 1's device
    with pytest.raises(HTTPException) as exc:
        svc.validate_active_device(device_id="dev_user1", user_id="usr_farmer_02")
    assert exc.value.status_code == 403
    assert "different user account" in exc.value.detail.lower()

    # User 2 cannot revoke User 1's device
    with pytest.raises(HTTPException) as exc2:
        svc.revoke_device(device_id="dev_user1", user_id="usr_farmer_02")
    assert exc2.value.status_code == 404
