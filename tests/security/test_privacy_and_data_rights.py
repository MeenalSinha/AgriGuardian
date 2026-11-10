"""Tests for Phase 12 Privacy, Consent Management & Data Subject Rights (DPDP Act)."""

import os
import tempfile
import pytest
from datetime import datetime, timezone

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    CropCycle,
    CropGrowthStage,
    CropSeason,
    Farm,
    FarmerProfile,
    Plot,
    UserRecord,
    UserRole,
)
from backend.app.schemas.privacy import (
    ConsentStatus,
    ConsentType,
    UserConsentRecord,
)


@pytest.fixture
def privacy_repo():
    """Create isolated test repository."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)
    yield repo
    if os.path.exists(tmp.name):
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def test_consent_recording_and_listing(privacy_repo):
    """Verify that user consents can be recorded and listed with purpose and policy version."""
    now_str = datetime.now(timezone.utc).isoformat()
    consent = UserConsentRecord(
        consent_id="cns_loc_001",
        user_id="usr_priv_01",
        farmer_id="fmr_priv_01",
        consent_type=ConsentType.LOCATION,
        status=ConsentStatus.GRANTED,
        policy_version="v1.2",
        purpose="Used to fetch localized Open-Meteo agricultural weather and mandi prices.",
        ip_address="192.168.1.1",
        created_at=now_str,
        updated_at=now_str,
    )

    saved = privacy_repo.record_consent(consent)
    assert saved.consent_id == "cns_loc_001"

    # Query consents
    consents = privacy_repo.get_user_consents("usr_priv_01")
    assert len(consents) == 1
    assert consents[0].consent_type == ConsentType.LOCATION
    assert consents[0].status == ConsentStatus.GRANTED
    assert consents[0].policy_version == "v1.2"


def test_consent_withdrawal(privacy_repo):
    """Verify that user can explicitly withdraw previously granted consent."""
    now_str = datetime.now(timezone.utc).isoformat()
    consent = UserConsentRecord(
        consent_id="cns_voice_001",
        user_id="usr_priv_02",
        farmer_id="fmr_priv_02",
        consent_type=ConsentType.VOICE_RECORDING,
        status=ConsentStatus.GRANTED,
        policy_version="v1.0",
        purpose="Voice processing via Gnani AI",
        created_at=now_str,
        updated_at=now_str,
    )
    privacy_repo.record_consent(consent)

    # Withdraw consent
    success = privacy_repo.withdraw_consent("usr_priv_02", ConsentType.VOICE_RECORDING.value)
    assert success is True

    # Check updated status
    consents = privacy_repo.get_user_consents("usr_priv_02")
    assert len(consents) == 1
    assert consents[0].status == ConsentStatus.WITHDRAWN


def test_farmer_full_data_export(privacy_repo):
    """Verify that data export bundle contains complete profile, farms, crops, and ledgers."""
    user = UserRecord(
        user_id="usr_exp_01",
        username="export_farmer",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    privacy_repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_exp_01",
        user_id="usr_exp_01",
        full_name="Export Farmer",
        phone_number="9876543210",
        preferred_language="hi-IN",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    privacy_repo.create_farmer_profile(profile)

    farm = Farm(
        farm_id="farm_exp_01",
        farmer_id="fmr_exp_01",
        farm_name="Export Farm",
        state="Haryana",
        district="Karnal",
        total_area=5.0,
        total_area_acres=5.0,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    privacy_repo.create_farm(farm)

    export_bundle = privacy_repo.get_farmer_full_export_data("fmr_exp_01")
    assert export_bundle["farmer_id"] == "fmr_exp_01"
    assert export_bundle["profile"]["full_name"] == "Export Farmer"
    assert len(export_bundle["farms"]) == 1
    assert export_bundle["farms"][0]["farm_name"] == "Export Farm"
    assert "expenses" in export_bundle
    assert "harvests" in export_bundle
    assert "sales" in export_bundle
    assert "consents" in export_bundle


def test_farmer_account_anonymized_deletion(privacy_repo):
    """Verify right-to-erasure account deletion anonymizes personal identity."""
    user = UserRecord(
        user_id="usr_del_01",
        username="farmer_to_delete",
        password_hash="secret_hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    privacy_repo.create_user(user)

    profile = FarmerProfile(
        farmer_id="fmr_del_01",
        user_id="usr_del_01",
        full_name="Farmer To Delete",
        phone_number="9988776655",
        preferred_language="hi-IN",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    privacy_repo.create_farmer_profile(profile)

    # Perform anonymized deletion
    deleted = privacy_repo.delete_farmer_account_anonymized("usr_del_01", "fmr_del_01")
    assert deleted is True

    # User record must be scrubbed
    scrubbed_user = privacy_repo.get_user_by_id("usr_del_01")
    assert "deleted_user" in scrubbed_user.username
    assert scrubbed_user.password_hash == "ACCOUNT_DELETED"

    # Profile must be scrubbed of phone and name
    scrubbed_profile = privacy_repo.get_farmer_profile("fmr_del_01")
    assert scrubbed_profile.full_name == "Deleted Farmer Account"
    assert scrubbed_profile.phone_number is None

