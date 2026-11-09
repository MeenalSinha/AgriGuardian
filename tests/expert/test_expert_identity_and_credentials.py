"""Unit tests for Agricultural Expert identity, profiles, and institutional credentials."""

import os
import tempfile
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.expert import (
    ExpertAccountStatus,
    ExpertCredentialRecord,
    ExpertProfile,
    ExpertSpecializationDomain,
    ExpertSpecializationItem,
)
from backend.app.schemas.farm import UserRecord, UserRole


@pytest.fixture
def expert_repo():
    """Create isolated SQLite database for expert testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = FarmDatabaseRepository(db_path=path)
    yield repo
    try:
        os.remove(path)
    except Exception:
        pass


def test_expert_profile_lifecycle_and_credentials(expert_repo):
    """Verify creating, retrieving, updating expert profiles and attaching credentials."""
    # 1. Create linked agronomist user
    user = UserRecord(
        user_id="usr_agro_01",
        username="dr_sharma_path",
        password_hash="dummy_hash",
        salt="dummy_salt",
        role=UserRole.AGRONOMIST,
        created_at="2026-11-09T10:00:00",
    )
    expert_repo.create_user(user)

    # 2. Create expert profile
    specs = ExpertSpecializationItem(
        crop_types=["Wheat", "Tomato", "Rice"],
        domains=[ExpertSpecializationDomain.PLANT_PATHOLOGY],
        states=["Haryana", "Punjab"],
        languages=["hi-IN", "en-IN"],
    )
    profile = ExpertProfile(
        expert_id="exp_01",
        user_id=user.user_id,
        full_name="Dr. Rajesh Sharma",
        email="rsharma@hau.ac.in",
        phone_number="+91 98123 45678",
        institution="CCS Haryana Agricultural University (HAU)",
        designation="Senior Plant Pathologist",
        experience_years=15,
        specializations=specs,
        verification_status=ExpertAccountStatus.PENDING_VERIFICATION,
    )
    created = expert_repo.create_expert_profile(profile)
    assert created.expert_id == "exp_01"
    assert created.full_name == "Dr. Rajesh Sharma"
    assert created.verification_status == ExpertAccountStatus.PENDING_VERIFICATION

    # 3. Attach ICAR credential
    cred = ExpertCredentialRecord(
        credential_id="crd_01",
        expert_id="exp_01",
        institution_name="ICAR-IARI",
        degree_or_certification="Ph.D. Plant Pathology",
        registration_number="ICAR-PATH-2015-489",
    )
    expert_repo.add_expert_credential(cred)

    # Retrieve and verify credential attachment
    fetched = expert_repo.get_expert_profile("exp_01")
    assert fetched is not None
    assert len(fetched.credentials) == 1
    assert fetched.credentials[0].registration_number == "ICAR-PATH-2015-489"
    assert fetched.credentials[0].is_verified is False

    # 4. Admin verifies the expert
    admin_user = UserRecord(
        user_id="usr_admin_01",
        username="admin_gov",
        password_hash="dummy_hash",
        salt="dummy_salt",
        role=UserRole.ADMINISTRATOR,
        created_at="2026-11-09T10:00:00",
    )
    expert_repo.create_user(admin_user)

    approved = expert_repo.verify_expert("exp_01", admin_user.user_id, ExpertAccountStatus.ACTIVE_VERIFIED)
    assert approved is True

    # Check updated status
    verified_profile = expert_repo.get_expert_profile("exp_01")
    assert verified_profile.verification_status == ExpertAccountStatus.ACTIVE_VERIFIED
    assert verified_profile.verified_by_admin_id == admin_user.user_id
    assert verified_profile.credentials[0].is_verified is True

    # 5. Retrieve by user_id
    by_user = expert_repo.get_expert_profile_by_user_id("usr_agro_01")
    assert by_user is not None
    assert by_user.expert_id == "exp_01"

