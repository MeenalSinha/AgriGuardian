"""Unit tests for farmer authentication, password hashing, and token lifecycle."""

import os
import tempfile
import time
import pytest

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.farm import (
    FarmerProfile,
    LiteracyPreference,
    UserLoginRequest,
    UserRegistrationRequest,
    UserRole,
)
from backend.app.services.auth.auth_service import AuthService, AuthenticationError


@pytest.fixture
def auth_fixture():
    """Create isolated test repository and AuthService."""
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    repo = FarmDatabaseRepository(db_path=temp_db.name)
    auth_srv = AuthService(repo=repo)
    yield auth_srv, repo
    if os.path.exists(temp_db.name):
        try:
            os.remove(temp_db.name)
        except Exception:
            pass


def test_password_hashing(auth_fixture):
    """Verify PBKDF2 salted password hashing generates distinct hashes and validates correctly."""
    auth_srv, _ = auth_fixture

    pwd = "KisanPassword@2026"
    hash1, salt1 = auth_srv.hash_password(pwd)
    hash2, salt2 = auth_srv.hash_password(pwd)

    # Hashes must differ because salts are random
    assert hash1 != hash2
    assert salt1 != salt2

    # Verification must succeed for valid password and fail for invalid password
    assert auth_srv.verify_password(pwd, hash1, salt1) is True
    assert auth_srv.verify_password("WrongPassword", hash1, salt1) is False


def test_farmer_registration_and_login(auth_fixture):
    """Verify registration creates user and linked farmer profile, and login returns valid token."""
    auth_srv, repo = auth_fixture

    req = UserRegistrationRequest(
        username="ramesh_farmer",
        password="SecureKisanPwd123",
        full_name="Ramesh Patel",
        phone_number="9876543210",
        preferred_language="hi-IN",
        role=UserRole.FARMER,
    )

    user, profile, token = auth_srv.register_farmer(req)

    assert user.user_id.startswith("usr_")
    assert user.username == "ramesh_farmer"
    assert profile.farmer_id.startswith("fmr_")
    assert profile.full_name == "Ramesh Patel"
    assert profile.preferred_language == "hi-IN"
    assert profile.profile_completion_pct > 0
    assert token is not None and "." in token

    # Test login with valid credentials
    login_resp = auth_srv.authenticate_user(UserLoginRequest(
        username="ramesh_farmer",
        password="SecureKisanPwd123",
    ))
    assert login_resp.username == "ramesh_farmer"
    assert login_resp.farmer_id == profile.farmer_id
    assert login_resp.full_name == "Ramesh Patel"
    assert login_resp.access_token is not None


def test_duplicate_registration_rejection(auth_fixture):
    """Verify duplicate username or mobile number is rejected with ValueError."""
    auth_srv, _ = auth_fixture

    req = UserRegistrationRequest(
        username="duplicate_farmer",
        password="Password123",
        full_name="First Farmer",
    )
    auth_srv.register_farmer(req)

    with pytest.raises(ValueError, match="already registered"):
        auth_srv.register_farmer(req)


def test_invalid_login_rejection(auth_fixture):
    """Verify invalid password or non-existent username raises AuthenticationError."""
    auth_srv, _ = auth_fixture

    req = UserRegistrationRequest(
        username="auth_test_farmer",
        password="CorrectPassword123",
        full_name="Auth Tester",
    )
    auth_srv.register_farmer(req)

    with pytest.raises(AuthenticationError, match="Invalid username or password"):
        auth_srv.authenticate_user(UserLoginRequest(username="auth_test_farmer", password="WrongPassword"))

    with pytest.raises(AuthenticationError, match="Invalid username or password"):
        auth_srv.authenticate_user(UserLoginRequest(username="ghost_user", password="Password123"))


def test_token_validation_and_expiration(auth_fixture):
    """Verify cryptographic token verification and expiration rejection."""
    auth_srv, _ = auth_fixture

    req = UserRegistrationRequest(
        username="token_farmer",
        password="Password123",
        full_name="Token Farmer",
    )
    user, profile, token = auth_srv.register_farmer(req)

    # Valid token decoding
    payload = auth_srv.decode_and_validate_token(token)
    assert payload["sub"] == user.user_id
    assert payload["farmer_id"] == profile.farmer_id
    assert payload["username"] == "token_farmer"

    # Corrupted signature rejection
    corrupted_token = token[:-5] + "aaaaa"
    with pytest.raises(AuthenticationError, match="signature verification failed"):
        auth_srv.decode_and_validate_token(corrupted_token)

    # Expired token rejection
    auth_srv.token_expire_seconds = -10  # Immediately expired
    expired_token = auth_srv.create_access_token(user, profile)
    with pytest.raises(AuthenticationError, match="expired"):
        auth_srv.decode_and_validate_token(expired_token)
