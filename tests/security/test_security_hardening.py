"""Comprehensive security, authentication, password hashing, and HTTP hardening test suite (Phase 12)."""

import os
import tempfile
import pytest

from backend.app.core.config import Settings
from backend.app.core.middleware import (
    CorrelationIdMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from backend.app.core.production_validator import ProductionConfigurationError, validate_production_configuration
from backend.app.core.rate_limiter import LoginRateLimiter
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
def isolated_auth():
    """Create isolated test repository and AuthService."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)
    auth_srv = AuthService(repo=repo)
    yield auth_srv, repo
    if os.path.exists(tmp.name):
        try:
            os.remove(tmp.name)
        except Exception:
            pass


def test_argon2id_password_hashing_and_verification(isolated_auth):
    """Verify that passwords hashed with Argon2id produce valid verifiable hashes."""
    auth_srv, _ = isolated_auth
    pwd = "FarmerPassword2026"
    h, salt = auth_srv.hash_password(pwd)

    # Hash must be Argon2id formatted
    assert h.startswith("$argon2id$") or h.startswith("$argon2")
    assert auth_srv.verify_password(pwd, h, salt) is True
    assert auth_srv.verify_password("IncorrectPassword", h, salt) is False


def test_legacy_pbkdf2_migration_to_argon2id(isolated_auth):
    """Verify that legacy PBKDF2 hashes are successfully authenticated and auto-upgraded to Argon2id."""
    auth_srv, repo = isolated_auth

    # 1. Manually insert user with legacy PBKDF2 hash
    import hashlib
    import secrets
    legacy_salt = secrets.token_hex(16)
    legacy_hash = hashlib.pbkdf2_hmac(
        "sha256",
        "LegacyKisanPwd123".encode("utf-8"),
        bytes.fromhex(legacy_salt),
        100000,
    ).hex()

    user_id = "usr_legacy_01"
    from backend.app.schemas.farm import UserRecord
    legacy_user = UserRecord(
        user_id=user_id,
        username="legacy_kisan",
        password_hash=legacy_hash,
        salt=legacy_salt,
        role=UserRole.FARMER,
        created_at="2026-01-01T00:00:00",
        last_login="2026-01-01T00:00:00",
    )
    repo.create_user(legacy_user)

    profile = FarmerProfile(
        farmer_id="fmr_legacy_01",
        user_id=user_id,
        full_name="Legacy Farmer",
        preferred_language="hi-IN",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    repo.create_farmer_profile(profile)

    # 2. Authenticate user: legacy hash should verify successfully
    login_resp = auth_srv.authenticate_user(UserLoginRequest(
        username="legacy_kisan",
        password="LegacyKisanPwd123",
    ))
    assert login_resp.username == "legacy_kisan"

    # 3. Check that user's password in database has been auto-upgraded to Argon2id
    updated_user = repo.get_user_by_id(user_id)
    assert updated_user.password_hash.startswith("$argon2id$") or updated_user.password_hash.startswith("$argon2")


def test_password_complexity_enforcement():
    """Verify that short passwords or passwords lacking letters/numbers are rejected."""
    # Too short (< 8 chars)
    with pytest.raises(Exception):
        UserRegistrationRequest(
            username="test_kisan",
            password="123",
            full_name="Test Kisan",
        )

    # Only numbers
    with pytest.raises(Exception):
        UserRegistrationRequest(
            username="test_kisan",
            password="1234567890",
            full_name="Test Kisan",
        )

    # Only letters
    with pytest.raises(Exception):
        UserRegistrationRequest(
            username="test_kisan",
            password="abcdefghij",
            full_name="Test Kisan",
        )


def test_registration_role_escalation_prevention(isolated_auth):
    """Verify that public registration rejects or ignores privileged role escalation attempts."""
    # Attempt to supply ADMINISTRATOR role in registration schema
    with pytest.raises(ValueError, match="Unauthorized role assignment|Self-registration"):
        UserRegistrationRequest(
            username="malicious_user",
            password="SecurePass2026",
            full_name="Hacker User",
            role=UserRole.ADMINISTRATOR,
        )

    with pytest.raises(ValueError, match="Unauthorized role assignment|Self-registration"):
        UserRegistrationRequest(
            username="malicious_user2",
            password="SecurePass2026",
            full_name="Hacker User 2",
            role=UserRole.FIELD_OFFICER,
        )


def test_brute_force_login_rate_limiter():
    """Verify that multiple failed login attempts trigger lockout."""
    limiter = LoginRateLimiter(max_attempts=3, window_seconds=60, lockout_seconds=300)
    user_key = "attacked_user"

    # 1. Initial state: not locked out
    is_locked, _ = limiter.is_locked_out(user_key)
    assert is_locked is False

    # 2. Record 2 failures: still not locked out
    limiter.record_failure(user_key)
    limiter.record_failure(user_key)
    is_locked, _ = limiter.is_locked_out(user_key)
    assert is_locked is False

    # 3. Record 3rd failure: reaches threshold and locks out
    limiter.record_failure(user_key)
    is_locked, remaining = limiter.is_locked_out(user_key)
    assert is_locked is True
    assert remaining > 0

    # 4. Successful login clears lockout
    limiter.record_success(user_key)
    is_locked, _ = limiter.is_locked_out(user_key)
    assert is_locked is False


def test_token_revocation_and_blacklisting(isolated_auth):
    """Verify that revoked tokens cannot be used to authenticate."""
    auth_srv, repo = isolated_auth

    req = UserRegistrationRequest(
        username="logout_farmer",
        password="LogoutPass123",
        full_name="Logout Farmer",
    )
    user, profile, token = auth_srv.register_farmer(req)

    # Token must decode and validate initially
    payload = auth_srv.decode_and_validate_token(token)
    assert payload["sub"] == user.user_id

    # Revoke token
    auth_srv.revoke_token(token)

    # Token validation must now fail
    with pytest.raises(AuthenticationError, match="revoked"):
        auth_srv.decode_and_validate_token(token)


def _run_asgi_sync(coro):
    """Synchronously advance an ASGI coroutine without requiring an OS event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration:
        pass


def test_http_security_headers_and_correlation_id_middleware():
    """Verify that SecurityHeadersMiddleware and CorrelationIdMiddleware enforce defensive headers and request IDs."""
    async def dummy_app(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"status": "ok"}',
        })

    app_with_headers = SecurityHeadersMiddleware(dummy_app)
    app_with_both = CorrelationIdMiddleware(app_with_headers)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/health/live",
        "headers": [(b"x-custom-header", b"test")],
    }
    sent_messages = []
    async def dummy_send(msg):
        sent_messages.append(msg)

    _run_asgi_sync(app_with_both(scope, None, dummy_send))

    start_msg = [m for m in sent_messages if m["type"] == "http.response.start"][0]
    headers_dict = {k.decode("latin1"): v.decode("latin1") for k, v in start_msg["headers"]}

    assert headers_dict.get("x-content-type-options") == "nosniff"
    assert headers_dict.get("x-frame-options") == "DENY"
    assert headers_dict.get("x-xss-protection") == "1; mode=block"
    assert "default-src 'self'" in headers_dict.get("content-security-policy", "")
    assert headers_dict.get("x-request-id") is not None
    assert headers_dict.get("x-request-id").startswith("req_")


def test_request_size_limit_middleware():
    """Verify that RequestSizeLimitMiddleware rejects oversized payloads with 413 Payload Too Large."""
    async def dummy_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    limited_app = RequestSizeLimitMiddleware(dummy_app, max_bytes=1000)

    # 1. Payload within limit
    sent_ok = []
    scope_ok = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [(b"content-length", b"500")],
    }
    async def send_ok(m):
        sent_ok.append(m)

    _run_asgi_sync(limited_app(scope_ok, None, send_ok))
    assert sent_ok[0]["status"] == 200

    # 2. Payload exceeding limit
    sent_fail = []
    scope_fail = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [(b"content-length", b"2000")],
    }
    async def send_fail(m):
        sent_fail.append(m)

    _run_asgi_sync(limited_app(scope_fail, None, send_fail))
    assert sent_fail[0]["status"] == 413


def test_production_validator_enforcement():
    """Verify that production validator rejects weak secrets and wildcard CORS in production."""
    # Weak secret in production mode must raise ProductionConfigurationError
    prod_bad_settings = Settings(
        app_env="production",
        jwt_secret_key="short",
        allowed_origins=["http://localhost:8501"],
    )
    with pytest.raises(ProductionConfigurationError, match="too short"):
        validate_production_configuration(prod_bad_settings)

    # Wildcard origin in production mode must raise
    prod_wildcard_settings = Settings(
        app_env="production",
        jwt_secret_key="a_very_strong_production_key_with_sufficient_entropy_64chars",
        allowed_origins=["*"],
    )
    with pytest.raises(ProductionConfigurationError, match="Wildcard CORS"):
        validate_production_configuration(prod_wildcard_settings)

    # Valid production settings must pass
    prod_good_settings = Settings(
        app_env="production",
        jwt_secret_key="a_very_strong_production_key_with_sufficient_entropy_64chars",
        allowed_origins=["https://agriguardian.gov.in"],
        debug=False,
    )
    validate_production_configuration(prod_good_settings)

