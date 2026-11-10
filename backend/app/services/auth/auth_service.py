"""Authentication and security service with Argon2id password hashing, PBKDF2 migration, and token lifecycle (Phase 12)."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
from backend.app.core.rate_limiter import get_login_rate_limiter
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import (
    FarmerProfile,
    LiteracyPreference,
    UserLoginRequest,
    UserLoginResponse,
    UserRecord,
    UserRegistrationRequest,
    UserRole,
)

logger = get_logger("AuthService")

# Initialize Argon2 PasswordHasher if available
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, VerificationError
    HAS_ARGON2 = True
    _argon2_hasher = PasswordHasher(
        time_cost=2,
        memory_cost=65536,  # 64 MB
        parallelism=1,
        hash_len=32,
        salt_len=16,
    )
    logger.info("Argon2id password hashing engine initialized successfully.")
except ImportError:
    HAS_ARGON2 = False
    _argon2_hasher = None
    logger.warning("argon2-cffi not installed; falling back to PBKDF2-HMAC-SHA256.")


class AuthenticationError(Exception):
    """Raised when authentication credentials or token validation fails."""
    pass


class AuthService:
    """Provides secure farmer registration, authentication, session tokens, and identity resolution."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.repo = repo or get_farm_repo()
        secret = getattr(self.settings, "jwt_secret_key", None) or os.getenv("JWT_SECRET_KEY")
        if not secret:
            secret = secrets.token_hex(32)
        self.secret_key = secret.encode("utf-8")
        self.token_expire_seconds = getattr(self.settings, "access_token_expire_minutes", 1440) * 60
        self.rate_limiter = get_login_rate_limiter()

    # =========================================================================
    # PASSWORD HASHING (Argon2id with PBKDF2 Backward Compatibility)
    # =========================================================================

    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash a plaintext password using Argon2id (default) or salted PBKDF2 fallback."""
        if not salt:
            salt = secrets.token_hex(16)

        if HAS_ARGON2 and _argon2_hasher is not None:
            # Argon2 embeds its cryptographic salt directly within the hash string
            argon2_hash = _argon2_hasher.hash(password)
            return argon2_hash, salt

        # Fallback to PBKDF2-HMAC-SHA256 (100,000 iterations)
        salt_bytes = bytes.fromhex(salt)
        pwd_hash = hashlib.pbkdf2_hmac(
            hash_name="sha256",
            password=password.encode("utf-8"),
            salt=salt_bytes,
            iterations=100000,
        ).hex()
        return pwd_hash, salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against stored hash using constant-time comparison.
        
        Supports both modern Argon2id hashes and legacy PBKDF2 hashes.
        """
        if not password or not password_hash:
            return False

        # Check if hash is Argon2id
        if password_hash.startswith("$argon2id$") or password_hash.startswith("$argon2"):
            if HAS_ARGON2 and _argon2_hasher is not None:
                try:
                    return _argon2_hasher.verify(password_hash, password)
                except (VerifyMismatchError, VerificationError):
                    return False
                except Exception as e:
                    logger.error(f"Argon2 verification unexpected error: {e}")
                    return False
            return False

        # Fallback: legacy PBKDF2-HMAC-SHA256 verification
        try:
            salt_bytes = bytes.fromhex(salt)
            calc_hash = hashlib.pbkdf2_hmac(
                hash_name="sha256",
                password=password.encode("utf-8"),
                salt=salt_bytes,
                iterations=100000,
            ).hex()
            return secrets.compare_digest(calc_hash, password_hash)
        except Exception:
            return False

    # =========================================================================
    # TOKEN LIFECYCLE (HMAC-SHA256 Cryptographic Bearer Token)
    # =========================================================================

    def create_access_token(self, user: UserRecord, farmer: Optional[FarmerProfile] = None) -> str:
        """Generate a cryptographically signed URL-safe bearer token."""
        payload = {
            "sub": user.user_id,
            "farmer_id": farmer.farmer_id if farmer else None,
            "username": user.username,
            "role": user.role.value,
            "lang": farmer.preferred_language if farmer else "en",
            "iat": int(time.time()),
            "exp": int(time.time()) + self.token_expire_seconds,
            "jti": uuid.uuid4().hex,
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
        
        # Sign payload with HMAC-SHA256
        signature = hmac.new(self.secret_key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{b64_payload}.{signature}"

    def decode_and_validate_token(self, token: str) -> Dict[str, Any]:
        """Validate signature, check expiration, token revocation, and decode token payload."""
        if not token or "." not in token:
            raise AuthenticationError("Invalid or malformed authentication token format.")

        parts = token.split(".")
        if len(parts) != 2:
            raise AuthenticationError("Invalid token structure.")

        b64_payload, signature = parts[0], parts[1]

        # Verify signature in constant time
        expected_sig = hmac.new(self.secret_key, b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not secrets.compare_digest(signature, expected_sig):
            raise AuthenticationError("Token cryptographic signature verification failed.")

        # Decode payload
        try:
            # Re-pad base64
            padded = b64_payload + "=" * ((4 - len(b64_payload) % 4) % 4)
            payload_data = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        except Exception as e:
            raise AuthenticationError(f"Failed to decode token payload: {e}")

        # Check expiration
        exp = payload_data.get("exp", 0)
        if time.time() > exp:
            raise AuthenticationError("Authentication token has expired. Please log in again.")

        # Check token blacklist / revocation status
        jti = payload_data.get("jti")
        if jti and self.repo.is_token_revoked(jti):
            raise AuthenticationError("Authentication token has been revoked. Please log in again.")

        return payload_data

    def revoke_token(self, token: str) -> None:
        """Revoke a token by blacklisting its JTI until expiration."""
        payload = self.decode_and_validate_token(token)
        jti = payload.get("jti")
        user_id = payload.get("sub")
        exp = payload.get("exp", int(time.time()) + self.token_expire_seconds)
        if jti and user_id:
            self.repo.revoke_token(jti=jti, user_id=user_id, expires_at=exp)
            logger.info(f"Revoked token JTI {jti} for user {user_id}")

    # =========================================================================
    # REGISTRATION & LOGIN WORKFLOWS
    # =========================================================================

    def register_farmer(self, req: UserRegistrationRequest) -> Tuple[UserRecord, FarmerProfile, str]:
        """Register a new farmer account and provision their initial profile.
        
        CRITICAL SECURITY RULE: Public self-registration strictly provisions ordinary FARMER accounts.
        Privileged roles (FIELD_OFFICER, AGRONOMIST, ADMINISTRATOR) must be provisioned through
        administrative workflows. Any client-supplied privileged role is safely rejected/ignored.
        """
        existing = self.repo.get_user_by_username(req.username.strip())
        if existing:
            raise ValueError(f"Username or mobile '{req.username}' is already registered.")

        # Hash password securely
        pwd_hash, salt = self.hash_password(req.password)
        now_str = datetime.now(timezone.utc).isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        # Always enforce FARMER role for public registration
        user = UserRecord(
            user_id=user_id,
            username=req.username.strip(),
            password_hash=pwd_hash,
            salt=salt,
            role=UserRole.FARMER,
            created_at=now_str,
            last_login=now_str,
        )
        self.repo.create_user(user)

        # Create linked Farmer Profile
        farmer_id = f"fmr_{uuid.uuid4().hex[:12]}"
        profile = FarmerProfile(
            farmer_id=farmer_id,
            user_id=user_id,
            full_name=req.full_name.strip(),
            phone_number=req.phone_number.strip() if req.phone_number else None,
            preferred_language=req.preferred_language,
            experience_years=None,
            literacy_preference=LiteracyPreference.VOICE_PRIMARY,
            kisan_credit_card_holder=False,
            soil_health_card_available=False,
            profile_completion_pct=25.0,  # Initial registration gives 25%
            created_at=now_str,
            updated_at=now_str,
        )
        self.repo.create_farmer_profile(profile)

        token = self.create_access_token(user, profile)
        logger.info(f"Registered new farmer '{req.username}' with ID {farmer_id}")
        return user, profile, token

    def provision_privileged_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        full_name: Optional[str] = None,
    ) -> Tuple[UserRecord, str]:
        """Provision a privileged user account (Field Officer, Agronomist/Expert, Administrator).
        
        Must only be invoked by authorized administrators.
        """
        existing = self.repo.get_user_by_username(username.strip())
        if existing:
            raise ValueError(f"Username '{username}' is already registered.")

        pwd_hash, salt = self.hash_password(password)
        now_str = datetime.now(timezone.utc).isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        user = UserRecord(
            user_id=user_id,
            username=username.strip(),
            password_hash=pwd_hash,
            salt=salt,
            role=role,
            created_at=now_str,
            last_login=now_str,
        )
        self.repo.create_user(user)
        token = self.create_access_token(user)
        logger.info(f"Administratively provisioned privileged user '{username}' with role {role.value}")
        return user, token

    def authenticate_user(self, req: UserLoginRequest, client_ip: Optional[str] = None) -> UserLoginResponse:
        """Verify login credentials with brute-force rate limiting and transparent hash upgrade."""
        username_clean = req.username.strip()

        # Check brute-force lockout on username or IP
        is_locked, remaining = self.rate_limiter.is_locked_out(username_clean)
        if is_locked:
            logger.warning(f"Login rejected: '{username_clean}' is temporarily locked out ({remaining}s remaining).")
            raise AuthenticationError(
                f"Too many failed login attempts. Please wait {remaining} seconds before trying again."
            )

        user = self.repo.get_user_by_username(username_clean)
        if not user:
            self.rate_limiter.record_failure(username_clean)
            if client_ip:
                self.rate_limiter.record_failure(client_ip)
            # Generic error message to prevent username enumeration
            raise AuthenticationError("Invalid username or password.")

        if not self.verify_password(req.password, user.password_hash, user.salt):
            self.rate_limiter.record_failure(username_clean)
            if client_ip:
                self.rate_limiter.record_failure(client_ip)
            # Generic error message to prevent username enumeration
            raise AuthenticationError("Invalid username or password.")

        # Login succeeded: clear rate limiter counter
        self.rate_limiter.record_success(username_clean)
        if client_ip:
            self.rate_limiter.record_success(client_ip)

        # Transparently upgrade legacy PBKDF2 hash to modern Argon2id
        if HAS_ARGON2 and not (user.password_hash.startswith("$argon2id$") or user.password_hash.startswith("$argon2")):
            try:
                new_hash, new_salt = self.hash_password(req.password)
                self.repo.update_user_password(user.user_id, new_hash, new_salt)
                logger.info(f"Transparently migrated password hash for user '{user.username}' to Argon2id.")
            except Exception as e:
                logger.warning(f"Failed to auto-upgrade password hash to Argon2id for '{user.username}': {e}")

        profile = self.repo.get_farmer_profile_by_user_id(user.user_id)
        if not profile and user.role == UserRole.FARMER:
            raise AuthenticationError("User account exists but has no linked farmer profile.")

        # Update last login
        now_str = datetime.now(timezone.utc).isoformat()
        self.repo.update_user_last_login(user.user_id, now_str)

        token = self.create_access_token(user, profile)
        logger.info(f"Authenticated user '{user.username}' (role={user.role.value})")

        return UserLoginResponse(
            access_token=token,
            token_type="bearer",
            user_id=user.user_id,
            farmer_id=profile.farmer_id if profile else "",
            username=user.username,
            full_name=profile.full_name if profile else user.username,
            role=user.role,
            preferred_language=profile.preferred_language if profile else "hi-IN",
            profile_completion_pct=profile.profile_completion_pct if profile else 100.0,
        )

    def get_current_user(self, token: str) -> UserRecord:
        """Resolve current authenticated user record from token without requiring farmer profile."""
        payload = self.decode_and_validate_token(token)
        user_id = payload.get("sub")
        user = self.repo.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User account not found.")
        return user

    def require_role(self, token: str, allowed_roles: list[UserRole]) -> UserRecord:
        """Enforce role-based access control (RBAC) on token."""
        user = self.get_current_user(token)
        if user.role not in allowed_roles:
            role_names = [r.value for r in allowed_roles]
            raise PermissionError(
                f"Forbidden: User '{user.username}' with role '{user.role.value}' lacks required permissions. "
                f"Requires one of: {role_names}"
            )
        return user

    def get_current_farmer(self, token: str) -> Tuple[UserRecord, FarmerProfile]:
        """Resolve current farmer identity from token."""
        payload = self.decode_and_validate_token(token)
        user_id = payload.get("sub")
        farmer_id = payload.get("farmer_id")

        user = self.repo.get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User account not found.")

        profile = self.repo.get_farmer_profile(farmer_id)
        if not profile:
            raise AuthenticationError("Farmer profile not found.")

        return user, profile
