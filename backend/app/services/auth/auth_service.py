"""Authentication and security service with salted PBKDF2 password hashing and token lifecycle."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import get_logger
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


class AuthenticationError(Exception):
    """Raised when authentication credentials or token validation fails."""
    pass


class AuthService:
    """Provides secure farmer registration, authentication, session tokens, and identity resolution."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.repo = repo or get_farm_repo()
        self.secret_key = getattr(self.settings, "jwt_secret_key", "agri-guardian-secure-phase6-secret-key-3848201").encode("utf-8")
        self.token_expire_seconds = getattr(self.settings, "access_token_expire_minutes", 1440) * 60

    # =========================================================================
    # PASSWORD HASHING (PBKDF2-HMAC-SHA256, Salted, 100,000 Iterations)
    # =========================================================================

    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash a plaintext password with a random cryptographic salt."""
        if not salt:
            salt = secrets.token_hex(16)
        
        salt_bytes = bytes.fromhex(salt)
        pwd_hash = hashlib.pbkdf2_hmac(
            hash_name="sha256",
            password=password.encode("utf-8"),
            salt=salt_bytes,
            iterations=100000,
        ).hex()
        return pwd_hash, salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against salt and hash using constant-time comparison."""
        calc_hash, _ = self.hash_password(password, salt=salt)
        return secrets.compare_digest(calc_hash, password_hash)

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
        """Validate signature, check expiration, and decode token payload."""
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

        return payload_data

    # =========================================================================
    # REGISTRATION & LOGIN WORKFLOWS
    # =========================================================================

    def register_farmer(self, req: UserRegistrationRequest) -> Tuple[UserRecord, FarmerProfile, str]:
        """Register a new farmer account and provision their initial profile."""
        existing = self.repo.get_user_by_username(req.username.strip())
        if existing:
            raise ValueError(f"Username or mobile '{req.username}' is already registered.")

        # Hash password securely
        pwd_hash, salt = self.hash_password(req.password)
        now_str = datetime.now().isoformat()
        user_id = f"usr_{uuid.uuid4().hex[:12]}"

        user = UserRecord(
            user_id=user_id,
            username=req.username.strip(),
            password_hash=pwd_hash,
            salt=salt,
            role=req.role,
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

    def authenticate_user(self, req: UserLoginRequest) -> UserLoginResponse:
        """Verify login credentials and return bearer token with profile summary."""
        user = self.repo.get_user_by_username(req.username.strip())
        if not user:
            raise AuthenticationError("Invalid username or password.")

        if not self.verify_password(req.password, user.password_hash, user.salt):
            raise AuthenticationError("Invalid username or password.")

        profile = self.repo.get_farmer_profile_by_user_id(user.user_id)
        if not profile:
            raise AuthenticationError("User account exists but has no linked farmer profile.")

        # Update last login
        now_str = datetime.now().isoformat()
        self.repo.update_user_last_login(user.user_id, now_str)

        token = self.create_access_token(user, profile)
        logger.info(f"Authenticated farmer '{user.username}' ({profile.farmer_id})")

        return UserLoginResponse(
            access_token=token,
            token_type="bearer",
            user_id=user.user_id,
            farmer_id=profile.farmer_id,
            username=user.username,
            full_name=profile.full_name,
            role=user.role,
            preferred_language=profile.preferred_language,
            profile_completion_pct=profile.profile_completion_pct,
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

