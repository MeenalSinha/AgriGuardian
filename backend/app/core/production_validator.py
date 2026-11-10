"""Production Configuration Validator for AgriGuardian (Phase 12).

Enforces strict zero-insecurity startup invariants in production environments.
Halts startup immediately if insecure defaults, weak secrets, or wildcard CORS are detected.
"""

from typing import List
from backend.app.core.config import Settings
from backend.app.core.logging import get_logger

logger = get_logger("ProductionValidator")

INSECURE_SECRET_PATTERNS = [
    "your_jwt_secret_key_here",
    "your_production_high_entropy",
    "secret",
    "changeme",
    "admin",
    "password",
    "12345678",
    "test_secret",
    "default_secret",
]


class ProductionConfigurationError(RuntimeError):
    """Raised when critical configuration fails production readiness checks."""
    pass


def validate_production_configuration(settings: Settings) -> None:
    """Validate system configuration against strict production invariants.
    
    In development or test environments, logs security warnings.
    In staging or production environments, immediately halts startup if invariants fail.
    """
    env = (settings.app_env or "development").strip().lower()
    is_production = env in ("production", "prod", "staging")
    violations: List[str] = []

    # 1. JWT Secret Key Rigor
    secret = (settings.jwt_secret_key or "").strip()
    if not secret:
        violations.append("JWT_SECRET_KEY is empty or missing.")
    elif len(secret) < 32:
        violations.append(f"JWT_SECRET_KEY is too short ({len(secret)} characters); production requires at least 32 high-entropy characters.")
    elif any(pat in secret.lower() for pat in INSECURE_SECRET_PATTERNS):
        violations.append("JWT_SECRET_KEY appears to contain an insecure template placeholder or dictionary word.")

    # 2. CORS Allowlist
    origins = settings.allowed_origins or []
    if "*" in origins:
        violations.append("Wildcard CORS origin ('*') is strictly prohibited in production when credentials are enabled.")

    # 3. Debug Flag
    if settings.debug:
        violations.append("DEBUG mode is enabled; production environments must set DEBUG=false.")

    # 4. Enforce Decision
    if violations:
        error_summary = "\n  - " + "\n  - ".join(violations)
        if is_production:
            logger.critical(f"STARTUP HALTED DUE TO PRODUCTION CONFIGURATION VIOLATIONS:{error_summary}")
            raise ProductionConfigurationError(
                f"AgriGuardian refused to start in '{env}' mode due to insecure configuration:{error_summary}"
            )
        else:
            logger.warning(
                f"Running in '{env}' mode with configuration warnings (safe for development only):{error_summary}"
            )
    else:
        logger.info(f"Production configuration validation passed successfully for environment '{env}'.")

