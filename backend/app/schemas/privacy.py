"""Pydantic models and schemas for Privacy, Consent Management & Data Subject Rights (Phase 12)."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConsentType(str, Enum):
    LOCATION = "location"
    VOICE_RECORDING = "voice_recording"
    IMAGE_DIAGNOSTICS = "image_diagnostics"
    SMS_NOTIFICATIONS = "sms_notifications"
    WHATSAPP_NOTIFICATIONS = "whatsapp_notifications"
    ANALYTICS = "analytics"


class ConsentStatus(str, Enum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"


class UserConsentRecord(BaseModel):
    consent_id: str
    user_id: str
    farmer_id: Optional[str] = None
    consent_type: ConsentType
    status: ConsentStatus
    policy_version: str = "v1.0"
    purpose: str
    ip_address: Optional[str] = None
    created_at: str
    updated_at: str


class ConsentSubmitRequest(BaseModel):
    consent_type: ConsentType
    status: ConsentStatus = ConsentStatus.GRANTED
    policy_version: str = Field(default="v1.0", description="Version of privacy policy / terms consented to")
    purpose: str = Field(..., min_length=5, description="Clear purpose description for data processing")


class ConsentWithdrawRequest(BaseModel):
    consent_type: ConsentType


class FarmerDataExportResponse(BaseModel):
    """Machine-readable comprehensive archive for data portability under DPDP Act / GDPR."""
    export_timestamp: str
    user_id: str
    farmer_id: str
    profile: Dict[str, Any]
    farms: List[Dict[str, Any]]
    plots: List[Dict[str, Any]]
    crop_cycles: List[Dict[str, Any]]
    expenses: List[Dict[str, Any]]
    harvests: List[Dict[str, Any]]
    sales: List[Dict[str, Any]]
    budgets: List[Dict[str, Any]]
    consents: List[Dict[str, Any]]
    active_devices: List[Dict[str, Any]]


class AccountDeletionRequest(BaseModel):
    confirm_password: str = Field(..., description="Password confirmation to verify identity before deletion")
    reason: Optional[str] = Field(default=None, description="Optional feedback on reason for deletion")


class AccountDeletionResponse(BaseModel):
    status: str
    message_en: str
    message_hi: str
    deleted_at: str
    retention_notice: str

