"""Pydantic schemas and enums for Phase 10 Offline-First Sync & Connectivity Resilience."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    ANDROID_CHROME = "android_chrome"
    PWA_MOBILE = "pwa_mobile"
    DESKTOP_BROWSER = "desktop_browser"
    FIELD_TABLET = "field_tablet"


class SyncResourceType(str, Enum):
    FARM = "farm"
    FARM_PLOT = "farm_plot"
    CROP_CYCLE = "crop_cycle"
    FARM_ACTIVITY = "farm_activity"
    FARM_REMINDER = "farm_reminder"
    NOTIFICATION = "notification"
    ADVISORY_CASE = "advisory_case"
    FIELD_OBSERVATION = "field_observation"
    CROP_DIAGNOSIS_DRAFT = "crop_diagnosis_draft"


class MutationOperation(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    COMPLETE = "COMPLETE"


class MutationStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    CONFLICT = "conflict"
    REJECTED = "rejected"


class ConflictResolutionStrategy(str, Enum):
    CLIENT_WINS = "client_wins"
    SERVER_WINS = "server_wins"
    MANUAL_MERGE = "manual_merge"
    PENDING = "pending"


class ConflictStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class NetworkQuality(str, Enum):
    ONLINE = "online"
    WEAK = "weak"
    OFFLINE = "offline"


# =============================================================================
# DEVICE REGISTRATION SCHEMAS
# =============================================================================

class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., description="Client-generated persistent UUID for this installation")
    device_name: str = Field(..., description="Human-readable device name (e.g., 'Ramesh Redmi Note 10')")
    device_type: DeviceType = Field(default=DeviceType.PWA_MOBILE, description="Client device category")
    app_version: str = Field(default="1.10.0", description="Installed PWA or client version")


class DeviceResponse(BaseModel):
    device_id: str
    user_id: str
    farmer_id: Optional[str] = None
    device_name: str
    device_type: str
    app_version: str
    last_sync_at: Optional[str] = None
    is_revoked: bool = False
    created_at: str
    updated_at: str


class DeviceRevokeRequest(BaseModel):
    device_id: str


# =============================================================================
# BOOTSTRAP SYNCHRONIZATION SCHEMAS
# =============================================================================

class SyncBootstrapResponse(BaseModel):
    farmer_profile: Optional[Dict[str, Any]] = None
    farms: List[Dict[str, Any]] = Field(default_factory=list)
    plots: List[Dict[str, Any]] = Field(default_factory=list)
    crop_cycles: List[Dict[str, Any]] = Field(default_factory=list)
    activities: List[Dict[str, Any]] = Field(default_factory=list)
    reminders: List[Dict[str, Any]] = Field(default_factory=list)
    notifications: List[Dict[str, Any]] = Field(default_factory=list)
    advisories: List[Dict[str, Any]] = Field(default_factory=list)
    server_timestamp: str
    sequence_cursor: int = 0


# =============================================================================
# INCREMENTAL PULL SCHEMAS
# =============================================================================

class SyncPullRequest(BaseModel):
    device_id: str = Field(..., description="Registered device requesting updates")
    since_sequence_id: int = Field(default=0, ge=0, description="Last acknowledged sequence_id")
    limit: int = Field(default=100, ge=1, le=500, description="Max changes to return per page")


class ChangeFeedItem(BaseModel):
    sequence_id: int
    resource_type: str
    resource_id: str
    farmer_id: str
    operation: str
    version_number: int = 1
    payload: Optional[Dict[str, Any]] = None
    created_at: str


class SyncPullResponse(BaseModel):
    changes: List[ChangeFeedItem] = Field(default_factory=list)
    new_sequence_id: int
    has_more: bool = False
    server_timestamp: str


# =============================================================================
# INCREMENTAL PUSH (MUTATIONS) SCHEMAS
# =============================================================================

class SyncMutationItem(BaseModel):
    mutation_id: str = Field(..., description="Unique client-generated mutation UUID")
    resource_type: str = Field(..., description="Target entity type (e.g. farm_activity, farm_reminder)")
    operation: str = Field(..., description="Action: CREATE, UPDATE, DELETE, COMPLETE")
    client_record_id: str = Field(..., description="Client temporary or local ID")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    base_version: int = Field(default=0, ge=0, description="Server version this edit was branched from")


class SyncPushRequest(BaseModel):
    device_id: str = Field(..., description="Registered device submitting mutations")
    mutations: List[SyncMutationItem] = Field(..., max_length=100, description="Batch of pending mutations")


class MutationResultItem(BaseModel):
    mutation_id: str
    client_record_id: str
    server_record_id: str
    status: str
    server_version: int = 1
    conflict_details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SyncPushResponse(BaseModel):
    processed_count: int
    applied_count: int
    conflict_count: int
    results: List[MutationResultItem] = Field(default_factory=list)
    server_timestamp: str


# =============================================================================
# CONFLICT SCHEMAS
# =============================================================================

class ConflictItem(BaseModel):
    conflict_id: str
    mutation_id: str
    device_id: str
    farmer_id: str
    resource_type: str
    client_record_id: str
    server_record_id: Optional[str] = None
    client_payload: Dict[str, Any]
    server_payload: Optional[Dict[str, Any]] = None
    resolution_strategy: str = "pending"
    status: str = "pending"
    created_at: str
    resolved_at: Optional[str] = None


class ConflictResolveRequest(BaseModel):
    resolution_strategy: ConflictResolutionStrategy = Field(
        ..., description="Resolution: client_wins, server_wins, or manual_merge"
    )
    merged_payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom merged fields if manual_merge is chosen"
    )


class ConflictResolveResponse(BaseModel):
    conflict_id: str
    status: str
    resolution_strategy: str
    resolved_at: str


# =============================================================================
# ATTACHMENT SCHEMAS
# =============================================================================

class AttachmentUploadResponse(BaseModel):
    attachment_id: str
    file_name: str
    mime_type: str
    byte_size: int
    sha256_checksum: str
    storage_path: str
    created_at: str


class AttachmentMetadataResponse(BaseModel):
    attachment_id: str
    farmer_id: str
    device_id: str
    resource_type: str
    resource_id: Optional[str] = None
    sha256_checksum: str
    file_name: str
    mime_type: str
    byte_size: int
    storage_path: str
    is_uploaded: bool = True
    created_at: str


# =============================================================================
# TELEMETRY SCHEMAS
# =============================================================================

class SyncTelemetryReport(BaseModel):
    device_id: str
    sync_type: str
    duration_ms: int
    pull_count: int = 0
    push_count: int = 0
    conflict_count: int = 0
    bytes_transferred: int = 0
    status: str = "success"
