"""Pydantic schemas and enums for Phase 9 Proactive Notifications, Outbox Orchestration, and Smart Farm Reminders."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NotificationCategory(str, Enum):
    """Functional domains for proactive notifications."""
    WEATHER_ALERT = "weather_alert"
    MARKET_INTELLIGENCE = "market_intelligence"
    EXPERT_REVIEW = "expert_review"
    CROP_REMINDER = "crop_reminder"
    SYSTEM_GOVERNANCE = "system_governance"


class NotificationPriority(str, Enum):
    """Urgency hierarchy for notification dispatch and quiet hour bypass."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"  # Bypasses quiet hours immediately


class NotificationChannel(str, Enum):
    """Available delivery channels for notifications."""
    IN_APP = "in_app"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    VOICE_CALL = "voice_call"  # Automated voice call powered by Gnani Timbre


class NotificationStatus(str, Enum):
    """Lifecycle status of an in-app farmer notification."""
    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"
    ARCHIVED = "archived"


class OutboxStatus(str, Enum):
    """Transactional state of an outbox message delivery attempt."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class ReminderStatus(str, Enum):
    """Status of a scheduled agronomic farm reminder."""
    SCHEDULED = "scheduled"
    DUE = "due"
    COMPLETED = "completed"
    SNOOZED = "snoozed"
    DISMISSED = "dismissed"


class ReminderActivityType(str, Enum):
    """Agronomic activities tracked in smart farm reminders."""
    IRRIGATION = "irrigation"
    FERTILIZATION = "fertilization"
    PEST_SPRAY = "pest_spray"
    SCOUTING = "scouting"
    WEEDING = "weeding"
    HARVESTING = "harvesting"
    SOWING = "sowing"
    CUSTOM = "custom"


class NotificationEvent(BaseModel):
    """Standardized event emitted by weather, market, expert, or crop subsystems."""
    event_id: str
    event_type: str
    category: NotificationCategory
    priority: NotificationPriority
    farmer_id: str
    farm_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    related_entity_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    dedup_key: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class NotificationRecord(BaseModel):
    """Persisted farmer notification record."""
    notification_id: str
    farmer_id: str
    category: NotificationCategory
    priority: NotificationPriority
    title_en: str
    title_hi: str
    body_en: str
    body_hi: str
    action_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: NotificationStatus = NotificationStatus.UNREAD
    read_at: Optional[str] = None
    dismissed_at: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None


class NotificationPreference(BaseModel):
    """Per-farmer notification configuration and quiet hours settings."""
    farmer_id: str
    channels_enabled: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP, NotificationChannel.SMS]
    )
    categories_enabled: List[NotificationCategory] = Field(
        default_factory=lambda: [
            NotificationCategory.WEATHER_ALERT,
            NotificationCategory.MARKET_INTELLIGENCE,
            NotificationCategory.EXPERT_REVIEW,
            NotificationCategory.CROP_REMINDER,
        ]
    )
    quiet_hours_enabled: bool = True
    quiet_hours_start: str = "21:00"  # 9:00 PM
    quiet_hours_end: str = "06:00"    # 6:00 AM
    emergency_override: bool = True   # EMERGENCY priority overrides quiet hours
    preferred_language: str = "hi-IN"
    updated_at: Optional[str] = None


class NotificationChannelConsent(BaseModel):
    """Audit log of explicit farmer consent for outbound delivery channels."""
    consent_id: str
    farmer_id: str
    channel: NotificationChannel
    is_granted: bool
    granted_at: str
    revoked_at: Optional[str] = None
    ip_or_source: str = "web_portal"


class ConsentUpdateRequest(BaseModel):
    """Payload to grant or revoke channel consent."""
    channel: NotificationChannel
    is_granted: bool
    source: str = "web_portal"


class NotificationOutboxItem(BaseModel):
    """Queue record for reliable transactional message dispatch."""
    outbox_id: str
    notification_id: str
    farmer_id: str
    channel: NotificationChannel
    status: OutboxStatus = OutboxStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[str] = None
    lease_expires_at: Optional[str] = None
    last_error: Optional[str] = None
    created_at: str
    updated_at: str


class NotificationDeliveryAttempt(BaseModel):
    """Audit record of a physical or simulated provider delivery attempt."""
    attempt_id: str
    outbox_id: str
    channel: NotificationChannel
    provider_name: str
    status: str  # success, failure, not_configured
    latency_ms: float
    error_message: Optional[str] = None
    provider_response_code: Optional[str] = None
    attempted_at: str


class NotificationTemplate(BaseModel):
    """Bilingual notification template with parameter interpolation."""
    template_id: str
    category: NotificationCategory
    event_type: str
    title_template_en: str
    title_template_hi: str
    body_template_en: str
    body_template_hi: str
    action_url_template: Optional[str] = None
    default_priority: NotificationPriority = NotificationPriority.MEDIUM


class FarmReminderRecord(BaseModel):
    """Smart farm activity reminder with weather risk awareness."""
    reminder_id: str
    farmer_id: str
    farm_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    activity_type: ReminderActivityType
    title_en: str
    title_hi: str
    description_en: Optional[str] = None
    description_hi: Optional[str] = None
    scheduled_date: str  # YYYY-MM-DD
    due_time: Optional[str] = None
    status: ReminderStatus = ReminderStatus.SCHEDULED
    snooze_until: Optional[str] = None
    completed_at: Optional[str] = None
    weather_risk_flag: bool = False
    weather_risk_reason_en: Optional[str] = None
    weather_risk_reason_hi: Optional[str] = None
    created_at: str
    updated_at: str


class FarmReminderCreate(BaseModel):
    """Payload to create a new farm reminder."""
    farm_id: Optional[str] = None
    crop_cycle_id: Optional[str] = None
    activity_type: ReminderActivityType
    title_en: str
    title_hi: Optional[str] = None
    description_en: Optional[str] = None
    description_hi: Optional[str] = None
    scheduled_date: str
    due_time: Optional[str] = "08:00"


class FarmReminderSnooze(BaseModel):
    """Payload to snooze a farm reminder."""
    snooze_days: int = Field(default=1, ge=1, le=14)
    custom_date: Optional[str] = None


class OutboxProcessResult(BaseModel):
    """Summary of outbox batch execution cycle."""
    processed_count: int = 0
    delivered_count: int = 0
    failed_count: int = 0
    dead_letter_count: int = 0
    duration_ms: float = 0.0


class ChannelHealthInfo(BaseModel):
    """Operational health telemetry for an alert delivery channel."""
    channel: NotificationChannel
    configured: bool
    provider_name: str
    details: str
    total_dispatched: int = 0
    total_failed: int = 0
