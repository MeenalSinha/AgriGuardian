# AgriGuardian — Phase 9 REST API Documentation
## Proactive Notifications, Alert Orchestration & Smart Farm Reminders

All notification endpoints require authentication via Bearer token (`Authorization: Bearer <access_token>`). Role-based access control (RBAC) and object-level multi-tenant isolation are strictly enforced across all operations.

---

## 1. Notification Center Endpoints

### 1.1 List Farmer Notifications
- **Method:** `GET`
- **Path:** `/farmer/notifications`
- **Description:** Retrieve paginated notifications for the authenticated farmer with optional filtering.
- **Allowed Roles:** `farmer`
- **Query Parameters:**
  - `category` (optional, string): Filter by category (`weather_alert`, `market_intelligence`, `expert_review`, `crop_reminder`, `advisory_update`, `system`)
  - `unread_only` (optional, boolean, default `false`): Return only unread notifications
  - `limit` (optional, integer, default `50`)
- **Response `200 OK`:**
  ```json
  [
    {
      "notification_id": "notif_7b4c81a2e990",
      "farmer_id": "fmr_0192a8b3",
      "category": "weather_alert",
      "priority": "emergency",
      "title_en": "Critical Weather Warning: Flash Flood Alert",
      "title_hi": "गंभीर मौसम चेतावनी: अचानक बाढ़ का खतरा",
      "body_en": "Torrential rain and flash flood risk in Karnal district in next 3 hours. Move livestock to high ground.",
      "body_hi": "अगले 3 घंटों में करनाल जिले में भारी बारिश और बाढ़ का खतरा। पशुओं को ऊंचे स्थानों पर ले जाएं।",
      "action_url": "/weather",
      "metadata": {
        "hazard_type": "Flash Flood",
        "severity": "Emergency",
        "district": "Karnal"
      },
      "status": "unread",
      "read_at": null,
      "dismissed_at": null,
      "created_at": "2026-11-09T22:30:00.000000",
      "expires_at": null
    }
  ]
  ```

---

### 1.2 Get Unread Notification Count
- **Method:** `GET`
- **Path:** `/farmer/notifications/unread-count`
- **Description:** Get unread notifications counter for sidebar notification badge.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "unread_count": 3
  }
  ```

---

### 1.3 Mark Notification as Read
- **Method:** `POST`
- **Path:** `/farmer/notifications/{notification_id}/read`
- **Description:** Mark a specific notification as read. Fails with `404` or `403` if notification belongs to another farmer.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "notification_id": "notif_7b4c81a2e990",
    "status": "read",
    "read_at": "2026-11-09T23:05:12.441829"
  }
  ```

---

### 1.4 Dismiss Notification
- **Method:** `POST`
- **Path:** `/farmer/notifications/{notification_id}/dismiss`
- **Description:** Dismiss a notification from the primary inbox.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "notification_id": "notif_7b4c81a2e990",
    "status": "dismissed",
    "dismissed_at": "2026-11-09T23:05:30.120541"
  }
  ```

---

### 1.5 Mark All Notifications Read
- **Method:** `POST`
- **Path:** `/farmer/notifications/mark-all-read`
- **Description:** Bulk mark all active unread notifications as read for the authenticated farmer.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "marked_count": 5
  }
  ```

---

## 2. Notification Preferences & DPDP Consents

### 2.1 Get Notification Preferences
- **Method:** `GET`
- **Path:** `/farmer/notifications/preferences`
- **Description:** Retrieve current channel subscriptions, quiet hours, and emergency override settings.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "farmer_id": "fmr_0192a8b3",
    "channels_enabled": ["in_app", "sms", "whatsapp", "voice_call"],
    "categories_enabled": ["weather_alert", "crop_reminder", "market_intelligence", "expert_review"],
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "06:00",
    "emergency_override": true,
    "preferred_language": "hi-IN",
    "updated_at": "2026-11-09T18:00:00"
  }
  ```

---

### 2.2 Update Notification Preferences
- **Method:** `PUT`
- **Path:** `/farmer/notifications/preferences`
- **Description:** Update communication channels, quiet hours intervals, and category opt-ins.
- **Allowed Roles:** `farmer`
- **Request Body:**
  ```json
  {
    "channels_enabled": ["in_app", "sms", "voice_call"],
    "categories_enabled": ["weather_alert", "crop_reminder"],
    "quiet_hours_start": "21:30",
    "quiet_hours_end": "05:30",
    "emergency_override": true,
    "preferred_language": "hi-IN"
  }
  ```
- **Response `200 OK`:** Returns updated `NotificationPreference` record.

---

### 2.3 Get Channel Consents (DPDP Audit Trail)
- **Method:** `GET`
- **Path:** `/farmer/notifications/consents`
- **Description:** List explicit channel opt-in/opt-out consents under Digital Personal Data Protection Act.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  [
    {
      "farmer_id": "fmr_0192a8b3",
      "channel": "sms",
      "is_granted": true,
      "consent_source": "farmer_preferences_ui",
      "updated_at": "2026-11-09T18:00:00"
    },
    {
      "farmer_id": "fmr_0192a8b3",
      "channel": "whatsapp",
      "is_granted": false,
      "consent_source": "farmer_preferences_ui",
      "updated_at": "2026-11-09T18:00:00"
    }
  ]
  ```

---

### 2.4 Update Channel Consent
- **Method:** `PUT`
- **Path:** `/farmer/notifications/consents`
- **Description:** Record explicit opt-in or revocation of consent for a specific delivery channel.
- **Allowed Roles:** `farmer`
- **Request Body:**
  ```json
  {
    "channel": "whatsapp",
    "is_granted": true,
    "consent_source": "farmer_preferences_ui"
  }
  ```
- **Response `200 OK`:** Returns recorded `NotificationChannelConsent` object.

---

## 3. Smart Farm Reminders Endpoints

### 3.1 List Farm Reminders
- **Method:** `GET`
- **Path:** `/farmer/reminders`
- **Description:** Query reminders with optional status (`pending`, `snoozed`, `completed`, `cancelled`) and date filters.
- **Allowed Roles:** `farmer`
- **Query Parameters:**
  - `status` (optional, string)
  - `from_date` (optional, string `YYYY-MM-DD`)
  - `to_date` (optional, string `YYYY-MM-DD`)
- **Response `200 OK`:**
  ```json
  [
    {
      "reminder_id": "rem_9a12c8fe0114",
      "farmer_id": "fmr_0192a8b3",
      "farm_id": "frm_01",
      "crop_cycle_id": "cyc_wheat_2026",
      "activity_type": "pest_spray",
      "title_en": "Apply Propiconazole 25% EC for Yellow Rust",
      "title_hi": "पीला रतुआ नियंत्रण के लिए प्रोपिकोनाज़ोल 25% EC का छिड़काव करें",
      "description_en": "Spray 200ml per acre in 200L water. Ensure protective gear.",
      "description_hi": "प्रति एकड़ 200 मिली दवा 200 लीटर पानी में मिलाकर छिड़कें।",
      "scheduled_date": "2026-11-12",
      "status": "pending",
      "has_weather_conflict": true,
      "weather_risk_reason_en": "Rain forecast (4.2mm) during scheduled spray. Risk of chemical runoff and reduced efficacy.",
      "weather_risk_reason_hi": "निर्धारित छिड़काव के दौरान बारिश का पूर्वानुमान (4.2mm)। कीटनाशक धुलने का खतरा।",
      "created_at": "2026-11-09T18:00:00",
      "updated_at": "2026-11-09T18:00:00"
    }
  ]
  ```

---

### 3.2 Create Farm Reminder
- **Method:** `POST`
- **Path:** `/farmer/reminders`
- **Description:** Create custom agronomic activity reminder. Microclimate weather risk is evaluated synchronously.
- **Allowed Roles:** `farmer`
- **Request Body:**
  ```json
  {
    "farm_id": "frm_01",
    "crop_cycle_id": "cyc_wheat_2026",
    "activity_type": "fertilization",
    "title_en": "Top Dressing of Urea",
    "title_hi": "यूरिया की टॉप ड्रेसिंग",
    "description_en": "Apply 45kg urea per acre before irrigation.",
    "description_hi": "सिंचाई से पहले प्रति एकड़ 45 किग्रा यूरिया डालें।",
    "scheduled_date": "2026-11-14"
  }
  ```
- **Response `200 OK`:** Returns created `FarmReminderRecord` with evaluated `has_weather_conflict`.

---

### 3.3 Complete Farm Reminder
- **Method:** `POST`
- **Path:** `/farmer/reminders/{reminder_id}/complete`
- **Description:** Mark task as completed.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:**
  ```json
  {
    "reminder_id": "rem_9a12c8fe0114",
    "status": "completed",
    "completed_at": "2026-11-09T23:10:00.000000"
  }
  ```

---

### 3.4 Snooze Farm Reminder
- **Method:** `POST`
- **Path:** `/farmer/reminders/{reminder_id}/snooze`
- **Description:** Postpone reminder by specified days or to a custom target date. Re-evaluates weather conflict risk.
- **Allowed Roles:** `farmer`
- **Request Body:**
  ```json
  {
    "snooze_days": 3,
    "custom_date": null
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "reminder_id": "rem_9a12c8fe0114",
    "status": "snoozed",
    "new_scheduled_date": "2026-11-15"
  }
  ```

---

### 3.5 Auto-Generate Reminders from Crop Cycles
- **Method:** `POST`
- **Path:** `/farmer/reminders/generate-from-cycles`
- **Description:** Scan Phase 6 crop cycles and automatically schedule reminders for planned agronomic milestones.
- **Allowed Roles:** `farmer`
- **Response `200 OK`:** Returns list of newly scheduled `FarmReminderRecord` objects.

---

## 4. Admin Queue & Telemetry Endpoints

### 4.1 Inspect Outbox Queue
- **Method:** `GET`
- **Path:** `/admin/notifications/outbox`
- **Description:** View outbox queue records, retry counters, and dead-letter queue.
- **Allowed Roles:** `administrator`, `agronomist`
- **Query Parameters:**
  - `status` (optional, string: `pending`, `processing`, `delivered`, `failed`, `dead_letter`)
  - `limit` (optional, integer, default `50`)
- **Response `200 OK`:** Array of `NotificationOutboxItem` objects.

---

### 4.2 Process Outbox Batch
- **Method:** `POST`
- **Path:** `/admin/notifications/outbox/process`
- **Description:** Triggers immediate atomic worker lease execution for pending outbox items.
- **Allowed Roles:** `administrator`
- **Query Parameters:**
  - `batch_size` (optional, integer, default `50`)
- **Response `200 OK`:**
  ```json
  {
    "status": "batch_completed",
    "processed": 12,
    "delivered": 10,
    "failed": 2,
    "dead_lettered": 0,
    "timestamp": "2026-11-09T23:12:00.000000"
  }
  ```

---

### 4.3 Manual Retry from Dead-Letter Queue
- **Method:** `POST`
- **Path:** `/admin/notifications/outbox/{outbox_id}/retry`
- **Description:** Reset a dead-letter item to `pending` status with retry count reset to 0.
- **Allowed Roles:** `administrator`
- **Response `200 OK`:**
  ```json
  {
    "outbox_id": "obx_42f01a7e9b01",
    "status": "pending",
    "retry_count": 0
  }
  ```

---

### 4.4 Channel Gateways Telemetry
- **Method:** `GET`
- **Path:** `/admin/notifications/telemetry/channels`
- **Description:** Return truthful configuration and readiness telemetry for all physical channel adapters.
- **Allowed Roles:** `administrator`, `agronomist`
- **Response `200 OK`:**
  ```json
  {
    "in_app": {
      "channel": "in_app",
      "provider": "InAppNotificationCenterProvider",
      "configured": true,
      "environment_mode": "local_sqlite",
      "note": "Always active and guaranteed zero-loss delivery to farmer inbox."
    },
    "voice_call": {
      "channel": "voice_call",
      "provider": "GnaniTimbreVoiceNotificationProvider",
      "configured": true,
      "environment_mode": "gnani_timbre_v2.5",
      "note": "Active with verified Gnani API credentials."
    },
    "sms": {
      "channel": "sms",
      "provider": "SmsNotificationProvider",
      "configured": false,
      "environment_mode": "sandbox",
      "note": "SMS gateway credentials not set in environment."
    },
    "whatsapp": {
      "channel": "whatsapp",
      "provider": "WhatsAppNotificationProvider",
      "configured": false,
      "environment_mode": "sandbox",
      "note": "WhatsApp business API credentials not set in environment."
    },
    "email": {
      "channel": "email",
      "provider": "EmailNotificationProvider",
      "configured": false,
      "environment_mode": "sandbox",
      "note": "SMTP credentials not configured."
    }
  }
  ```

