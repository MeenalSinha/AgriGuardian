# AGRIGUARDIAN — PHASE 10 SYNCHRONIZATION API DOCUMENTATION
## RESTful Endpoints, Request/Response Schemas, Idempotency & Conflict Management

**API Version:** `v1`  
**Base Path:** `/api/v1/sync`  
**Authentication:** JWT Bearer Token (`Authorization: Bearer <token>`)

---

## 1. Overview & Protocol Principles

The AgriGuardian Synchronization API enables low-bandwidth, bidirectional data synchronization between offline client applications (such as the Progressive Web App at `/pwa`) and the centralized FastAPI backend.

### Core Protocol Principles:
1. **Delta Pulling via Change Feed:** Clients pull only modified records using a sequence-based cursor (`since_sequence_id`), significantly reducing data usage over 2G/EDGE networks.
2. **Idempotent Batch Push:** Client mutations include unique UUIDs (`mutation_id`). The server deduplicates retried requests via `sync_mutation_results`, returning previously calculated results without side effects.
3. **Optimistic Locking:** Every syncable entity (`farms`, `farm_plots`, `crop_cycles`, `farm_activities`, `farm_reminders`) tracks a `version_number`. Client mutations submit their `base_version`. If `server_version > base_version`, the server flags a collision and records a conflict instead of blindly overwriting.
4. **Deletion Tombstones:** Deletions are published to the change feed with `operation: "DELETE"`, enabling client IndexedDB stores to purge deleted entities.
5. **SHA-256 Verified Attachments:** Offline photographs are validated with SHA-256 hashes to guarantee data integrity during intermittent network transmission.

---

## 2. API Endpoints Reference

### 2.1 Device Management

#### `POST /api/v1/sync/devices/register`
Register a new client device or update an existing device registration.

**Request Body (`DeviceRegisterRequest`):**
```json
{
  "device_id": "dev_f47ac10b",
  "device_name": "Redmi Note 12 - Farmer Ramesh",
  "device_type": "pwa_mobile",
  "app_version": "1.10.0"
}
```

**Response (`200 OK` - `DeviceResponse`):**
```json
{
  "device_id": "dev_f47ac10b",
  "user_id": "usr_9b1deb4d",
  "farmer_id": "fmr_1a2b3c4d",
  "device_name": "Redmi Note 12 - Farmer Ramesh",
  "device_type": "pwa_mobile",
  "app_version": "1.10.0",
  "is_active": true,
  "registered_at": "2026-11-10T02:00:00Z",
  "last_sync_at": "2026-11-10T02:15:00Z"
}
```

---

#### `GET /api/v1/sync/devices`
List all registered devices belonging to the authenticated user.

**Response (`200 OK` - `List[DeviceResponse]`):**
```json
[
  {
    "device_id": "dev_f47ac10b",
    "user_id": "usr_9b1deb4d",
    "farmer_id": "fmr_1a2b3c4d",
    "device_name": "Redmi Note 12 - Farmer Ramesh",
    "device_type": "pwa_mobile",
    "app_version": "1.10.0",
    "is_active": true,
    "registered_at": "2026-11-10T02:00:00Z",
    "last_sync_at": "2026-11-10T02:15:00Z"
  }
]
```

---

#### `DELETE /api/v1/sync/devices/{device_id}`
Revoke device access. Once revoked, the device cannot pull data or push mutations.

**Response (`200 OK`):**
```json
{
  "detail": "Device dev_f47ac10b revoked successfully"
}
```

---

### 2.2 Data Ingestion & Retrieval (Bootstrap & Pull)

#### `GET /api/v1/sync/bootstrap`
Fetch the complete offline bootstrap bundle for newly initialized client devices.

**Query Parameters:**
- `device_id` (required, string): The registered device identifier.

**Response (`200 OK` - `SyncBootstrapResponse`):**
```json
{
  "farmer_profile": {
    "farmer_id": "fmr_1a2b3c4d",
    "user_id": "usr_9b1deb4d",
    "full_name": "Ramesh Kumar Patel",
    "state": "Madhya Pradesh",
    "district": "Sehore"
  },
  "farms": [
    {
      "farm_id": "farm_001",
      "farmer_id": "fmr_1a2b3c4d",
      "farm_name": "Main Narmada Field",
      "total_area_acres": 4.5,
      "version_number": 1
    }
  ],
  "plots": [],
  "crop_cycles": [
    {
      "cycle_id": "cycle_soy_01",
      "farm_id": "farm_001",
      "crop_name": "Soybean",
      "variety_name": "JS 335",
      "current_stage": "vegetative",
      "version_number": 1
    }
  ],
  "activities": [
    {
      "activity_id": "act_weeding_01",
      "farm_id": "farm_001",
      "activity_type": "weeding",
      "activity_title": "Manual Inter-row Weeding",
      "scheduled_date": "2026-11-12",
      "status": "planned",
      "version_number": 1
    }
  ],
  "reminders": [
    {
      "reminder_id": "rem_spray_01",
      "activity_type": "pest_spray",
      "title_en": "Spray Chlorpyrifos",
      "title_hi": "कीटनाशक छिड़काव",
      "scheduled_date": "2026-11-14",
      "status": "scheduled",
      "version_number": 1
    }
  ],
  "notifications": [],
  "advisory_cases": [],
  "server_sequence_id": 42,
  "bundle_timestamp": "2026-11-10T02:20:00Z"
}
```

---

#### `POST /api/v1/sync/pull`
Pull incremental changes that occurred on the server since a specific sequence ID.

**Request Body (`SyncPullRequest`):**
```json
{
  "device_id": "dev_f47ac10b",
  "since_sequence_id": 42,
  "limit": 50
}
```

**Response (`200 OK` - `SyncPullResponse`):**
```json
{
  "changes": [
    {
      "sequence_id": 43,
      "resource_type": "farm_reminder",
      "resource_id": "rem_spray_01",
      "operation": "UPDATE",
      "version_number": 2,
      "payload": {
        "reminder_id": "rem_spray_01",
        "title_en": "Spray Chlorpyrifos (Delayed by Agronomist)",
        "scheduled_date": "2026-11-16",
        "status": "scheduled"
      },
      "timestamp": "2026-11-10T02:25:00Z"
    },
    {
      "sequence_id": 44,
      "resource_type": "farm_activity",
      "resource_id": "act_old_weeding",
      "operation": "DELETE",
      "version_number": 2,
      "payload": null,
      "timestamp": "2026-11-10T02:26:00Z"
    }
  ],
  "new_sequence_id": 44,
  "has_more": false,
  "server_timestamp": "2026-11-10T02:27:00Z"
}
```

---

### 2.3 Batch Mutation Push

#### `POST /api/v1/sync/push`
Push a batch of offline-generated mutations to the backend.

**Request Body (`SyncPushRequest`):**
```json
{
  "device_id": "dev_f47ac10b",
  "mutations": [
    {
      "mutation_id": "mut_create_act_99",
      "resource_type": "farm_activity",
      "operation": "CREATE",
      "client_record_id": "act_local_99",
      "payload": {
        "farm_id": "farm_001",
        "activity_type": "irrigation",
        "activity_title": "Drip Irrigation Cycle",
        "scheduled_date": "2026-11-11",
        "status": "planned"
      },
      "base_version": 0
    },
    {
      "mutation_id": "mut_complete_rem_01",
      "resource_type": "farm_reminder",
      "operation": "COMPLETE",
      "client_record_id": "rem_spray_01",
      "payload": {
        "reminder_id": "rem_spray_01"
      },
      "base_version": 1
    }
  ]
}
```

**Response (`200 OK` - `SyncPushResponse`):**
```json
{
  "processed_count": 2,
  "applied_count": 2,
  "conflict_count": 0,
  "results": [
    {
      "mutation_id": "mut_create_act_99",
      "client_record_id": "act_local_99",
      "server_record_id": "act_local_99",
      "status": "applied",
      "server_version": 1,
      "conflict_details": null,
      "error_message": null
    },
    {
      "mutation_id": "mut_complete_rem_01",
      "client_record_id": "rem_spray_01",
      "server_record_id": "rem_spray_01",
      "status": "applied",
      "server_version": 2,
      "conflict_details": null,
      "error_message": null
    }
  ],
  "server_timestamp": "2026-11-10T02:28:00Z"
}
```

---

### 2.4 Conflict Detection & Resolution

When a push mutation detects `server_version > mutation.base_version`, the server returns `status: "conflict"` and stores the item in `sync_conflicts`.

#### `GET /api/v1/sync/conflicts`
List pending conflicts for the authenticated farmer.

**Query Parameters:**
- `status` (optional, string, default `"pending"`): Filter by conflict status (`pending`, `resolved`).

**Response (`200 OK` - `List[ConflictItem]`):**
```json
[
  {
    "conflict_id": "cnf_8f3a9e21",
    "mutation_id": "mut_edit_rem_stale",
    "device_id": "dev_f47ac10b",
    "farmer_id": "fmr_1a2b3c4d",
    "resource_type": "farm_reminder",
    "client_record_id": "rem_spray_01",
    "server_record_id": "rem_spray_01",
    "client_payload": {
      "title_en": "Spray Chlorpyrifos Early",
      "scheduled_date": "2026-11-13"
    },
    "server_payload": {
      "title_en": "Spray Chlorpyrifos (Delayed by Agronomist)",
      "scheduled_date": "2026-11-16",
      "version_number": 2
    },
    "resolution_strategy": "pending",
    "status": "pending",
    "created_at": "2026-11-10T02:28:00Z",
    "resolved_at": null
  }
]
```

---

#### `POST /api/v1/sync/conflicts/{conflict_id}/resolve`
Resolve an existing conflict.

**Request Body (`ConflictResolveRequest`):**
```json
{
  "resolution_strategy": "client_wins",
  "merged_payload": null
}
```
*Options for `resolution_strategy`:* `client_wins`, `server_wins`, `manual_merge`.

**Response (`200 OK` - `ConflictResolveResponse`):**
```json
{
  "conflict_id": "cnf_8f3a9e21",
  "status": "resolved",
  "applied_strategy": "client_wins",
  "resolved_payload": {
    "title_en": "Spray Chlorpyrifos Early",
    "scheduled_date": "2026-11-13"
  },
  "resolved_at": "2026-11-10T02:30:00Z"
}
```

---

### 2.5 Offline Photographs & Attachments

#### `POST /api/v1/sync/attachments/upload`
Upload a photograph captured offline using `multipart/form-data`.

**Form Fields:**
- `device_id` (string, required): Registered device identifier.
- `resource_type` (string, required): e.g. `crop_diagnosis`, `farm_activity`.
- `resource_id` (string, optional): Associated entity ID if known.
- `file` (UploadFile, required): The image binary (JPEG/PNG/WebP, max 10MB).

**Response (`200 OK` - `AttachmentUploadResponse`):**
```json
{
  "attachment_id": "att_c81e728d",
  "resource_type": "crop_diagnosis",
  "resource_id": "case_diag_01",
  "filename": "leaf_rust_01.jpg",
  "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "file_size": 245192,
  "uploaded_at": "2026-11-10T02:31:00Z"
}
```

---

#### `GET /api/v1/sync/attachments/{attachment_id}`
Stream and download the stored attachment file.

**Response (`200 OK`):**
Binary image stream with appropriate `Content-Type` header (e.g. `image/jpeg`).

---

### 2.6 Sync Telemetry & Network Status

#### `GET /api/v1/sync/status`
Lightweight network probe endpoint for evaluating latency, connectivity, and server wall-clock time.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "server_time": "2026-11-10T02:32:00.123456+00:00",
  "service": "AgriGuardian Sync Service",
  "version": "1.10.0"
}
```

---

#### `GET /api/v1/sync/telemetry`
Retrieve sync telemetry logs for performance monitoring and diagnostic audits.

**Query Parameters:**
- `limit` (optional, integer, default `50`): Maximum records to retrieve.

**Response (`200 OK` - `List[SyncTelemetryReport]`):**
```json
[
  {
    "session_id": "tel_dev_f47ac10b_a1b2c3d4e5",
    "device_id": "dev_f47ac10b",
    "farmer_id": "fmr_1a2b3c4d",
    "sync_type": "push",
    "duration_ms": 42,
    "pull_count": 0,
    "push_count": 2,
    "conflict_count": 0,
    "bytes_transferred": 512,
    "status": "success",
    "created_at": "2026-11-10T02:28:00Z"
  }
]
```

---

## 3. Error Handling & HTTP Status Codes

| Code | Name | Description |
|------|------|-------------|
| `400` | Bad Request | Malformed payload or invalid enum parameter |
| `401` | Unauthorized | Missing or expired JWT bearer token |
| `403` | Forbidden | Cross-tenant access attempt or inactive device |
| `404` | Not Found | Target device, conflict, or attachment not found |
| `409` | Conflict | Optimistic locking collision on record update |
| `413` | Content Too Large | Attachment exceeds 10 MB maximum limit |
| `415` | Unsupported Media Type | Non-image MIME type uploaded |
| `500` | Internal Server Error | Unhandled server exception |
