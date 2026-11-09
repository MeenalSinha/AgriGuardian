# AgriGuardian — Phase 9 Operations & Maintenance Guide
## Managing Proactive Notifications, Outbox Workers & Channel Gateways

This guide details operational procedures for running, monitoring, and maintaining the AgriGuardian Phase 9 Proactive Notifications and Smart Reminders infrastructure in production.

---

## 1. Environment Configuration

Phase 9 integrates with physical gateways via environment variables configured in `.env`:

```env
# ==============================================================================
# PHASE 9: NOTIFICATIONS & OUTBOX CONFIGURATION
# ==============================================================================

# Voice Call Alerts (Gnani Timbre TTS)
GNANI_API_KEY=your_gnani_timbre_api_key_here

# SMS Gateway (CDAC / Twilio / MSG91)
SMS_GATEWAY_URL=https://api.sms-provider.in/v1/send
SMS_GATEWAY_API_KEY=your_sms_api_key
SMS_SENDER_ID=AGRGRD

# WhatsApp Business Cloud API
WHATSAPP_API_URL=https://graph.facebook.com/v18.0/your_phone_number_id/messages
WHATSAPP_ACCESS_TOKEN=your_whatsapp_access_token

# Email Gateway (SMTP)
SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
SMTP_USERNAME=postmaster@agridomain.in
SMTP_PASSWORD=your_smtp_password
SMTP_FROM_EMAIL=alerts@agriguardian.in
```

> **Truth-First Reliability:** If external credentials (`SMS_*`, `WHATSAPP_*`, `SMTP_*`) are omitted, the system safely operates in sandbox mode. Messages targeting unconfigured gateways are recorded with explicit error reason `Provider Not Configured` and never fake delivery confirmations.

---

## 2. Running the Outbox Background Worker

The transactional outbox worker processes pending messages using atomic database leasing (`lease_token` and `lease_expires_at`).

### Option A: Standalone Python Worker Loop
Run the worker script as a background daemon process:
```bash
python -m backend.app.services.notifications.notification_outbox_service --daemon --interval 15
```

### Option B: Crontab Execution (Every Minute)
On Linux/Unix servers or scheduled tasks on Windows:
```cron
* * * * * cd /opt/agriguardian && python -c "from backend.app.services.notifications.notification_service import NotificationService; NotificationService.get_instance().process_outbox(batch_size=50)" >> /var/log/agriguardian_outbox.log 2>&1
```

### Option C: Trigger via REST API
System administrators can trigger batch processing on-demand:
```bash
curl -X POST "http://localhost:8000/admin/notifications/outbox/process?batch_size=50" \
     -H "Authorization: Bearer <admin_token>"
```

---

## 3. Worker Lease Recovery Mechanics

To prevent duplicate deliveries across parallel worker instances while handling worker crashes:
1. **Acquiring Lease:** Worker generates a unique UUID `lease_token` and sets `lease_expires_at = now + 60s`.
2. **Crash Handling:** If a worker process is terminated (OOM, restart, unhandled exception) before completing a batch, its claimed rows remain locked only until `lease_expires_at`.
3. **Automatic Reclamation:** On every execution cycle, the outbox service executes:
   ```sql
   UPDATE notification_outbox
   SET status = 'pending', lease_token = NULL, lease_expires_at = NULL
   WHERE status = 'processing' AND lease_expires_at < CURRENT_TIMESTAMP;
   ```
   This automatically returns abandoned jobs to the queue for immediate pickup.

---

## 4. Dead-Letter Queue (DLQ) Monitoring & Runbook

When an outbox item exceeds `max_retries` (default: 3 attempts with exponential backoff: 30s, 60s, 120s), it enters the `dead_letter` status.

### Diagnosing DLQ Messages
1. **Inspect Dead-Letter Items via UI:**
   Navigate to **🔔 Alerts & Reminders → Tab 4: Outbox Queue & Channel Telemetry**. Set the status filter to `dead_letter`.
2. **Inspect via Database Query:**
   ```sql
   SELECT outbox_id, channel, recipient, retry_count, last_error, updated_at
   FROM notification_outbox
   WHERE status = 'dead_letter';
   ```
3. **Inspecting Provider Attempts:**
   ```sql
   SELECT attempt_id, provider_name, response_code, error_message, latency_ms, created_at
   FROM notification_delivery_attempts
   WHERE outbox_id = 'target_outbox_id'
   ORDER BY created_at DESC;
   ```

### Manual Replay Runbook
After correcting network reachability, invalid telephone format, or third-party provider credentials:
1. **Via UI:** Click the **Retry Dead-Letter Item 🔄** button on the target card.
2. **Via REST API:**
   ```bash
   curl -X POST "http://localhost:8000/admin/notifications/outbox/{outbox_id}/retry" \
        -H "Authorization: Bearer <admin_token>"
   ```
   This resets `status = 'pending'`, zeroes the `retry_count`, clears `last_error`, and schedules the item for immediate delivery.

---

## 5. DPDP Act & Privacy Compliance Operations

Phase 9 implements strict privacy controls per India's Digital Personal Data Protection (DPDP) Act:
1. **Audit Trail:** Every channel consent grant or revocation is recorded in `notification_channel_consents` with timestamp and source.
2. **Consent Withdrawal Enforcement:**
   When a farmer toggles off a channel or revokes consent, the `NotificationPolicyEngine` immediately blocks any future non-emergency event from queuing outbox jobs for that channel.
3. **Quiet Hours Protection:**
   Routine notifications (market price surges, task reminders) arriving between `quiet_hours_start` (22:00) and `quiet_hours_end` (06:00) are delivered exclusively to the in-app notification center, avoiding intrusive nocturnal SMS or phone calls.
4. **Emergency Exemption:**
   Severe life-safety weather emergencies (flash floods, severe cyclonic gusts) automatically override quiet hours only when `emergency_override` is enabled (default: true).

---

## 6. Database Health & Retention Maintenance

To ensure optimal query performance on high-volume production deployments:
1. **Sliding Window Pruning:**
   Historical events older than 90 days can be archived or purged:
   ```sql
   DELETE FROM notification_events WHERE created_at < date('now', '-90 days');
   DELETE FROM notification_delivery_attempts WHERE created_at < date('now', '-90 days');
   ```
2. **Vacuuming SQLite:**
   Run periodically during off-peak maintenance windows:
   ```sql
   VACUUM;
   ANALYZE;
   ```

