// AgriGuardian Master Client Synchronizer
// Coordinates Bootstrap, Push Mutations, Attachment Uploads, and Incremental Pulls

import { dbStorage } from '../storage/indexed_db.js';
import { mutationQueue } from './offline_mutation_queue.js';
import { networkManager } from './offline_network_manager.js';
import { MutationStatus, NetworkQuality } from '../types.js';

export class OfflineSyncManager {
  constructor(storage = dbStorage, queue = mutationQueue, netManager = networkManager) {
    this.storage = storage;
    this.queue = queue;
    this.netManager = netManager;
    this.isSyncing = false;
    this.authToken = null;
    this.syncListeners = new Set();

    // Auto-sync when connectivity transitions to ONLINE
    this.netManager.subscribe((status) => {
      if (status === NetworkQuality.ONLINE) {
        console.log('[SyncManager] Online detected. Scheduling background synchronization...');
        this.syncAll().catch((err) => console.warn('[SyncManager] Auto-sync error:', err));
      }
    });
  }

  setAuthToken(token) {
    this.authToken = token;
  }

  onSyncStatusChange(cb) {
    this.syncListeners.add(cb);
    return () => this.syncListeners.delete(cb);
  }

  _notifyListeners(state) {
    for (const cb of this.syncListeners) {
      try {
        cb(state);
      } catch (e) {
        console.error(e);
      }
    }
  }

  async getDeviceId() {
    let devId = await this.storage.getMeta('device_id');
    if (!devId) {
      devId = 'dev_' + (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 12));
      await this.storage.setMeta('device_id', devId);
    }
    return devId;
  }

  _getHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    return headers;
  }

  // Ensure device is registered with backend
  async ensureDeviceRegistered() {
    const deviceId = await this.getDeviceId();
    const registered = await this.storage.getMeta('device_registered');
    if (registered) return deviceId;

    try {
      const res = await fetch('/api/v1/sync/devices/register', {
        method: 'POST',
        headers: this._getHeaders(),
        body: JSON.stringify({
          device_id: deviceId,
          device_name: navigator.userAgent.includes('Mobile') ? 'Farmer Mobile Device' : 'Desktop Browser',
          device_type: 'pwa_mobile',
          app_version: '1.10.0'
        })
      });
      if (res.ok) {
        await this.storage.setMeta('device_registered', true);
        console.log(`[SyncManager] Device ${deviceId} registered with server.`);
      }
    } catch (e) {
      console.warn('[SyncManager] Device registration postponed (offline).');
    }
    return deviceId;
  }

  // Initial Bootstrap: downloads complete authorized offline state
  async bootstrap() {
    if (this.netManager.status === NetworkQuality.OFFLINE) {
      throw new Error('Cannot bootstrap while offline.');
    }

    const deviceId = await this.ensureDeviceRegistered();
    this._notifyListeners({ phase: 'bootstrap_start' });

    console.log('[SyncManager] Fetching offline bootstrap bundle...');
    const res = await fetch(`/api/v1/sync/bootstrap?device_id=${encodeURIComponent(deviceId)}`, {
      method: 'GET',
      headers: this._getHeaders()
    });

    if (!res.ok) {
      throw new Error(`Bootstrap failed with status ${res.status}`);
    }

    const data = await res.json();

    // Populate IndexedDB stores
    if (data.farms) await this.storage.replaceAll('farms', data.farms);
    if (data.plots) await this.storage.replaceAll('plots', data.plots);
    if (data.crop_cycles) await this.storage.replaceAll('crop_cycles', data.crop_cycles);
    if (data.activities) await this.storage.replaceAll('activities', data.activities);
    if (data.reminders) await this.storage.replaceAll('reminders', data.reminders);
    if (data.notifications) await this.storage.replaceAll('notifications', data.notifications);
    if (data.advisories) await this.storage.replaceAll('advisories', data.advisories);

    await this.storage.setMeta('sequence_cursor', data.sequence_cursor || 0);
    await this.storage.setMeta('last_sync_timestamp', data.server_timestamp);
    if (data.farmer_profile) {
      await this.storage.setMeta('farmer_profile', data.farmer_profile);
      await this.storage.setMeta('farmer_id', data.farmer_profile.farmer_id);
    }

    console.log('[SyncManager] Bootstrap successfully applied to local IndexedDB.');
    this._notifyListeners({ phase: 'bootstrap_complete', timestamp: data.server_timestamp });
    return data;
  }

  // Push pending mutations to backend
  async pushPendingMutations() {
    const pending = await this.queue.getPendingMutations();
    if (pending.length === 0) return { processed: 0, applied: 0, conflicts: 0 };

    const deviceId = await this.ensureDeviceRegistered();
    console.log(`[SyncManager] Pushing ${pending.length} pending mutations...`);

    const mutationIds = pending.map((m) => m.mutation_id);
    await this.queue.markSyncing(mutationIds);

    const payload = {
      device_id: deviceId,
      mutations: pending.map((m) => ({
        mutation_id: m.mutation_id,
        resource_type: m.resource_type,
        operation: m.operation,
        client_record_id: m.client_record_id,
        payload: m.payload,
        base_version: m.base_version || 0
      }))
    };

    const res = await fetch('/api/v1/sync/push', {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      for (const m of pending) {
        await this.queue.recordResult(m.mutation_id, MutationStatus.FAILED, null, 1, `HTTP ${res.status}`);
      }
      throw new Error(`Push failed with HTTP ${res.status}`);
    }

    const pushResult = await res.json();
    for (const item of pushResult.results) {
      const newStatus = item.status === 'applied' ? MutationStatus.SYNCED :
                        item.status === 'conflict' ? MutationStatus.CONFLICT :
                        MutationStatus.REJECTED;

      await this.queue.recordResult(
        item.mutation_id,
        newStatus,
        item.server_record_id,
        item.server_version,
        item.error_message,
        item.conflict_details
      );
    }

    return pushResult;
  }

  // Pull incremental changes from backend
  async pullChanges() {
    const deviceId = await this.ensureDeviceRegistered();
    const cursor = (await this.storage.getMeta('sequence_cursor', 0)) || 0;

    console.log(`[SyncManager] Pulling changes after sequence_id ${cursor}...`);
    const res = await fetch('/api/v1/sync/pull', {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify({
        device_id: deviceId,
        since_sequence_id: cursor,
        limit: 100
      })
    });

    if (!res.ok) {
      throw new Error(`Pull failed with HTTP ${res.status}`);
    }

    const pullResult = await res.json();
    for (const chg of pullResult.changes) {
      const storeName = this._mapResourceTypeToStore(chg.resource_type);
      if (!storeName) continue;

      if (chg.operation === 'DELETE') {
        await this.storage.delete(storeName, chg.resource_id);
      } else if (chg.payload) {
        await this.storage.put(storeName, chg.payload);
      }
    }

    await this.storage.setMeta('sequence_cursor', pullResult.new_sequence_id);
    await this.storage.setMeta('last_sync_timestamp', pullResult.server_timestamp);

    return pullResult;
  }

  // Upload queued offline photos / attachments
  async uploadPendingAttachments() {
    const all = await this.storage.getAll('attachments');
    const pending = all.filter((a) => !a.is_uploaded && a.blob);
    const deviceId = await this.ensureDeviceRegistered();

    for (const att of pending) {
      try {
        const formData = new FormData();
        formData.append('device_id', deviceId);
        formData.append('resource_type', att.resource_type || 'crop_diagnosis');
        if (att.resource_id) formData.append('resource_id', att.resource_id);
        formData.append('file', att.blob, att.file_name || 'photo.jpg');

        const headers = {};
        if (this.authToken) headers['Authorization'] = `Bearer ${this.authToken}`;

        const res = await fetch('/api/v1/sync/attachments', {
          method: 'POST',
          headers,
          body: formData
        });

        if (res.ok) {
          const uploaded = await res.json();
          att.is_uploaded = true;
          att.server_attachment_id = uploaded.attachment_id;
          att.uploaded_at = new Date().toISOString();
          delete att.blob; // Free up binary blob from IndexedDB after successful upload
          await this.storage.put('attachments', att);
          console.log(`[SyncManager] Uploaded attachment ${att.attachment_id}`);
        }
      } catch (err) {
        console.warn(`[SyncManager] Attachment upload failed for ${att.attachment_id}:`, err);
      }
    }
  }

  _mapResourceTypeToStore(resourceType) {
    switch (resourceType) {
      case 'farm': return 'farms';
      case 'farm_plot': return 'plots';
      case 'crop_cycle': return 'crop_cycles';
      case 'farm_activity': return 'activities';
      case 'farm_reminder': return 'reminders';
      case 'notification': return 'notifications';
      case 'advisory_case': return 'advisories';
      default: return null;
    }
  }

  // Master Sync Execution
  async syncAll() {
    if (this.isSyncing) {
      console.log('[SyncManager] Sync already in progress, skipping concurrent run.');
      return;
    }

    if (this.netManager.status === NetworkQuality.OFFLINE) {
      console.log('[SyncManager] Offline. Synchronization deferred.');
      return { status: 'offline', message: 'You are currently offline. Changes are saved locally.' };
    }

    this.isSyncing = true;
    this._notifyListeners({ phase: 'sync_start' });

    try {
      // 1. Push pending offline mutations
      const pushRes = await this.pushPendingMutations();

      // 2. Upload pending attachments
      await this.uploadPendingAttachments();

      // 3. Pull incremental deltas from server
      const pullRes = await this.pullChanges();

      const lastSync = await this.storage.getMeta('last_sync_timestamp');
      this._notifyListeners({
        phase: 'sync_success',
        timestamp: lastSync,
        push: pushRes,
        pull: pullRes
      });

      return {
        status: 'success',
        last_sync: lastSync,
        applied: pushRes ? pushRes.applied_count : 0,
        conflicts: pushRes ? pushRes.conflict_count : 0,
        changes_pulled: pullRes ? pullRes.changes.length : 0
      };
    } catch (err) {
      console.error('[SyncManager] Sync error:', err);
      this._notifyListeners({ phase: 'sync_error', error: err.message });
      throw err;
    } finally {
      this.isSyncing = false;
    }
  }
}

export const syncManager = new OfflineSyncManager();
