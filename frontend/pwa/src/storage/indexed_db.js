// AgriGuardian IndexedDB Offline Storage Manager
// Provides structured, user-isolated local persistence for farms, crops, activities, reminders, and offline mutations

const DB_NAME = 'AgriGuardianOfflineDB';
const DB_VERSION = 1;

export class IndexedDBStorage {
  constructor() {
    this.db = null;
    this.initPromise = this._init();
  }

  async _init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Structured entity stores
        if (!db.objectStoreNames.contains('farms')) {
          db.createObjectStore('farms', { keyPath: 'farm_id' });
        }
        if (!db.objectStoreNames.contains('plots')) {
          db.createObjectStore('plots', { keyPath: 'plot_id' });
        }
        if (!db.objectStoreNames.contains('crop_cycles')) {
          db.createObjectStore('crop_cycles', { keyPath: 'cycle_id' });
        }
        if (!db.objectStoreNames.contains('activities')) {
          const actStore = db.createObjectStore('activities', { keyPath: 'activity_id' });
          actStore.createIndex('status', 'status', { unique: false });
          actStore.createIndex('scheduled_date', 'scheduled_date', { unique: false });
        }
        if (!db.objectStoreNames.contains('reminders')) {
          const remStore = db.createObjectStore('reminders', { keyPath: 'reminder_id' });
          remStore.createIndex('status', 'status', { unique: false });
          remStore.createIndex('scheduled_date', 'scheduled_date', { unique: false });
        }
        if (!db.objectStoreNames.contains('notifications')) {
          db.createObjectStore('notifications', { keyPath: 'notification_id' });
        }
        if (!db.objectStoreNames.contains('advisories')) {
          db.createObjectStore('advisories', { keyPath: 'case_id' });
        }

        // Offline Durable Queue & Attachments
        if (!db.objectStoreNames.contains('mutations')) {
          const mutStore = db.createObjectStore('mutations', { keyPath: 'mutation_id' });
          mutStore.createIndex('status', 'status', { unique: false });
          mutStore.createIndex('resource_type', 'resource_type', { unique: false });
        }
        if (!db.objectStoreNames.contains('attachments')) {
          const attStore = db.createObjectStore('attachments', { keyPath: 'attachment_id' });
          attStore.createIndex('resource_type', 'resource_type', { unique: false });
        }

        // Configuration & Metadata store (device_id, last_sync_time, cursor, language)
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'key' });
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        resolve(this.db);
      };

      request.onerror = (event) => {
        console.error('[IndexedDB] Failed to open database:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  async ready() {
    if (!this.db) {
      await this.initPromise;
    }
    return this.db;
  }

  // Generic Transaction Helpers
  async put(storeName, item) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.put(item);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async get(storeName, key) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async getAll(storeName) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readonly');
      const store = tx.objectStore(storeName);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async delete(storeName, key) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.delete(key);
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  async clearStore(storeName) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      const req = store.clear();
      req.onsuccess = () => resolve(true);
      req.onerror = () => reject(req.error);
    });
  }

  // Metadata Helpers
  async setMeta(key, value) {
    return this.put('metadata', { key, value, updated_at: new Date().toISOString() });
  }

  async getMeta(key, defaultValue = null) {
    const item = await this.get('metadata', key);
    return item ? item.value : defaultValue;
  }

  // Batch Replace (e.g. after bootstrap pull)
  async replaceAll(storeName, items) {
    const db = await this.ready();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, 'readwrite');
      const store = tx.objectStore(storeName);
      store.clear();
      for (const item of items) {
        store.put(item);
      }
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  }

  // Security: Clear user data on account switch or logout
  async clearUserData() {
    console.warn('[IndexedDB] Purging local user data on account switch/logout.');
    const stores = ['farms', 'plots', 'crop_cycles', 'activities', 'reminders', 'notifications', 'advisories', 'mutations', 'attachments'];
    for (const s of stores) {
      await this.clearStore(s);
    }
    await this.setMeta('farmer_id', null);
    await this.setMeta('last_sync_timestamp', null);
    await this.setMeta('sequence_cursor', 0);
  }
}

export const dbStorage = new IndexedDBStorage();
