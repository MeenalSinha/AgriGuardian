// AgriGuardian Synchronization Center
// Dedicated offline operations control pane for manual sync, inspection of queued mutations, and low-data options

import { syncManager } from '../sync/offline_sync_manager.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';
import { networkManager } from '../sync/offline_network_manager.js';
import { dbStorage } from '../storage/indexed_db.js';
import { NetworkQuality } from '../types.js';

export class SyncCenter {
  constructor(containerId = 'sync-center-root') {
    this.container = document.getElementById(containerId);
    this.pendingMutations = [];
    this.pendingAttachments = [];
    this.conflicts = [];
    this.isSyncing = false;
    this.lowDataMode = false;

    syncManager.onSyncStatusChange((state) => {
      this.isSyncing = state.phase === 'sync_start' || state.phase === 'bootstrap_start';
      this.refresh();
    });
  }

  async refresh() {
    this.pendingMutations = await mutationQueue.getPendingMutations();
    const allAttachments = await dbStorage.getAll('attachments');
    this.pendingAttachments = allAttachments.filter((a) => !a.is_uploaded);
    this.lowDataMode = (await dbStorage.getMeta('low_data_mode')) || false;
    this.render();
  }

  formatDate(isoString) {
    if (!isoString) return '-';
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return isoString;
    }
  }

  async handleManualSync() {
    const btn = this.container.querySelector('#btn-manual-sync');
    if (btn) btn.disabled = true;
    try {
      await syncManager.syncAll();
      alert('Synchronization completed successfully!');
    } catch (err) {
      alert(`Sync failed: ${err.message}`);
    } finally {
      await this.refresh();
    }
  }

  async handleBootstrap() {
    const confirmed = confirm('Download complete farm context from server? This will refresh your local cache.');
    if (!confirmed) return;
    try {
      await syncManager.bootstrap();
      alert('Bootstrap synchronization successful!');
    } catch (err) {
      alert(`Bootstrap failed: ${err.message}`);
    } finally {
      await this.refresh();
    }
  }

  async toggleLowDataMode(enabled) {
    this.lowDataMode = enabled;
    await dbStorage.setMeta('low_data_mode', enabled);
    this.refresh();
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';
    const isOnline = networkManager.status !== NetworkQuality.OFFLINE;

    this.container.innerHTML = `
      <div class="sync-center-card">
        <div class="sync-header">
          <h2>🔄 ${lang === 'hi' ? 'सिंक्रोनाइज़ेशन केंद्र' : 'Synchronization Center'}</h2>
          <span class="sync-status-tag ${isOnline ? 'tag-online' : 'tag-offline'}">
            ${isOnline ? (lang === 'hi' ? 'क्लाउड कनेक्टेड' : 'Cloud Connected') : (lang === 'hi' ? 'ऑफलाइन' : 'Offline')}
          </span>
        </div>

        <p class="sync-desc">
          ${lang === 'hi'
            ? 'खेत में किए गए सभी कार्य सुरक्षित रूप से इस डिवाइस पर सहेजे जाते हैं। इंटरनेट मिलने पर वे स्वचालित रूप से क्लाउड पर सिंक हो जाएंगे।'
            : 'All field observations, activities, and reminders are safely queued on your device. When network returns, they sync automatically with zero data loss.'}
        </p>

        <!-- Actions Bar -->
        <div class="sync-actions-grid">
          <button id="btn-manual-sync" class="btn-primary" ${!isOnline || this.isSyncing ? 'disabled' : ''}>
            ${this.isSyncing ? '⏳ ' + (lang === 'hi' ? 'सिंक हो रहा है...' : 'Syncing...') : '⚡ ' + (lang === 'hi' ? 'अभी सिंक करें' : 'Sync Now')}
          </button>
          <button id="btn-bootstrap" class="btn-secondary" ${!isOnline || this.isSyncing ? 'disabled' : ''}>
            📥 ${lang === 'hi' ? 'पूरा डेटा डाउनलोड करें' : 'Full Data Bootstrap'}
          </button>
        </div>

        <!-- Low Data Mode Preference -->
        <div class="preference-box">
          <label class="toggle-label">
            <input type="checkbox" id="chk-low-data" ${this.lowDataMode ? 'checked' : ''}>
            <span><strong>📉 ${lang === 'hi' ? 'लो-डेटा मोड' : 'Low-Bandwidth Mode'}</strong> (${lang === 'hi' ? 'कम डेटा खपत, केवल मुख्य जानकारी' : 'Save mobile data, defer large images'})</span>
          </label>
        </div>

        <!-- Pending Mutations Queue -->
        <div class="sync-section">
          <h3>📋 ${lang === 'hi' ? 'लंबित सिंक रिकॉर्ड्स' : 'Pending Server Updates'} (${this.pendingMutations.length})</h3>
          ${this.pendingMutations.length === 0
            ? `<div class="empty-sync-box">✅ ${lang === 'hi' ? 'सभी बदलाव सर्वर पर सुरक्षित हैं।' : 'All local changes are fully synchronized.'}</div>`
            : `
              <div class="mutations-list">
                ${this.pendingMutations.map((m) => `
                  <div class="mutation-item status-${m.status}">
                    <div class="mut-info">
                      <span class="mut-op">${m.operation}</span>
                      <strong class="mut-res">${m.resource_type.replace('_', ' ')}</strong>
                      <span class="mut-time">${this.formatDate(m.created_at)}</span>
                    </div>
                    <div class="mut-status">
                      <span class="pill-badge pill-${m.status}">
                        ${m.status === 'pending_sync' ? '⏳ ' + (lang === 'hi' ? 'सिंक बाकी' : 'Pending Sync') : m.status}
                      </span>
                    </div>
                  </div>
                `).join('')}
              </div>
            `}
        </div>

        <!-- Pending Photographs -->
        <div class="sync-section">
          <h3>📷 ${lang === 'hi' ? 'अपलोड होने वाली तस्वीरें' : 'Queued Photographs'} (${this.pendingAttachments.length})</h3>
          ${this.pendingAttachments.length === 0
            ? `<div class="empty-sync-box">🌾 ${lang === 'hi' ? 'कोई लंबित तस्वीर नहीं।' : 'No photos pending upload.'}</div>`
            : `
              <div class="attachments-list">
                ${this.pendingAttachments.map((a) => `
                  <div class="attachment-item">
                    <span>🖼️ ${a.file_name}</span>
                    <span class="att-size">(${Math.round(a.byte_size / 1024)} KB)</span>
                    <span class="pill-badge pill-pending_sync">⏳ ${lang === 'hi' ? 'अपलोड बाकी' : 'Waiting for network'}</span>
                  </div>
                `).join('')}
              </div>
            `}
        </div>
      </div>
    `;

    // Event listeners
    const btnSync = this.container.querySelector('#btn-manual-sync');
    if (btnSync) btnSync.addEventListener('click', () => this.handleManualSync());

    const btnBoot = this.container.querySelector('#btn-bootstrap');
    if (btnBoot) btnBoot.addEventListener('click', () => this.handleBootstrap());

    const chkLowData = this.container.querySelector('#chk-low-data');
    if (chkLowData) {
      chkLowData.addEventListener('change', (e) => this.toggleLowDataMode(e.target.checked));
    }
  }
}
