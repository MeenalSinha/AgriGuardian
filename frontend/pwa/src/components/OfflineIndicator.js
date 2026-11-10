// AgriGuardian Offline Indicator Component
// Truthfully communicates connectivity state and exact data freshness timestamp without deception

import { networkManager } from '../sync/offline_network_manager.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';
import { dbStorage } from '../storage/indexed_db.js';
import { NetworkQuality } from '../types.js';

export class OfflineIndicator {
  constructor(containerId = 'offline-indicator-root') {
    this.container = document.getElementById(containerId);
    this.status = networkManager.status;
    this.pendingCount = 0;
    this.lastSync = null;

    networkManager.subscribe((st) => {
      this.status = st;
      this.render();
    });

    this.refreshState();
  }

  async refreshState() {
    this.pendingCount = await mutationQueue.getPendingCount();
    this.lastSync = await dbStorage.getMeta('last_sync_timestamp');
    this.render();
  }

  formatTimestamp(isoString) {
    if (!isoString) return 'Never synchronized';
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return isoString;
    }
  }

  render() {
    if (!this.container) return;

    const isOffline = this.status === NetworkQuality.OFFLINE;
    const isWeak = this.status === NetworkQuality.WEAK;

    let bannerClass = 'banner-online';
    let statusText = 'Online (Connected)';
    let statusTextHi = 'ऑनलाइन (सर्वर से जुड़ा हुआ)';

    if (isOffline) {
      bannerClass = 'banner-offline';
      statusText = 'Offline Mode — Local Cache Active';
      statusTextHi = 'ऑफ़लाइन मोड — स्थानीय डेटा सक्रिय';
    } else if (isWeak) {
      bannerClass = 'banner-weak';
      statusText = 'Weak Connection — Requests May Delay';
      statusTextHi = 'कमजोर नेटवर्क — अनुरोध में समय लग सकता है';
    }

    const lang = localStorage.getItem('agri_lang') || 'en';
    const activeStatusText = lang === 'hi' ? statusTextHi : statusText;
    const lastSyncFormatted = this.formatTimestamp(this.lastSync);
    const offlineSuffix = isOffline ? ' (Offline)' : '';

    this.container.innerHTML = `
      <div class="connectivity-banner ${bannerClass}">
        <div class="banner-main">
          <span class="banner-status-badge">${activeStatusText}</span>
          <span class="banner-sync-time">
            ${lang === 'hi' ? 'अंतिम अपडेट' : 'Last updated'}: <strong>${lastSyncFormatted}${offlineSuffix}</strong>
          </span>
        </div>
        ${this.pendingCount > 0 ? `
          <button class="banner-pending-pill" id="btn-open-sync-center">
            ${this.pendingCount} ${lang === 'hi' ? 'बदलाव सिंक होना बाकी' : 'change(s) pending sync'}
          </button>
        ` : ''}
      </div>
    `;

    const btn = this.container.querySelector('#btn-open-sync-center');
    if (btn) {
      btn.addEventListener('click', () => {
        const event = new CustomEvent('navigate-tab', { detail: 'sync' });
        window.dispatchEvent(event);
      });
    }
  }
}
