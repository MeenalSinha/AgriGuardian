// AgriGuardian PWA Main Application Coordinator
// Orchestrates offline storage, components, navigation, and service worker registration

import { dbStorage } from './storage/indexed_db.js';
import { networkManager } from './sync/offline_network_manager.js';
import { syncManager } from './sync/offline_sync_manager.js';
import { OfflineIndicator } from './components/OfflineIndicator.js';
import { OfflineFarmerDashboard } from './components/OfflineFarmerDashboard.js';
import { OfflineActivityForm } from './components/OfflineActivityForm.js';
import { OfflineReminderManager } from './components/OfflineReminderManager.js';
import { OfflineLeafCapture } from './components/OfflineLeafCapture.js';
import { SyncCenter } from './components/SyncCenter.js';
import { ConflictResolver } from './components/ConflictResolver.js';

class AgriGuardianApp {
  constructor() {
    this.activeTab = 'dashboard';
    this.currentLanguage = localStorage.getItem('agri_lang') || 'en';

    // Components
    this.indicator = new OfflineIndicator('offline-indicator-root');
    this.dashboard = new OfflineFarmerDashboard('farmer-dashboard-root');
    this.activityForm = new OfflineActivityForm('activity-form-root');
    this.reminderManager = new OfflineReminderManager('reminder-manager-root');
    this.leafCapture = new OfflineLeafCapture('leaf-capture-root');
    this.syncCenter = new SyncCenter('sync-center-root');
    this.conflictResolver = new ConflictResolver('conflict-resolver-root');

    this.init();
  }

  async init() {
    console.log('[App] Initializing AgriGuardian Offline-First PWA...');

    // 1. Register Service Worker
    this.registerServiceWorker();

    // 2. Setup Navigation
    this.setupNavigation();

    // 3. Setup Language Switcher
    this.setupLanguage();

    // 4. Initial Component Refresh
    await this.refreshActiveView();

    // 5. Listen for cross-component events
    window.addEventListener('navigate-tab', (e) => this.switchTab(e.detail));
    window.addEventListener('mutation-enqueued', () => this.indicator.refreshState());

    // 6. Ensure device ID
    await syncManager.ensureDeviceRegistered();
  }

  registerServiceWorker() {
    if ('serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/pwa/sw.js')
          .then((reg) => console.log('[ServiceWorker] Registered with scope:', reg.scope))
          .catch((err) => console.warn('[ServiceWorker] Registration failed:', err));
      });
    }
  }

  setupLanguage() {
    const btnLang = document.getElementById('btn-lang-toggle');
    if (btnLang) {
      btnLang.textContent = this.currentLanguage === 'hi' ? 'English' : 'हिंदी';
      btnLang.addEventListener('click', () => {
        this.currentLanguage = this.currentLanguage === 'hi' ? 'en' : 'hi';
        localStorage.setItem('agri_lang', this.currentLanguage);
        btnLang.textContent = this.currentLanguage === 'hi' ? 'English' : 'हिंदी';
        this.refreshActiveView();
        this.indicator.render();
      });
    }
  }

  setupNavigation() {
    const tabs = document.querySelectorAll('[data-tab]');
    tabs.forEach((t) => {
      t.addEventListener('click', () => {
        const target = t.getAttribute('data-tab');
        this.switchTab(target);
      });
    });
  }

  switchTab(tabName) {
    this.activeTab = tabName;

    // Update bottom nav active state
    document.querySelectorAll('[data-tab]').forEach((t) => {
      if (t.getAttribute('data-tab') === tabName) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });

    // Hide all view panes
    const panes = ['dashboard', 'activity', 'diagnosis', 'reminder', 'sync'];
    panes.forEach((p) => {
      const el = document.getElementById(`pane-${p}`);
      if (el) el.style.display = p === tabName ? 'block' : 'none';
    });

    this.refreshActiveView();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  async refreshActiveView() {
    switch (this.activeTab) {
      case 'dashboard':
        await this.dashboard.refresh();
        break;
      case 'activity':
        await this.activityForm.refresh();
        break;
      case 'diagnosis':
        await this.leafCapture.refresh();
        break;
      case 'reminder':
        await this.reminderManager.refresh();
        break;
      case 'sync':
        await this.syncCenter.refresh();
        await this.conflictResolver.refresh();
        break;
    }
    await this.indicator.refreshState();
  }
}

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.agriApp = new AgriGuardianApp();
});
