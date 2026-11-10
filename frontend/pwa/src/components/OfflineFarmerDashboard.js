// AgriGuardian Mobile Offline Farmer Dashboard
// Presents cached farm profiles, active crops, crop calendar, reminders, and recent advisories

import { dbStorage } from '../storage/indexed_db.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';
import { networkManager } from '../sync/offline_network_manager.js';
import { NetworkQuality } from '../types.js';

export class OfflineFarmerDashboard {
  constructor(containerId = 'farmer-dashboard-root') {
    this.container = document.getElementById(containerId);
    this.farmerProfile = null;
    this.farms = [];
    this.crops = [];
    this.activities = [];
    this.reminders = [];
    this.advisories = [];
    this.lastSync = null;
  }

  async refresh() {
    this.farmerProfile = (await dbStorage.getMeta('farmer_profile')) || {
      full_name: 'Shri Rameshwar Patel',
      village: 'Kailashpur',
      district: 'Sehore',
      state: 'Madhya Pradesh'
    };
    this.farms = await dbStorage.getAll('farms');
    this.crops = await dbStorage.getAll('crop_cycles');
    this.activities = await dbStorage.getAll('activities');
    this.reminders = await dbStorage.getAll('reminders');
    this.advisories = await dbStorage.getAll('advisories');
    this.lastSync = await dbStorage.getMeta('last_sync_timestamp');
    this.render();
  }

  async completeActivity(activityId) {
    const act = await dbStorage.get('activities', activityId);
    if (!act) return;

    act.status = 'completed';
    act.completed_date = new Date().toISOString().substring(0, 10);
    act.updated_at = new Date().toISOString();
    await dbStorage.put('activities', act);

    // Enqueue mutation
    await mutationQueue.enqueue('farm_activity', 'COMPLETE', activityId, {
      activity_id: activityId,
      status: 'completed',
      completed_date: act.completed_date
    });

    // Notify indicator and refresh
    window.dispatchEvent(new CustomEvent('mutation-enqueued'));
    this.refresh();
  }

  async completeReminder(reminderId) {
    const rem = await dbStorage.get('reminders', reminderId);
    if (!rem) return;

    rem.status = 'completed';
    rem.completed_at = new Date().toISOString();
    rem.updated_at = new Date().toISOString();
    await dbStorage.put('reminders', rem);

    await mutationQueue.enqueue('farm_reminder', 'COMPLETE', reminderId, {
      reminder_id: reminderId,
      status: 'completed'
    }, rem.version_number || 1);

    window.dispatchEvent(new CustomEvent('mutation-enqueued'));
    this.refresh();
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';
    const isOffline = networkManager.status === NetworkQuality.OFFLINE;

    this.container.innerHTML = `
      <div class="farmer-dashboard-view">
        <!-- Farmer Identity Banner -->
        <div class="farmer-profile-card">
          <div class="farmer-avatar">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2d6a4f" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          </div>
          <div class="farmer-meta">
            <h2>${this.farmerProfile.full_name || 'Farmer'}</h2>
            <p>${this.farmerProfile.village ? this.farmerProfile.village + ', ' : ''}${this.farmerProfile.district || 'District'}, ${this.farmerProfile.state || 'India'}</p>
          </div>
        </div>

        <!-- Quick Action Shortcuts -->
        <div class="quick-actions-bar">
          <button class="action-btn" id="btn-quick-activity">
            ${lang === 'hi' ? 'कार्य दर्ज करें' : 'Log Activity'}
          </button>
          <button class="action-btn" id="btn-quick-photo">
            ${lang === 'hi' ? 'फसल जांच' : 'Crop Diagnosis'}
          </button>
          <button class="action-btn" id="btn-quick-reminder">
            ${lang === 'hi' ? 'रिमाइंडर जोड़ें' : 'Add Reminder'}
          </button>
        </div>

        <!-- Today's / Upcoming Farm Activities -->
        <div class="dashboard-section">
          <div class="section-title-row">
            <h3>${lang === 'hi' ? 'आज और आगामी कृषि कार्य' : 'Scheduled Farm Activities'} (${this.activities.length})</h3>
          </div>
          ${this.activities.length === 0
            ? `<div class="empty-card">${lang === 'hi' ? 'कोई निर्धारित कार्य नहीं है।' : 'No activities scheduled.'}</div>`
            : `
              <div class="cards-list">
                ${this.activities.slice(0, 5).map((a) => `
                  <div class="item-card activity-card ${a.status === 'completed' ? 'item-completed' : ''}">
                    <div class="card-details">
                      <strong class="card-heading">${a.activity_title}</strong>
                      <span class="card-sub">${a.scheduled_date} &bull; ${a.activity_type}</span>
                      ${a.notes ? `<p class="card-notes">${a.notes}</p>` : ''}
                    </div>
                    <div class="card-actions">
                      ${a.status !== 'completed' ? `
                        <button class="btn-check" data-act-id="${a.activity_id}">
                          ${lang === 'hi' ? 'पूर्ण' : 'Done'}
                        </button>
                      ` : `
                        <span class="pill-badge pill-synced">${lang === 'hi' ? 'पूर्ण' : 'Completed'}</span>
                      `}
                    </div>
                  </div>
                `).join('')}
              </div>
            `}
        </div>

        <!-- Smart Reminders & Weather Risk Warnings -->
        <div class="dashboard-section">
          <div class="section-title-row">
            <h3>${lang === 'hi' ? 'स्मार्ट रिमाइंडर और मौसम चेतावनियां' : 'Smart Reminders & Weather Alerts'} (${this.reminders.length})</h3>
          </div>
          ${this.reminders.length === 0
            ? `<div class="empty-card">${lang === 'hi' ? 'कोई सक्रिय रिमाइंडर नहीं।' : 'No reminders active.'}</div>`
            : `
              <div class="cards-list">
                ${this.reminders.slice(0, 5).map((r) => `
                  <div class="item-card reminder-card ${r.weather_risk_flag ? 'card-hazard' : ''} ${r.status === 'completed' ? 'item-completed' : ''}">
                    <div class="card-details">
                      <strong class="card-heading">${lang === 'hi' ? r.title_hi || r.title_en : r.title_en}</strong>
                      <span class="card-sub">${r.scheduled_date} ${r.due_time || ''}</span>
                      ${r.weather_risk_flag ? `
                        <div class="weather-hazard-banner">
                          ${lang === 'hi' ? r.weather_risk_reason_hi || r.weather_risk_reason_en : r.weather_risk_reason_en}
                        </div>
                      ` : ''}
                    </div>
                    <div class="card-actions">
                      ${r.status !== 'completed' ? `
                        <button class="btn-check" data-rem-id="${r.reminder_id}">
                          ${lang === 'hi' ? 'संपन्न' : 'Done'}
                        </button>
                      ` : `
                        <span class="pill-badge pill-synced">${lang === 'hi' ? 'संपन्न' : 'Completed'}</span>
                      `}
                    </div>
                  </div>
                `).join('')}
              </div>
            `}
        </div>

        <!-- My Crops Summary -->
        <div class="dashboard-section">
          <div class="section-title-row">
            <h3>${lang === 'hi' ? 'सक्रिय फसलें' : 'Active Crops'} (${this.crops.length})</h3>
          </div>
          ${this.crops.length === 0
            ? `<div class="empty-card">${lang === 'hi' ? 'कोई फसल पंजीकृत नहीं।' : 'No active crop registered.'}</div>`
            : `
              <div class="crops-grid">
                ${this.crops.map((c) => `
                  <div class="crop-tile">
                    <h4>${c.crop_name}</h4>
                    <p class="crop-stage">${c.current_stage || 'Vegetative'}</p>
                    <span class="crop-season">${c.season ? c.season.toUpperCase() : 'KHARIF'} &bull; ${c.sowing_date || ''}</span>
                  </div>
                `).join('')}
              </div>
            `}
        </div>
      </div>
    `;

    // Hook up buttons
    this.container.querySelectorAll('[data-act-id]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-act-id');
        this.completeActivity(id);
      });
    });

    this.container.querySelectorAll('[data-rem-id]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const id = e.currentTarget.getAttribute('data-rem-id');
        this.completeReminder(id);
      });
    });

    // Quick action triggers
    const actBtn = this.container.querySelector('#btn-quick-activity');
    if (actBtn) {
      actBtn.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('navigate-tab', { detail: 'activity' }));
      });
    }

    const photoBtn = this.container.querySelector('#btn-quick-photo');
    if (photoBtn) {
      photoBtn.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('navigate-tab', { detail: 'diagnosis' }));
      });
    }

    const remBtn = this.container.querySelector('#btn-quick-reminder');
    if (remBtn) {
      remBtn.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('navigate-tab', { detail: 'reminder' }));
      });
    }
  }
}
