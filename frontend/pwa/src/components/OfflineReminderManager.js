// AgriGuardian Offline Reminder Manager Component
// Create and manage farm reminders with weather risk warnings even without internet connectivity

import { dbStorage } from '../storage/indexed_db.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';

export class OfflineReminderManager {
  constructor(containerId = 'reminder-manager-root') {
    this.container = document.getElementById(containerId);
    this.reminders = [];
  }

  async refresh() {
    this.reminders = await dbStorage.getAll('reminders');
    this.render();
  }

  async handleSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const titleEn = form.elements['title_en'].value.trim();
    const titleHi = form.elements['title_hi'].value.trim() || titleEn;
    const actType = form.elements['activity_type'].value;
    const date = form.elements['scheduled_date'].value;
    const time = form.elements['due_time'].value || '08:00';
    const desc = form.elements['description'].value.trim();

    if (!titleEn) {
      alert('Please enter a reminder title.');
      return;
    }

    const clientReminderId = 'rem_' + (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 10));
    const nowStr = new Date().toISOString();

    const reminderRecord = {
      reminder_id: clientReminderId,
      farmer_id: (await dbStorage.getMeta('farmer_id')) || 'fmr_default',
      activity_type: actType,
      title_en: titleEn,
      title_hi: titleHi,
      description_en: desc,
      description_hi: desc,
      scheduled_date: date,
      due_time: time,
      status: 'scheduled',
      weather_risk_flag: 0,
      version_number: 1,
      created_at: nowStr,
      updated_at: nowStr,
      _is_local_draft: true
    };

    await dbStorage.put('reminders', reminderRecord);
    await mutationQueue.enqueue('farm_reminder', 'CREATE', clientReminderId, reminderRecord, 0);

    window.dispatchEvent(new CustomEvent('mutation-enqueued'));
    alert('Reminder created locally! Queued for synchronization.');
    form.reset();
    this.refresh();
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';
    const tomorrow = new Date(Date.now() + 86400000).toISOString().substring(0, 10);

    this.container.innerHTML = `
      <div class="reminder-manager-view">
        <div class="form-card">
          <div class="form-header">
            <h2>⏰ ${lang === 'hi' ? 'नया रिमाइंडर जोड़ें' : 'Set Smart Farm Reminder'}</h2>
          </div>
          <form id="reminder-create-form">
            <div class="form-row">
              <div class="form-group">
                <label><strong>${lang === 'hi' ? 'शीर्षक (अंग्रेजी)' : 'Title (English)'} *</strong></label>
                <input type="text" name="title_en" class="form-control" placeholder="e.g. Zinc Spray at Flowering" required>
              </div>
              <div class="form-group">
                <label><strong>${lang === 'hi' ? 'शीर्षक (हिंदी)' : 'Title (Hindi)'}</strong></label>
                <input type="text" name="title_hi" class="form-control" placeholder="उदा. फूल आने पर जिंक का छिड़काव">
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label><strong>${lang === 'hi' ? 'कार्य प्रकार' : 'Activity Type'}</strong></label>
                <select name="activity_type" class="form-control">
                  <option value="pest_spray">🌧️ ${lang === 'hi' ? 'कीटनाशक / फफूंदनाशक छिड़काव' : 'Pest / Fungicide Spray'}</option>
                  <option value="fertilization">🌿 ${lang === 'hi' ? 'उर्वरक / पोषण' : 'Fertilization'}</option>
                  <option value="irrigation">💧 ${lang === 'hi' ? 'सिंचाई' : 'Irrigation'}</option>
                  <option value="harvesting">🌾 ${lang === 'hi' ? 'कटाई' : 'Harvesting'}</option>
                  <option value="other">📝 ${lang === 'hi' ? 'अन्य' : 'Other'}</option>
                </select>
              </div>

              <div class="form-group">
                <label><strong>${lang === 'hi' ? 'दिनांक' : 'Date'}</strong></label>
                <input type="date" name="scheduled_date" class="form-control" value="${tomorrow}" required>
              </div>

              <div class="form-group">
                <label><strong>${lang === 'hi' ? 'समय' : 'Time'}</strong></label>
                <input type="time" name="due_time" class="form-control" value="08:00">
              </div>
            </div>

            <div class="form-group">
              <label><strong>${lang === 'hi' ? 'विवरण / निर्देश' : 'Instructions'}</strong></label>
              <textarea name="description" class="form-control" rows="2" placeholder="${lang === 'hi' ? 'खुराक, सावधानी आदि...' : 'Dosage, safety warnings...'}"></textarea>
            </div>

            <div class="form-actions">
              <button type="submit" class="btn-primary">
                🔔 ${lang === 'hi' ? 'रिमाइंडर सुरक्षित करें' : 'Save Reminder (Offline Safe)'}
              </button>
            </div>
          </form>
        </div>

        <div class="dashboard-section" style="margin-top: 24px;">
          <h3>📋 ${lang === 'hi' ? 'सक्रिय और निर्धारित रिमाइंडर' : 'All Scheduled Reminders'} (${this.reminders.length})</h3>
          <div class="cards-list">
            ${this.reminders.length === 0
              ? `<div class="empty-card">🔔 ${lang === 'hi' ? 'कोई रिमाइंडर नहीं।' : 'No reminders found.'}</div>`
              : this.reminders.map((r) => `
                <div class="item-card reminder-card ${r.weather_risk_flag ? 'card-hazard' : ''}">
                  <div class="card-details">
                    <strong class="card-heading">${lang === 'hi' ? r.title_hi || r.title_en : r.title_en}</strong>
                    <span class="card-sub">📅 ${r.scheduled_date} ${r.due_time || ''} • 🏷️ ${r.activity_type}</span>
                    ${r.description_en ? `<p class="card-notes">${r.description_en}</p>` : ''}
                    ${r.weather_risk_flag ? `
                      <div class="weather-hazard-banner">
                        ⚠️ ${lang === 'hi' ? r.weather_risk_reason_hi || r.weather_risk_reason_en : r.weather_risk_reason_en}
                      </div>
                    ` : ''}
                  </div>
                  <div class="card-actions">
                    <span class="pill-badge pill-${r.status}">${r.status}</span>
                  </div>
                </div>
              `).join('')}
          </div>
        </div>
      </div>
    `;

    const form = this.container.querySelector('#reminder-create-form');
    if (form) form.addEventListener('submit', (e) => this.handleSubmit(e));
  }
}
