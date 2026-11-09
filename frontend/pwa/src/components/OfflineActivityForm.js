// AgriGuardian Offline Farm Activity Logger
// Records farm operations locally with durable journal queuing for seamless background sync

import { dbStorage } from '../storage/indexed_db.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';

export class OfflineActivityForm {
  constructor(containerId = 'activity-form-root') {
    this.container = document.getElementById(containerId);
    this.farms = [];
    this.crops = [];
  }

  async refresh() {
    this.farms = await dbStorage.getAll('farms');
    this.crops = await dbStorage.getAll('crop_cycles');
    this.render();
  }

  async handleSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const title = form.elements['title'].value.trim();
    const type = form.elements['activity_type'].value;
    const farmId = form.elements['farm_id'].value;
    const date = form.elements['scheduled_date'].value;
    const isCompleted = form.elements['is_completed'].checked;
    const cost = parseFloat(form.elements['cost'].value || '0');
    const notes = form.elements['notes'].value.trim();

    if (!title) {
      alert('Please enter an activity title.');
      return;
    }

    const clientActivityId = 'act_' + (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 10));
    const nowStr = new Date().toISOString();

    const activityRecord = {
      activity_id: clientActivityId,
      farm_id: farmId || 'farm_default',
      farmer_id: (await dbStorage.getMeta('farmer_id')) || 'fmr_default',
      activity_type: type,
      activity_title: title,
      description: notes,
      scheduled_date: date || nowStr.substring(0, 10),
      completed_date: isCompleted ? (date || nowStr.substring(0, 10)) : null,
      status: isCompleted ? 'completed' : 'planned',
      cost_incurred: cost,
      notes: notes,
      created_at: nowStr,
      updated_at: nowStr,
      version_number: 1,
      _is_local_draft: true
    };

    // Save locally
    await dbStorage.put('activities', activityRecord);

    // Enqueue mutation
    await mutationQueue.enqueue('farm_activity', 'CREATE', clientActivityId, activityRecord, 0);

    window.dispatchEvent(new CustomEvent('mutation-enqueued'));
    alert('Activity saved locally! It is queued for background sync.');
    window.dispatchEvent(new CustomEvent('navigate-tab', { detail: 'dashboard' }));
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';
    const today = new Date().toISOString().substring(0, 10);

    this.container.innerHTML = `
      <div class="form-card">
        <div class="form-header">
          <h2>🚜 ${lang === 'hi' ? 'नया कृषि कार्य दर्ज करें' : 'Record Farm Activity'}</h2>
          <span class="pill-badge pill-pending_sync">⏳ ${lang === 'hi' ? 'ऑफ़लाइन सुरक्षित' : 'Offline Safe'}</span>
        </div>

        <form id="activity-form">
          <div class="form-group">
            <label for="act-title"><strong>${lang === 'hi' ? 'कार्य का नाम / शीर्षक' : 'Activity Title'} *</strong></label>
            <input type="text" id="act-title" name="title" class="form-control" placeholder="${lang === 'hi' ? 'उदा. यूरिया खाद का छिड़काव' : 'e.g. Urea Top Dressing'}" required>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="act-type"><strong>${lang === 'hi' ? 'कार्य प्रकार' : 'Activity Type'}</strong></label>
              <select id="act-type" name="activity_type" class="form-control">
                <option value="fertilizer_application">🌿 ${lang === 'hi' ? 'उर्वरक / खाद' : 'Fertilizer Application'}</option>
                <option value="irrigation">💧 ${lang === 'hi' ? 'सिंचाई' : 'Irrigation'}</option>
                <option value="pest_spray">🌧️ ${lang === 'hi' ? 'कीटनाशक छिड़काव' : 'Pest Spray'}</option>
                <option value="weeding">🌱 ${lang === 'hi' ? 'निराई-गुड़ाई' : 'Weeding'}</option>
                <option value="sowing">🚜 ${lang === 'hi' ? 'बुवाई / रोपाई' : 'Sowing / Transplanting'}</option>
                <option value="harvesting">🌾 ${lang === 'hi' ? 'कटाई' : 'Harvesting'}</option>
                <option value="scouting">🔍 ${lang === 'hi' ? 'खेत निरीक्षण / अवलोकन' : 'Field Scouting'}</option>
                <option value="general_field_work">📝 ${lang === 'hi' ? 'अन्य सामान्य कार्य' : 'General Field Work'}</option>
              </select>
            </div>

            <div class="form-group">
              <label for="act-date"><strong>${lang === 'hi' ? 'दिनांक' : 'Date'}</strong></label>
              <input type="date" id="act-date" name="scheduled_date" class="form-control" value="${today}">
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="act-cost"><strong>${lang === 'hi' ? 'खर्च (₹)' : 'Cost Incurred (₹)'}</strong></label>
              <input type="number" id="act-cost" name="cost" class="form-control" placeholder="0" min="0" step="50">
            </div>

            <div class="form-group checkbox-group">
              <label class="toggle-label">
                <input type="checkbox" name="is_completed" checked>
                <span><strong>✅ ${lang === 'hi' ? 'कार्य आज ही पूर्ण हुआ' : 'Mark as Completed'}</strong></span>
              </label>
            </div>
          </div>

          <div class="form-group">
            <label for="act-notes"><strong>${lang === 'hi' ? 'टिप्पणी / खेत अवलोकन' : 'Field Notes & Observations'}</strong></label>
            <textarea id="act-notes" name="notes" class="form-control" rows="3" placeholder="${lang === 'hi' ? 'मात्रा, फसल स्थिति या कोई अन्य जानकारी...' : 'Dosage, crop health, soil moisture...'}"></textarea>
          </div>

          <div class="form-actions">
            <button type="submit" class="btn-primary">
              💾 ${lang === 'hi' ? 'सुरक्षित करें (ऑफ़लाइन)' : 'Save Activity (Offline Safe)'}
            </button>
          </div>
        </form>
      </div>
    `;

    const form = this.container.querySelector('#activity-form');
    if (form) form.addEventListener('submit', (e) => this.handleSubmit(e));
  }
}
