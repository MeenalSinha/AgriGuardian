// AgriGuardian Offline Crop Disease Photograph & Symptom Capture Component
// Captures high-resolution leaf photos in the field, preserves binary blobs in IndexedDB, and queues draft for AI diagnosis

import { dbStorage } from '../storage/indexed_db.js';
import { mutationQueue } from '../sync/offline_mutation_queue.js';

export class OfflineLeafCapture {
  constructor(containerId = 'leaf-capture-root') {
    this.container = document.getElementById(containerId);
    this.selectedFile = null;
    this.previewUrl = null;
    this.crops = [];
  }

  async refresh() {
    this.crops = await dbStorage.getAll('crop_cycles');
    this.render();
  }

  async handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    this.selectedFile = file;
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    this.previewUrl = URL.createObjectURL(file);

    const imgPreview = this.container.querySelector('#leaf-preview-img');
    const previewContainer = this.container.querySelector('#preview-box');
    if (imgPreview && previewContainer) {
      imgPreview.src = this.previewUrl;
      previewContainer.style.display = 'block';
    }
  }

  async handleSubmit(e) {
    e.preventDefault();
    if (!this.selectedFile) {
      alert('Please select or capture a crop leaf photograph.');
      return;
    }

    const form = e.target;
    const cropName = form.elements['crop_name'].value;
    const symptoms = form.elements['symptoms'].value.trim();
    const severity = form.elements['severity'].value;

    const attachmentId = 'att_' + (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 10));
    const draftCaseId = 'diag_draft_' + Math.random().toString(36).substring(2, 10);
    const nowStr = new Date().toISOString();

    // 1. Store attachment blob in IndexedDB
    const attachmentRecord = {
      attachment_id: attachmentId,
      farmer_id: (await dbStorage.getMeta('farmer_id')) || 'fmr_default',
      resource_type: 'crop_diagnosis',
      resource_id: draftCaseId,
      file_name: this.selectedFile.name || 'leaf_capture.jpg',
      mime_type: this.selectedFile.type || 'image/jpeg',
      byte_size: this.selectedFile.size,
      blob: this.selectedFile,
      is_uploaded: false,
      created_at: nowStr
    };
    await dbStorage.put('attachments', attachmentRecord);

    // 2. Store advisory draft in IndexedDB
    const draftCase = {
      case_id: draftCaseId,
      crop_name: cropName,
      symptoms: symptoms,
      severity: severity,
      attachment_id: attachmentId,
      status: 'draft_pending_sync',
      created_at: nowStr,
      _is_local_draft: true
    };
    await dbStorage.put('advisories', draftCase);

    // 3. Queue mutation
    await mutationQueue.enqueue('crop_diagnosis_draft', 'CREATE', draftCaseId, {
      case_id: draftCaseId,
      crop_name: cropName,
      symptoms: symptoms,
      severity: severity,
      attachment_id: attachmentId
    }, 0);

    window.dispatchEvent(new CustomEvent('mutation-enqueued'));
    alert('Leaf photograph & symptoms saved locally as draft! When internet connectivity returns, it will upload automatically to the AI diagnostic engine.');
    window.dispatchEvent(new CustomEvent('navigate-tab', { detail: 'dashboard' }));
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';

    this.container.innerHTML = `
      <div class="form-card">
        <div class="form-header">
          <h2>${lang === 'hi' ? 'फसल रोग निदान एवं लक्षण फोटो' : 'Crop Disease Photo & Diagnosis'}</h2>
          <span class="pill-badge pill-pending_sync">${lang === 'hi' ? 'ऑफ़लाइन ड्राफ्ट' : 'Offline Draft'}</span>
        </div>

        <p class="form-intro">
          ${lang === 'hi'
            ? 'खेत में बिना इंटरनेट के भी प्रभावित पत्ती या पौधे की तस्वीर खींचें। इंटरनेट मिलने पर AI मॉडल रोग की पहचान कर दवा की सिफारिश करेगा।'
            : 'Capture leaf damage or pest symptoms in the field even without internet. When connection returns, AgriGuardian will run AI disease detection and agronomist verification.'}
        </p>

        <form id="leaf-capture-form">
          <div class="form-group">
            <label><strong>${lang === 'hi' ? 'फसल चुनें' : 'Select Crop'} *</strong></label>
            <select name="crop_name" class="form-control" required>
              <option value="Wheat">${lang === 'hi' ? 'गेहूं (Wheat)' : 'Wheat'}</option>
              <option value="Rice">${lang === 'hi' ? 'धान / चावल (Rice)' : 'Rice'}</option>
              <option value="Tomato">${lang === 'hi' ? 'टमाटर (Tomato)' : 'Tomato'}</option>
              <option value="Cotton">${lang === 'hi' ? 'कपास (Cotton)' : 'Cotton'}</option>
              <option value="Corn">${lang === 'hi' ? 'मक्का (Corn / Maize)' : 'Corn / Maize'}</option>
              <option value="Potato">${lang === 'hi' ? 'आलू (Potato)' : 'Potato'}</option>
              <option value="Soybean">${lang === 'hi' ? 'सोयाबीन (Soybean)' : 'Soybean'}</option>
            </select>
          </div>

          <div class="form-group">
            <label><strong>${lang === 'hi' ? 'पत्ती / पौधे की तस्वीर लें' : 'Leaf Photograph'} *</strong></label>
            <div class="file-picker-box">
              <input type="file" id="leaf-file-input" accept="image/*" capture="environment" style="display: none;">
              <button type="button" class="btn-secondary" id="btn-trigger-camera">
                ${lang === 'hi' ? 'कैमरा खोलें / फोटो चुनें' : 'Open Camera / Choose Photo'}
              </button>
            </div>
            <div id="preview-box" class="image-preview-container" style="display: none; margin-top: 12px;">
              <img id="leaf-preview-img" class="leaf-preview" alt="Leaf Preview">
            </div>
          </div>

          <div class="form-group">
            <label><strong>${lang === 'hi' ? 'प्रकट लक्षण (दाग, पीलापन, मुड़ना आदि)' : 'Observed Symptoms'}</strong></label>
            <textarea name="symptoms" class="form-control" rows="3" placeholder="${lang === 'hi' ? 'उदा. पत्तियों के नीचे काले धब्बे और किनारों का पीला पड़ना...' : 'e.g. Yellowing margins, brown spots on underside, curling leaves...'}"></textarea>
          </div>

          <div class="form-group">
            <label><strong>${lang === 'hi' ? 'प्रभावित क्षेत्र का स्तर' : 'Severity Assessment'}</strong></label>
            <select name="severity" class="form-control">
              <option value="mild">${lang === 'hi' ? 'हल्का / शुरुआती (1-5% पत्तियां)' : 'Mild / Early (1-5% leaves)'}</option>
              <option value="moderate" selected>${lang === 'hi' ? 'मध्यम (5-25% पत्तियां)' : 'Moderate (5-25% leaves)'}</option>
              <option value="severe">${lang === 'hi' ? 'गंभीर / फैला हुआ (>25% पत्तियां)' : 'Severe (>25% crop affected)'}</option>
            </select>
          </div>

          <div class="form-actions">
            <button type="submit" class="btn-primary">
              ${lang === 'hi' ? 'निदान ड्राफ्ट सहेजें (ऑफ़लाइन)' : 'Save Diagnosis Draft (Offline)'}
            </button>
          </div>
        </form>
      </div>
    `;

    const fileInput = this.container.querySelector('#leaf-file-input');
    const btnCam = this.container.querySelector('#btn-trigger-camera');
    if (btnCam && fileInput) {
      btnCam.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', (e) => this.handleFileChange(e));
    }

    const form = this.container.querySelector('#leaf-capture-form');
    if (form) form.addEventListener('submit', (e) => this.handleSubmit(e));
  }
}
