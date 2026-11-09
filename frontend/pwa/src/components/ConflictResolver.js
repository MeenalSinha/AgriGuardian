// AgriGuardian Conflict Resolver Dialog Component
// Provides side-by-side difference inspection and safe resolution options without silent data loss

import { conflictManager } from '../sync/offline_conflict_manager.js';
import { dbStorage } from '../storage/indexed_db.js';

export class ConflictResolver {
  constructor(containerId = 'conflict-resolver-root') {
    this.container = document.getElementById(containerId);
    this.conflicts = [];
  }

  async refresh() {
    try {
      const res = await fetch('/api/v1/sync/conflicts?status=pending');
      if (res.ok) {
        this.conflicts = await res.json();
      } else {
        this.conflicts = [];
      }
    } catch {
      this.conflicts = [];
    }
    this.render();
  }

  async resolveConflict(conflictId, strategy) {
    try {
      const res = await fetch(`/api/v1/sync/conflicts/${encodeURIComponent(conflictId)}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution_strategy: strategy })
      });
      if (res.ok) {
        alert('Conflict resolved successfully!');
        this.refresh();
      } else {
        alert('Could not resolve conflict online. Please check network.');
      }
    } catch (e) {
      alert(`Error resolving conflict: ${e.message}`);
    }
  }

  render() {
    if (!this.container) return;

    const lang = localStorage.getItem('agri_lang') || 'en';

    if (this.conflicts.length === 0) {
      this.container.innerHTML = `
        <div class="empty-sync-box" style="margin: 16px 0;">
          🛡️ ${lang === 'hi' ? 'कोई सक्रिय डेटा विरोध नहीं है। सभी रिकॉर्ड संरेखित हैं।' : 'No active conflicts detected. All records are aligned.'}
        </div>
      `;
      return;
    }

    this.container.innerHTML = `
      <div class="conflicts-panel">
        <h3>⚠️ ${lang === 'hi' ? 'डेटा विरोध समाधान' : 'Resolve Conflicting Changes'} (${this.conflicts.length})</h3>
        <p class="conflict-intro">
          ${lang === 'hi'
            ? 'नीचे दिए गए रिकॉर्ड में आपके ऑफ़लाइन बदलाव और सर्वर पर मौजूद डेटा में अंतर है। कृपया अपनी पसंद चुनें:'
            : 'The items below were edited offline while server data also changed. Please choose which version to preserve:'}
        </p>

        <div class="conflict-cards-list">
          ${this.conflicts.map((c) => {
            const diffs = conflictManager.computeDiff(c.client_payload, c.server_payload);
            return `
              <div class="conflict-item-card">
                <div class="conflict-item-header">
                  <strong>🏷️ ${c.resource_type.replace('_', ' ').toUpperCase()}</strong>
                  <span class="pill-badge pill-conflict">Conflict</span>
                </div>

                <div class="diff-table-container">
                  <table class="diff-table">
                    <thead>
                      <tr>
                        <th>${lang === 'hi' ? 'फ़ील्ड' : 'Field'}</th>
                        <th>📱 ${lang === 'hi' ? 'आपका ऑफ़लाइन संस्करण' : 'Your Offline Version'}</th>
                        <th>☁️ ${lang === 'hi' ? 'सर्वर संस्करण' : 'Server Version'}</th>
                      </tr>
                    </thead>
                    <tbody>
                      ${diffs.map((d) => `
                        <tr>
                          <td><strong>${d.field}</strong></td>
                          <td class="diff-client">${JSON.stringify(d.clientValue)}</td>
                          <td class="diff-server">${JSON.stringify(d.serverValue)}</td>
                        </tr>
                      `).join('')}
                    </tbody>
                  </table>
                </div>

                <div class="conflict-actions-row">
                  <button class="btn-primary" data-resolve-id="${c.conflict_id}" data-strategy="client_wins">
                    📱 ${lang === 'hi' ? 'मेरा ऑफ़लाइन संस्करण रखें' : 'Keep My Offline Version'}
                  </button>
                  <button class="btn-secondary" data-resolve-id="${c.conflict_id}" data-strategy="server_wins">
                    ☁️ ${lang === 'hi' ? 'सर्वर संस्करण रखें' : 'Keep Server Version'}
                  </button>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    this.container.querySelectorAll('[data-resolve-id]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const cId = e.currentTarget.getAttribute('data-resolve-id');
        const strat = e.currentTarget.getAttribute('data-strategy');
        this.resolveConflict(cId, strat);
      });
    });
  }
}
