// AgriGuardian Dual Connectivity Monitor
// Combines navigator.onLine with active backend reachability checks to avoid false online status

import { NetworkQuality } from '../types.js';

export class OfflineNetworkManager {
  constructor() {
    this.status = navigator.onLine ? NetworkQuality.ONLINE : NetworkQuality.OFFLINE;
    this.listeners = new Set();
    this.pingInterval = null;
    this.isChecking = false;

    window.addEventListener('online', () => this.checkConnectivity(true));
    window.addEventListener('offline', () => this._setStatus(NetworkQuality.OFFLINE));

    // Initial check
    this.checkConnectivity();
    this.startHeartbeat();
  }

  subscribe(callback) {
    this.listeners.add(callback);
    callback(this.status);
    return () => this.listeners.delete(callback);
  }

  _setStatus(newStatus) {
    if (this.status !== newStatus) {
      console.log(`[NetworkManager] Status transitioned: ${this.status} -> ${newStatus}`);
      this.status = newStatus;
      for (const listener of this.listeners) {
        try {
          listener(this.status);
        } catch (e) {
          console.error(e);
        }
      }
    }
  }

  async checkConnectivity(immediate = false) {
    if (this.isChecking) return;
    this.isChecking = true;

    if (!navigator.onLine) {
      this._setStatus(NetworkQuality.OFFLINE);
      this.isChecking = false;
      return;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);

      const startTime = performance.now();
      const res = await fetch('/api/v1/sync/status', {
        method: 'GET',
        cache: 'no-store',
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      const latency = performance.now() - startTime;

      if (res.ok) {
        if (latency > 2500) {
          this._setStatus(NetworkQuality.WEAK);
        } else {
          this._setStatus(NetworkQuality.ONLINE);
        }
      } else {
        this._setStatus(NetworkQuality.OFFLINE);
      }
    } catch (err) {
      // Failed ping or timeout means unreachable backend
      this._setStatus(NetworkQuality.OFFLINE);
    } finally {
      this.isChecking = false;
    }
  }

  startHeartbeat(intervalMs = 20000) {
    if (this.pingInterval) clearInterval(this.pingInterval);
    this.pingInterval = setInterval(() => {
      // Only ping if navigator claims online, else rely on event
      if (navigator.onLine) {
        this.checkConnectivity();
      }
    }, intervalMs);
  }

  stopHeartbeat() {
    if (this.pingInterval) clearInterval(this.pingInterval);
  }
}

export const networkManager = new OfflineNetworkManager();
