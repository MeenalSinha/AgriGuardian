// AgriGuardian Offline Mutation Queue
// Durable client-side mutation journal ensuring zero data loss during network disruptions

import { dbStorage } from '../storage/indexed_db.js';
import { MutationStatus } from '../types.js';

export class OfflineMutationQueue {
  constructor(storage = dbStorage) {
    this.storage = storage;
  }

  _generateUUID() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return 'mut_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
  }

  async enqueue(resourceType, operation, clientRecordId, payload, baseVersion = 0) {
    const mutationId = 'mut_' + this._generateUUID();
    const mutation = {
      mutation_id: mutationId,
      resource_type: resourceType,
      operation: operation.toUpperCase(),
      client_record_id: clientRecordId,
      payload: payload,
      base_version: baseVersion,
      status: MutationStatus.PENDING_SYNC,
      retry_count: 0,
      last_error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    console.log(`[MutationQueue] Enqueued mutation ${mutationId} (${operation} on ${resourceType})`);
    await this.storage.put('mutations', mutation);
    return mutation;
  }

  async getPendingMutations() {
    const all = await this.storage.getAll('mutations');
    return all.filter((m) => m.status === MutationStatus.PENDING_SYNC || m.status === MutationStatus.FAILED);
  }

  async getPendingCount() {
    const pending = await this.getPendingMutations();
    return pending.length;
  }

  async markSyncing(mutationIds) {
    for (const id of mutationIds) {
      const mut = await this.storage.get('mutations', id);
      if (mut) {
        mut.status = MutationStatus.SYNCING;
        mut.updated_at = new Date().toISOString();
        await this.storage.put('mutations', mut);
      }
    }
  }

  async recordResult(mutationId, status, serverRecordId, serverVersion, error = null, conflictDetails = null) {
    const mut = await this.storage.get('mutations', mutationId);
    if (!mut) return;

    mut.status = status;
    mut.server_record_id = serverRecordId;
    mut.server_version = serverVersion;
    mut.last_error = error;
    mut.conflict_details = conflictDetails;
    mut.updated_at = new Date().toISOString();

    if (status === MutationStatus.FAILED || status === MutationStatus.REJECTED) {
      mut.retry_count = (mut.retry_count || 0) + 1;
    }

    await this.storage.put('mutations', mut);
    console.log(`[MutationQueue] Mutation ${mutationId} marked as ${status}`);
  }

  async retryFailed(mutationId) {
    const mut = await this.storage.get('mutations', mutationId);
    if (mut) {
      mut.status = MutationStatus.PENDING_SYNC;
      mut.last_error = null;
      mut.updated_at = new Date().toISOString();
      await this.storage.put('mutations', mut);
    }
  }

  async clearCompleted() {
    const all = await this.storage.getAll('mutations');
    for (const m of all) {
      if (m.status === MutationStatus.SYNCED) {
        await this.storage.delete('mutations', m.mutation_id);
      }
    }
  }
}

export const mutationQueue = new OfflineMutationQueue();
