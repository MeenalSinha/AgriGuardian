// AgriGuardian PWA Core Types and Status Constants

export const MutationStatus = {
  LOCAL_ONLY: 'local_only',
  PENDING_SYNC: 'pending_sync',
  SYNCING: 'syncing',
  SYNCED: 'synced',
  CONFLICT: 'conflict',
  FAILED: 'failed',
  REJECTED: 'rejected'
};

export const NetworkQuality = {
  ONLINE: 'online',
  WEAK: 'weak',
  OFFLINE: 'offline'
};

export const ConflictStrategy = {
  CLIENT_WINS: 'client_wins',
  SERVER_WINS: 'server_wins',
  MANUAL_MERGE: 'manual_merge'
};

export const ActivityStatus = {
  PLANNED: 'planned',
  DUE: 'due',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  POSTPONED: 'postponed',
  CANCELLED: 'cancelled'
};
