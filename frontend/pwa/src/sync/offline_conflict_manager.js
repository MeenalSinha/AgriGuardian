// AgriGuardian Conflict Detection & Visual Diff Manager
// Explains version collisions cleanly to the farmer in both English and Hindi

export class OfflineConflictManager {
  computeDiff(clientPayload = {}, serverPayload = {}) {
    const allKeys = new Set([...Object.keys(clientPayload), ...Object.keys(serverPayload)]);
    const diffs = [];

    const ignoredKeys = new Set(['created_at', 'updated_at', 'farmer_id', 'reminder_id', 'activity_id', 'cycle_id']);

    for (const key of allKeys) {
      if (ignoredKeys.has(key)) continue;

      const clientVal = clientPayload[key];
      const serverVal = serverPayload[key];

      if (JSON.stringify(clientVal) !== JSON.stringify(serverVal)) {
        diffs.push({
          field: key,
          clientValue: clientVal !== undefined ? clientVal : '(empty)',
          serverValue: serverVal !== undefined ? serverVal : '(empty)'
        });
      }
    }
    return diffs;
  }

  getExplanation(resourceType, diffs, language = 'en') {
    if (language === 'hi') {
      return `यह रिकॉर्ड किसी अन्य डिवाइस पर बदला जा चुका है। ${diffs.length} फ़ील्ड्स में अंतर पाया गया है। कृपया समीक्षा करें।`;
    }
    return `This ${resourceType.replace('_', ' ')} was modified on another device. ${diffs.length} field(s) have conflicting values. Review differences below before synchronizing.`;
  }
}

export const conflictManager = new OfflineConflictManager();
