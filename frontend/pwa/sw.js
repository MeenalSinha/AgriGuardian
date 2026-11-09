// AgriGuardian PWA Service Worker — Offline Shell Caching & Resilience
// Version: 1.10.0

const CACHE_NAME = 'agriguardian-shell-v1.10.0';

const PRECACHE_ASSETS = [
  '/pwa/',
  '/pwa/index.html',
  '/pwa/app.css',
  '/pwa/manifest.json',
  '/pwa/assets/icons/icon-192.svg',
  '/pwa/assets/icons/icon-512.svg',
  '/pwa/src/types.js',
  '/pwa/src/storage/indexed_db.js',
  '/pwa/src/sync/offline_network_manager.js',
  '/pwa/src/sync/offline_mutation_queue.js',
  '/pwa/src/sync/offline_conflict_manager.js',
  '/pwa/src/sync/offline_sync_manager.js',
  '/pwa/src/components/OfflineIndicator.js',
  '/pwa/src/components/SyncCenter.js',
  '/pwa/src/components/OfflineFarmerDashboard.js',
  '/pwa/src/components/OfflineActivityForm.js',
  '/pwa/src/components/OfflineReminderManager.js',
  '/pwa/src/components/OfflineLeafCapture.js',
  '/pwa/src/components/ConflictResolver.js',
  '/pwa/src/app.js'
];

// Install: precache the entire application shell
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Precaching AgriGuardian application shell...');
      // Use catch for each to prevent one missing file from aborting the install
      return Promise.allSettled(
        PRECACHE_ASSETS.map((url) => cache.add(url).catch((err) => console.warn(`[SW] Precache skipped for ${url}:`, err)))
      );
    })
  );
});

// Activate: clean up older cache versions
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log(`[ServiceWorker] Removing outdated cache: ${name}`);
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: Cache-first for shell/static, Network-first for dynamic API with safe fallback
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Exclude non-GET requests from service worker cache (mutations are handled by IndexedDB queue)
  if (event.request.method !== 'GET') {
    return;
  }

  // Static Assets / Shell: Cache-First
  if (url.pathname.startsWith('/pwa/')) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          // Stale-while-revalidate in background
          fetch(event.request).then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              caches.open(CACHE_NAME).then((cache) => cache.put(event.request, networkResponse));
            }
          }).catch(() => {});
          return cachedResponse;
        }
        return fetch(event.request).then((networkResponse) => {
          if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
            return networkResponse;
          }
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseToCache));
          return networkResponse;
        }).catch(() => {
          // Offline fallback
          return caches.match('/pwa/index.html');
        });
      })
    );
    return;
  }

  // API Requests: Network-first, do not blindly cache sensitive data in HTTP cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request).catch((err) => {
        // Return structured offline 503 response if network is dead
        return new Response(
          JSON.stringify({
            error: 'OFFLINE_UNAVAILABLE',
            detail: 'Network connection is unavailable. Cached local data is accessible in PWA offline mode.',
            timestamp: new Date().toISOString()
          }),
          {
            status: 503,
            statusText: 'Service Unavailable (Offline)',
            headers: { 'Content-Type': 'application/json' }
          }
        );
      })
    );
  }
});
