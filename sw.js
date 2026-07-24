/* sw.js - Service Worker para caché offline */
const VERSION = 'v3.0.0';
const STATIC_CACHE = `static-${VERSION}`;
const RUNTIME_CACHE = `runtime-${VERSION}`;
const MODEL_CACHE = `models-${VERSION}`;

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/test_camera.html',
  '/test_ocr.html',
  '/test_detector.html',
  '/manifest.json',
  '/assets/css/app.css',
  '/assets/css/index.css',
  '/assets/css/test_camera.css',
  '/assets/css/test_detector.css',
  '/assets/css/test_ocr.css',
  '/assets/js/app.js',
  '/assets/js/camera.js',
  '/assets/js/corrector.js',
  '/assets/js/database.js',
  '/assets/js/detector.js',
  '/assets/js/ocr.js',
  '/assets/js/validator.js',
  '/assets/js/test_camera.js',
  '/assets/js/test_detector.js',
  '/assets/js/test_ocr.js',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs',
  'https://cdn.jsdelivr.net/npm/dexie@4.0.8/dist/dexie.min.mjs'
];

const MODEL_URLS = [
  '/assets/models/yolov8_plate.onnx',
  '/assets/models/ppocr_rec.onnx',
  '/assets/models/ppocr_keys.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const staticCache = await caches.open(STATIC_CACHE);
    await safeCacheAll(staticCache, STATIC_ASSETS);
    const modelCache = await caches.open(MODEL_CACHE);
    await safeCacheAll(modelCache, MODEL_URLS);
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((k) => ![STATIC_CACHE, RUNTIME_CACHE, MODEL_CACHE].includes(k))
        .map((k) => caches.delete(k))
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin === location.origin && url.pathname.startsWith('/assets/models/')) {
    event.respondWith(cacheFirst(req, MODEL_CACHE));
    return;
  }

  if (url.origin !== location.origin) {
    event.respondWith(cacheFirst(req, STATIC_CACHE));
    return;
  }

  event.respondWith(networkFirst(req));
});

async function safeCacheAll(cache, urls) {
  await Promise.all(
    urls.map(async (u) => {
      try { await cache.add(u); } catch (_) {}
    })
  );
}

async function cacheFirst(req, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(req);
  if (cached) return cached;
  try {
    const res = await fetch(req);
    if (res && res.ok) cache.put(req, res.clone());
    return res;
  } catch (_) {
    return cached || new Response('', { status: 504 });
  }
}

async function networkFirst(req) {
  const cache = await caches.open(RUNTIME_CACHE);
  try {
    const res = await fetch(req);
    if (res && res.ok && req.url.startsWith(location.origin)) {
      cache.put(req, res.clone());
    }
    return res;
  } catch (_) {
    const cached = await cache.match(req);
    if (cached) return cached;
    const staticCache = await caches.open(STATIC_CACHE);
    return (await staticCache.match(req)) || (await staticCache.match('/index.html')) || new Response('', { status: 504 });
  }
}