import { Camera } from './camera.js';
import { PlateDetector } from './detector.js';
import { PlateOCR } from './ocr.js';
import { validatePlate, normalizePlate } from './validator.js';
import { Corrector } from './corrector.js';
import { getConfig, setConfig } from './database.js';

const OCR_MODEL = 'assets/models/ppocr_rec.onnx';
const DETECTOR_MODEL = 'assets/models/yolov8_plate.onnx';

const videoEl = document.getElementById('video');
const overlayEl = document.getElementById('overlay');
const plateEl = document.getElementById('plate-display');
const camera = new Camera(videoEl);

let detector, ocr, corrector;
let running = false;
let lastPlate = null;
let lastPlateBox = null;
let noDetectFrames = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let streakCandidate = null;
let streakCount = 0;
let streakRequired = 2;
let noDetectLimit = 20;
let lastRawText = '';
let lastCorrected = '';
let lastOCRValid = false;
// #7 OCR skip when box stable
let ocrCounter = 0;
let ocrSkipInterval = 3;
let lastBoxCenter = null;

async function init() {
  const resolution = await getConfig('resolution', '1280x720');
  const sensitivity = await getConfig('sensitivity', 0.15);
  streakRequired = await getConfig('streak', 2);
  noDetectLimit = await getConfig('noDetect', 20);
  const corrections = await getConfig('corrections', {
    letter: { '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B' },
    number: { 'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6' }
  });

  corrector = new Corrector(corrections);
  detector = new PlateDetector({ inputSize: 640, confThreshold: sensitivity });
  ocr = new PlateOCR();

  setLoading('Cargando detector...', 'YOLOv8');
  const detLoaded = await detector.loadModel(DETECTOR_MODEL);

  setLoading('Cargando OCR...', 'PP-OCRv4');
  const ocrLoaded = await ocr.loadModel(OCR_MODEL);

  setLoading('Calentando modelos...', 'Warmup');
  try { await Promise.race([
    Promise.all([detector.warmup(), ocr.warmup()]),
    new Promise((_, reject) => setTimeout(() => reject(new Error('warmup timeout')), 8000))
  ]); } catch (e) { console.warn('[App] warmup skipped:', e.message); }

  setLoading('Modelos listos', '');
  setTimeout(hideLoading, 500);

  updateSysInfo();
  setInterval(updateSysInfo, 2000);

  if (!detLoaded && !ocrLoaded) setStatus('Modo demo - coloca los .onnx en assets/models/');
  else setStatus('Listo. Presiona Iniciar');

  setupEvents();
}

function setupEvents() {
  document.getElementById('btn-start').addEventListener('click', start);
  document.getElementById('btn-stop').addEventListener('click', stop);
}

async function start() {
  const resolution = await getConfig('resolution', '1280x720');
  const fpsCfg = await getConfig('fps', 30);
  await camera.start(resolution, fpsCfg);
  running = true;
  lastPlate = null;
  lastPlateBox = null;
  noDetectFrames = 0;
  streakCandidate = null;
  streakCount = 0;
  if (detector) detector._smoothBox = null;
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  clearPlate();

  loop();
}

function stop() {
  running = false;
  camera.stop();
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  setStatus('Detenido.');
}

async function loop() {
  if (!running) return;
  const video = camera.getVideo();
  if (!video || !video.videoWidth) { requestAnimationFrame(loop); return; }

  resizeOverlay();

  const boxes = await detector.detect(video, overlayEl);

  if (boxes && boxes.length > 0) {
    noDetectFrames = 0;
    const box = boxes[0];

    if (lastPlate || streakCandidate) {
      lastPlateBox = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
    }

    // #7 OCR skip when box stable
    const center = { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 };
    const bw = box.x2 - box.x1, bh = box.y2 - box.y1;
    const isStable = lastBoxCenter &&
      Math.abs(center.x - lastBoxCenter.x) < bw * 0.15 &&
      Math.abs(center.y - lastBoxCenter.y) < bh * 0.15;
    lastBoxCenter = center;

    ocrCounter++;
    const skipOCR = isStable && (ocrCounter % ocrSkipInterval !== 0);

    if (skipOCR) {
      // Use cached text, still update box position
      if (lastPlate || streakCandidate) {
        lastPlateBox = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
      }
    } else {
      const crop = detector.cropPlate(video, box);
      if (crop) {
        const rawText = await ocr.recognize(crop);
        lastRawText = rawText || '(vacio)';
        if (rawText) {
          const normalized = normalizePlate(rawText);
          const corrected = corrector.correct(normalized);
          lastCorrected = corrected || '(vacio)';
          lastOCRValid = validatePlate(corrected) && corrected.length === 6;

          if (lastOCRValid) {
            if (corrected === streakCandidate) streakCount++;
            else { streakCandidate = corrected; streakCount = 1; }

            if (streakCount >= streakRequired && streakCandidate !== lastPlate) {
              lastPlate = streakCandidate;
              lastPlateBox = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
              showPlate(lastPlate);
              streakCount = 0;
            }
          } else {
            streakCandidate = null; streakCount = 0;
          }
        } else {
          lastCorrected = '(vacio)';
          lastOCRValid = false;
          streakCandidate = null; streakCount = 0;
        }
      }
    }
  } else {
    noDetectFrames++;
    streakCandidate = null; streakCount = 0;
    if (noDetectFrames > noDetectLimit && lastPlate) {
      lastPlate = null; lastPlateBox = null;
      clearPlate();
    }
  }

  drawOverlay();
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    document.getElementById('fps').textContent = `FPS ${frameCount}`;
    frameCount = 0;
    lastFpsTime = now;
  }
  requestAnimationFrame(loop);
}

function showPlate(plate) {
  if (!plateEl) return;
  let letters = '', numbers = '';
  if (/^[A-Z]{4}\d{2}$/.test(plate)) {
    letters = plate.slice(0, 4);
    numbers = plate.slice(4);
  } else if (/^[A-Z]{2}\d{4}$/.test(plate)) {
    letters = plate.slice(0, 2);
    numbers = plate.slice(2);
  }
  plateEl.querySelector('.plate-letters').textContent = letters;
  plateEl.querySelector('.plate-numbers').textContent = numbers;
  plateEl.classList.add('detected');
}

function clearPlate() {
  if (!plateEl) return;
  plateEl.querySelector('.plate-letters').textContent = 'ABCD';
  plateEl.querySelector('.plate-numbers').textContent = '12';
  plateEl.classList.remove('detected');
}

function drawOverlay() {
  if (!lastPlate || !lastPlateBox) return;
  const ctx = overlayEl.getContext('2d');
  const box = lastPlateBox;
  const iw = box.x2 - box.x1, ih = box.y2 - box.y1;
  if (iw < 2 || ih < 2) return;

  ctx.strokeStyle = '#00ff66';
  ctx.lineWidth = 4;
  ctx.strokeRect(box.x1, box.y1, iw, ih);

  const display = formatPlate(lastPlate);
  ctx.font = 'bold 20px monospace';
  const tw = ctx.measureText(display).width;
  const lx = box.x1;
  const ly = Math.max(4, box.y1 - 34);
  ctx.fillStyle = 'rgba(0,0,0,0.8)';
  ctx.fillRect(lx, ly, tw + 16, 30);
  ctx.fillStyle = '#00ff66';
  ctx.fillText(display, lx + 8, ly + 22);
}

function formatPlate(p) {
  if (/^[A-Z]{4}\d{2}$/.test(p)) return p.slice(0, 4) + ' ' + p.slice(4);
  if (/^[A-Z]{2}\d{4}$/.test(p)) return p.slice(0, 2) + ' ' + p.slice(2);
  return p;
}

function resizeOverlay() {
  const video = camera.getVideo();
  if (video && video.videoWidth) {
    if (overlayEl.width !== video.videoWidth || overlayEl.height !== video.videoHeight) {
      overlayEl.width = video.videoWidth;
      overlayEl.height = video.videoHeight;
    }
  }
}

function updateSysInfo() {
  const det = detector || { mock: true, _lastRawPreds: 0, _lastMaxConf: 0, _lastFallback: false };
  const ocrMock = ocr ? ocr.mock : true;
  const detStatus = det.mock ? '◌ MOCK' : '✓ OK';
  const ocrStatus = ocrMock ? '◌ MOCK' : '✓ OK';
  const appHtml =
    `<span class="label">Detector:</span> <span class="val">${detStatus}</span>\n` +
    `<span class="label">OCR:</span> <span class="val">${ocrStatus}</span>\n` +
    `<span class="label">Preds:</span> <span class="val">${det._lastRawPreds || 0}</span>\n` +
    `<span class="label">Max conf:</span> <span class="val">${((det._lastMaxConf || 0) * 100).toFixed(1)}%</span>\n` +
    `<span class="label">OCR raw:</span> <span class="val">"${lastRawText}"</span>\n` +
    `<span class="label">Placa:</span> <span class="val">"${lastCorrected}"</span>\n` +
    `<span class="label">Streak:</span> <span class="val">${streakCount}/${streakRequired} ${lastOCRValid ? '🟢' : '⚪'}</span>\n` +
    `<span class="label">FPS:</span> <span class="val">${frameCount}</span>`;
  document.getElementById('sys-app').innerHTML = appHtml;

  // Device info (collected once)
  const mem = navigator.deviceMemory !== undefined ? navigator.deviceMemory + ' GB' : 'N/D';
  const cpu = navigator.hardwareConcurrency || 'N/D';
  let gpu = 'N/D';
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    if (gl) gpu = gl.getParameter(gl.RENDERER);
  } catch (_) {}
  const deviceHtml =
    `<span class="label">RAM:</span> <span class="val">${mem}</span>\n` +
    `<span class="label">CPU cores:</span> <span class="val">${cpu}</span>\n` +
    `<span class="label">GPU:</span> <span class="val">${gpu}</span>\n` +
    `<span class="label">Plataforma:</span> <span class="val">${navigator.platform || 'N/D'}</span>`;
  document.getElementById('sys-device').innerHTML = deviceHtml;

  // Browser info
  const ua = navigator.userAgent;
  const shortUA = ua.length > 80 ? ua.slice(0, 80) + '...' : ua;
  const webglOk = (() => { try { return !!document.createElement('canvas').getContext('webgl'); } catch(_) { return false; } })();
  const webgpuOk = !!navigator.gpu;
  const wasmOk = typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function';
  const browserHtml =
    `<span class="label">UA:</span> <span class="val">${shortUA}</span>\n` +
    `<span class="label">WebGL:</span> <span class="val">${webglOk ? '✓' : '✗'}</span>\n` +
    `<span class="label">WebGPU:</span> <span class="val">${webgpuOk ? '✓' : '✗'}</span>\n` +
    `<span class="label">WASM:</span> <span class="val">${wasmOk ? '✓' : '✗'}</span>\n` +
    `<span class="label">Screen:</span> <span class="val">${screen.width}x${screen.height}</span>\n` +
    `<span class="label">DPR:</span> <span class="val">${window.devicePixelRatio || 1}</span>`;
  document.getElementById('sys-browser').innerHTML = browserHtml;
}

function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

function setLoading(text, detail) {
  const t = document.getElementById('loading-text');
  const d = document.getElementById('loading-detail');
  if (t) t.textContent = text;
  if (d) d.textContent = detail || '';
}

function hideLoading() {
  document.getElementById('loading-overlay')?.classList.add('hidden');
}

init().catch((e) => setStatus('Error: ' + e.message));
