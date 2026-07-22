import * as ort from 'onnxruntime-web';
import { validatePlate, normalizePlate } from './validator.js';
import { Corrector } from './corrector.js';

// ─── Config ─────────────────────────────────────
const OCR_MODEL = 'assets/models/ppocr_rec.onnx';
const DETECTOR_MODEL = 'assets/models/yolov8_plate.onnx';

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const plateResult = document.getElementById('plate-result');
const logBox = document.getElementById('log-box');
const fpsDisplay = document.getElementById('fps-display');
const detHistory = document.getElementById('detection-history');

let detectorSession = null;
let ocrSession = null;
let corrector = new Corrector();
let running = false;
let stream = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let streakCandidate = null;
let streakCount = 0;
let lastPlate = null;
let lastPlateBox = null;
let noDetectFrames = 0;
let detectionHistory = [];

// ─── Logging ────────────────────────────────────
function log(msg, type = 'info') {
  const d = document.createElement('div');
  d.className = type;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logBox.appendChild(d);
  logBox.scrollTop = logBox.scrollHeight;
  if (logBox.children.length > 200) logBox.removeChild(logBox.firstChild);
}

function getParams() {
  return {
    conf: parseFloat(document.getElementById('param-conf').value),
    iou: parseFloat(document.getElementById('param-iou').value),
    alpha: parseFloat(document.getElementById('param-alpha').value),
    hang: parseInt(document.getElementById('param-hang').value),
    edge: parseFloat(document.getElementById('param-edge').value),
    row: parseFloat(document.getElementById('param-row').value),
    col: parseFloat(document.getElementById('param-col').value),
    minH: parseInt(document.getElementById('param-minh').value),
    sizeLimit: parseInt(document.getElementById('param-sizelimit').value),
    streak: parseInt(document.getElementById('param-streak').value),
    noDetect: parseInt(document.getElementById('param-nodetect').value),
    maxBoxes: parseInt(document.getElementById('param-maxboxes').value),
  };
}

function updateConfigPreview() {
  const p = getParams();
  document.getElementById('config-preview').textContent = JSON.stringify(p, null, 2);
}
document.querySelectorAll('#paramTabs .nav-link').forEach(tab => {
  tab.addEventListener('shown.bs.tab', () => updateConfigPreview());
});
document.querySelectorAll('.controls-section input[type="range"]').forEach(el => {
  el.addEventListener('input', () => {
    const id = el.id.replace('param-', '');
    const val = document.getElementById(`val-${id}`);
    if (val) val.textContent = el.value;
    updateConfigPreview();
  });
});
updateConfigPreview();

// ─── Camera ─────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'environment' } });
    video.srcObject = stream;
    await video.play();
    log('Camara iniciada 1280x720', 'ok');
  } catch (e) {
    log('Error camara: ' + e.message, 'err');
  }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
}

// ─── Models ─────────────────────────────────────
async function loadModels() {
  log('Cargando modelos...');
  try {
    detectorSession = await ort.InferenceSession.create(DETECTOR_MODEL, { executionProviders: ['wasm'] });
    log(`Detector ONNX cargado`, 'ok');
  } catch (e) {
    log('Detector no cargado (fallback solo): ' + e.message, 'warn');
  }
  try {
    ocrSession = await ort.InferenceSession.create(OCR_MODEL, { executionProviders: ['wasm'] });
    log(`OCR ONNX cargado`, 'ok');
  } catch (e) {
    log('OCR no cargado: ' + e.message, 'err');
  }
}

// ─── Preprocess (640x640) ───────────────────────
function preprocess(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const isz = 640;
  const maxSize = Math.max(vw, vh);
  const scale = isz / maxSize;
  const ox = (isz - vw * scale) / 2, oy = (isz - vh * scale) / 2;

  const canvas = document.createElement('canvas');
  canvas.width = isz; canvas.height = isz;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, isz, isz);
  ctx.drawImage(video, ox, oy, vw * scale, vh * scale);

  const rgba = ctx.getImageData(0, 0, isz, isz).data;
  const ch = isz * isz;
  const data = new Float32Array(3 * ch);
  for (let i = 0; i < ch; i++) {
    const pi = i << 2;
    data[i] = rgba[pi] / 255;
    data[ch + i] = rgba[pi + 1] / 255;
    data[2 * ch + i] = rgba[pi + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', data, [1, 3, isz, isz]), transform: { scale, ox, oy, vw, vh } };
}

function decodeOutput(data, dims, inputSize) {
  const numPreds = dims[2];
  const raw = [];
  const maxCx = Math.max(...data.slice(0, Math.min(numPreds, 100)));
  const normalized = maxCx < 2;
  const scale = normalized ? inputSize : 1;
  for (let i = 0; i < numPreds; i++) {
    const cx = data[i], cy = data[numPreds + i];
    const w = data[2 * numPreds + i], h = data[3 * numPreds + i];
    const conf = data[4 * numPreds + i];
    if (conf < 0.05) continue;
    raw.push({ x1: (cx - w / 2) * scale, y1: (cy - h / 2) * scale, x2: (cx + w / 2) * scale, y2: (cy + h / 2) * scale, score: conf });
  }
  return raw;
}

function toOriginal(box, t) {
  return {
    x1: Math.max(0, (box.x1 - t.ox) / t.scale),
    y1: Math.max(0, (box.y1 - t.oy) / t.scale),
    x2: Math.min(t.vw, (box.x2 - t.ox) / t.scale),
    y2: Math.min(t.vh, (box.y2 - t.oy) / t.scale),
    score: box.score,
  };
}

function nms(detections, iouThresh) {
  detections.sort((a, b) => b.score - a.score);
  const picked = new Array(detections.length).fill(false);
  const result = [];
  for (let i = 0; i < detections.length; i++) {
    if (picked[i]) continue;
    result.push(detections[i]);
    for (let j = i + 1; j < detections.length; j++) {
      if (picked[j]) continue;
      const x1 = Math.max(detections[i].x1, detections[j].x1);
      const y1 = Math.max(detections[i].y1, detections[j].y1);
      const x2 = Math.min(detections[i].x2, detections[j].x2);
      const y2 = Math.min(detections[i].y2, detections[j].y2);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const areaA = (detections[i].x2 - detections[i].x1) * (detections[i].y2 - detections[i].y1);
      const areaB = (detections[j].x2 - detections[j].x1) * (detections[j].y2 - detections[j].y1);
      if (inter / (areaA + areaB - inter + 1e-6) >= iouThresh) picked[j] = true;
    }
  }
  return result;
}

// ─── Fallback ───────────────────────────────────
function detectFallback(video, params) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (vw < 50 || vh < 50) return null;

  const canvas = document.createElement('canvas');
  canvas.width = vw; canvas.height = vh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  const imageData = ctx.getImageData(0, 0, vw, vh);
  const data = imageData.data;

  const gray = new Uint8Array(vw * vh);
  for (let i = 0; i < vw * vh; i++) {
    const pi = i << 2;
    gray[i] = (data[pi] * 0.299 + data[pi + 1] * 0.587 + data[pi + 2] * 0.114) | 0;
  }

  const grad = new Float32Array(vw * vh);
  let maxGrad = 0;
  for (let y = 0; y < vh; y++) {
    for (let x = 0; x < vw - 1; x++) {
      const idx = y * vw + x;
      const g = Math.abs(gray[idx] - gray[idx + 1]);
      grad[idx] = g;
      if (g > maxGrad) maxGrad = g;
    }
  }
  if (maxGrad < 5) return null;

  const edgeThresh = maxGrad * params.edge;

  const rowCount = new Uint16Array(vh);
  for (let y = 0; y < vh; y++) {
    let c = 0;
    for (let x = 0; x < vw; x++) { if (grad[y * vw + x] > edgeThresh) c++; }
    rowCount[y] = c;
  }
  let maxR = 0;
  for (let y = 0; y < vh; y++) { if (rowCount[y] > maxR) maxR = rowCount[y]; }
  if (maxR < 3) return null;

  const rowMin = Math.max(3, maxR * params.row);
  let yS = -1, yE = -1;
  for (let y = 0; y < vh; y++) {
    if (rowCount[y] > rowMin) { if (yS === -1) yS = y; yE = y; }
  }
  if (yS === -1 || yE - yS < 3) return null;
  yS = Math.max(0, yS - 3); yE = Math.min(vh, yE + 3);
  const bandH = yE - yS;

  const colCount = new Uint16Array(vw);
  for (let x = 0; x < vw; x++) {
    let c = 0;
    for (let y = yS; y < yE; y++) { if (grad[y * vw + x] > edgeThresh) c++; }
    colCount[x] = c;
  }
  let maxC = 0;
  for (let x = 0; x < vw; x++) { if (colCount[x] > maxC) maxC = colCount[x]; }
  if (maxC < 3) return null;

  const colMin = Math.max(3, maxC * params.col);
  let xS = -1, xE = -1;
  for (let x = 0; x < vw; x++) {
    if (colCount[x] > colMin) { if (xS === -1) xS = x; xE = x; }
  }
  if (xS === -1 || xE - xS < 10) return null;
  xS = Math.max(0, xS - 4); xE = Math.min(vw, xE + 4);

  const pw = xE - xS, ph = yE - yS;
  if (pw < 10 || ph < params.minH) return null;
  const limit = params.sizeLimit / 100;
  if (pw > vw * limit || ph > vh * limit) return null;

  const aspect = pw / ph;
  const aspectScore = Math.max(0, 1 - Math.abs(aspect - 3.5) / 5);
  const densityScore = Math.min(1, (maxR + maxC) / (vw * 0.05 + vh * 0.5));
  const score = 0.2 + 0.4 * aspectScore + 0.4 * densityScore;
  return { x1: xS, y1: yS, x2: xE, y2: yE, score: Math.min(0.7, score) };
}

// ─── OCR ────────────────────────────────────────
async function recognize(crop) {
  if (!ocrSession) return '';
  const w = crop.width, h = crop.height;
  const newH = 48;
  const newW = Math.max(1, Math.round(w * (newH / h)));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.putImageData(crop, 0, 0);
  const d = document.createElement('canvas');
  d.width = newW; d.height = newH;
  d.getContext('2d').drawImage(c, 0, 0, newW, newH);
  const rgba = d.getContext('2d').getImageData(0, 0, newW, newH).data;

  const nchw = new Float32Array(3 * newH * newW);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcIdx = (y * newW + x) * 4;
      const dstIdx = y * newW + x;
      nchw[dstIdx] = rgba[srcIdx] / 127.5 - 1.0;
      nchw[1 * newH * newW + dstIdx] = rgba[srcIdx + 1] / 127.5 - 1.0;
      nchw[2 * newH * newW + dstIdx] = rgba[srcIdx + 2] / 127.5 - 1.0;
    }
  }

  const tensor = new ort.Tensor('float32', nchw, [1, 3, newH, newW]);
  const inputName = ocrSession.inputNames[0];
  const results = await ocrSession.run({ [inputName]: tensor });
  const output = results[ocrSession.outputNames[0]];
  return decodeCTC(output.data, output.dims);
}

let ppocrKeys = null;

async function loadKeys() {
  try {
    const res = await fetch('assets/models/ppocr_keys.json');
    if (res.ok) ppocrKeys = await res.json();
  } catch (_) {}
  if (!ppocrKeys || !ppocrKeys.length) {
    ppocrKeys = ['']; for (let i = 0; i < 6625; i++) ppocrKeys.push('');
    log('Keys no cargadas, OCR puede fallar', 'err');
  } else {
    log(`Keys cargadas: ${ppocrKeys.length} caracteres`, 'ok');
  }
}

function decodeCTC(data, dims) {
  const timeSteps = dims[1];
  const vocabSize = dims[2];
  let prev = 0;
  let text = '';
  for (let t = 0; t < timeSteps; t++) {
    let bestIdx = 0, bestVal = -Infinity;
    const offset = t * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      if (data[offset + v] > bestVal) { bestVal = data[offset + v]; bestIdx = v; }
    }
    if (bestIdx !== 0 && bestIdx !== prev && ppocrKeys) text += ppocrKeys[bestIdx] || '';
    prev = bestIdx;
  }
  return text;
}

// ─── Draw overlay ───────────────────────────────
function drawOverlay(boxes, fallbackBox) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  for (const b of boxes) {
    const iw = b.x2 - b.x1, ih = b.y2 - b.y1;
    if (iw < 2 || ih < 2) continue;
    ctx.strokeStyle = '#ffcc00';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x1, b.y1, iw, ih);
    ctx.fillStyle = '#ffcc00';
    ctx.font = '11px monospace';
    ctx.fillText(`${(b.score*100).toFixed(0)}%`, b.x1, b.y1 - 3);
  }
  if (fallbackBox) {
    ctx.strokeStyle = '#00aaff';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(fallbackBox.x1, fallbackBox.y1, fallbackBox.x2 - fallbackBox.x1, fallbackBox.y2 - fallbackBox.y1);
    ctx.setLineDash([]);
    ctx.fillStyle = '#00aaff';
    ctx.font = '11px monospace';
    ctx.fillText(`F ${(fallbackBox.score*100).toFixed(0)}%`, fallbackBox.x1, fallbackBox.y2 + 14);
  }
  if (lastPlateBox && lastPlate) {
    ctx.strokeStyle = '#00ff66';
    ctx.lineWidth = 3;
    ctx.strokeRect(lastPlateBox.x1, lastPlateBox.y1, lastPlateBox.x2 - lastPlateBox.x1, lastPlateBox.y2 - lastPlateBox.y1);
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    const txt = formatPlate(lastPlate);
    ctx.font = 'bold 16px monospace';
    const tw = ctx.measureText(txt).width;
    ctx.fillRect(lastPlateBox.x1, Math.max(0, lastPlateBox.y1 - 28), tw + 12, 24);
    ctx.fillStyle = '#00ff66';
    ctx.fillText(txt, lastPlateBox.x1 + 6, Math.max(0, lastPlateBox.y1 - 8));
  }
}

function formatPlate(p) {
  if (/^[A-Z]{4}\d{2}$/.test(p)) return p.slice(0,4) + ' ' + p.slice(4);
  if (/^[A-Z]{2}\d{4}$/.test(p)) return p.slice(0,2) + ' ' + p.slice(2);
  return p;
}

// ─── Loop ───────────────────────────────────────
let smoothBox = null;

async function loop() {
  if (!running) return;
  if (!video.videoWidth) { requestAnimationFrame(loop); return; }

  const params = getParams();

  // Resize overlay
  if (overlay.width !== video.videoWidth || overlay.height !== video.videoHeight) {
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  }

  let yoloBoxes = [];
  if (detectorSession && params.maxBoxes > 0) {
    const { tensor, transform } = preprocess(video);
    const results = await detectorSession.run({ [detectorSession.inputNames[0]]: tensor });
    const output = results[detectorSession.outputNames[0]];
    const raw = decodeOutput(output.data, output.dims, 640);
    const nmsBoxes = nms(raw, params.iou);
    const high = nmsBoxes.filter(b => b.score >= params.conf);
    yoloBoxes = high.map(b => toOriginal(b, transform)).filter(b => {
      const w = b.x2 - b.x1, h = b.y2 - b.y1;
      return w >= 30 && h >= 10 && w/h < 12;
    }).slice(0, params.maxBoxes);
  }

  let fallbackBox = null;
  if (yoloBoxes.length === 0) {
    fallbackBox = detectFallback(video, params);
    if (fallbackBox) {
      if (smoothBox) {
        const d = Math.hypot(fallbackBox.x1 - smoothBox.x1, fallbackBox.y1 - smoothBox.y1);
        const sz = Math.hypot(smoothBox.x2 - smoothBox.x1, smoothBox.y2 - smoothBox.y1);
        if (d > sz * 1.5) smoothBox = null;
      }
      if (smoothBox) {
        const a = params.alpha;
        smoothBox.x1 += (fallbackBox.x1 - smoothBox.x1) * a;
        smoothBox.y1 += (fallbackBox.y1 - smoothBox.y1) * a;
        smoothBox.x2 += (fallbackBox.x2 - smoothBox.x2) * a;
        smoothBox.y2 += (fallbackBox.y2 - smoothBox.y2) * a;
        smoothBox.score = Math.max(0.5, fallbackBox.score);
        smoothBox.hangCount = 0;
      } else {
        smoothBox = { ...fallbackBox, hangCount: 0 };
      }
      fallbackBox = { x1: smoothBox.x1, y1: smoothBox.y1, x2: smoothBox.x2, y2: smoothBox.y2, score: smoothBox.score };
    } else if (smoothBox) {
      smoothBox.hangCount++;
      if (smoothBox.hangCount > params.hang) {
        smoothBox = null;
      } else {
        smoothBox.score = Math.max(0.2, smoothBox.score * 0.96);
        fallbackBox = { x1: smoothBox.x1, y1: smoothBox.y1, x2: smoothBox.x2, y2: smoothBox.y2, score: smoothBox.score };
      }
    }
  } else {
    smoothBox = null;
  }

  const primary = yoloBoxes.length > 0 ? yoloBoxes[0] : fallbackBox;

  // Process detection
  if (primary) {
    noDetectFrames = 0;
    const crop = cropPlate(video, primary);
    if (crop) {
      const rawText = await recognize(crop);
      if (rawText) {
        const normalized = normalizePlate(rawText);
        const corrected = corrector.correct(normalized);
        log(`OCR: "${rawText}" → "${normalized}" → "${corrected}"`, corrected ? 'ok' : 'warn');

        if (validatePlate(corrected) && corrected.length === 6) {
          if (corrected === streakCandidate) {
            streakCount++;
          } else {
            streakCandidate = corrected;
            streakCount = 1;
          }
          log(`Streak: "${corrected}" x${streakCount}/${params.streak}`, streakCount >= params.streak ? 'ok' : 'info');
          if (streakCount >= params.streak && corrected !== lastPlate) {
            lastPlate = corrected;
            lastPlateBox = { x1: primary.x1, y1: primary.y1, x2: primary.x2, y2: primary.y2 };
            plateResult.textContent = formatPlate(corrected);
            plateResult.style.color = '#00d26a';
            const config = JSON.stringify(params);
            detectionHistory.unshift({ plate: corrected, time: new Date().toLocaleTimeString(), config });
            if (detectionHistory.length > 20) detectionHistory.pop();
            renderHistory();
            log(`✅ PLACA: ${corrected}`, 'ok');
            streakCount = 0;
          }
        } else {
          streakCandidate = null;
          streakCount = 0;
        }
      } else {
        streakCandidate = null;
        streakCount = 0;
      }
    }
  } else {
    noDetectFrames++;
    streakCandidate = null;
    streakCount = 0;
    if (noDetectFrames > params.noDetect && lastPlate) {
      lastPlate = null;
      lastPlateBox = null;
      plateResult.textContent = '------------';
      plateResult.style.color = '';
    }
  }

  drawOverlay(yoloBoxes, fallbackBox);

  // FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsDisplay.textContent = `FPS ${frameCount}`;
    frameCount = 0;
    lastFpsTime = now;
  }

  requestAnimationFrame(loop);
}

function cropPlate(video, box) {
  const w = Math.round(box.x2 - box.x1), h = Math.round(box.y2 - box.y1);
  if (w < 4 || h < 4) return null;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(video, box.x1, box.y1, w, h, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function renderHistory() {
  detHistory.innerHTML = '';
  detectionHistory.forEach((d, i) => {
    const card = document.createElement('div');
    card.className = 'detection-card';
    card.innerHTML = `<div class="d-flex justify-content-between">
      <span class="plate">${formatPlate(d.plate)}</span>
      <span class="text-muted">${d.time}</span>
    </div>
    <div class="mt-1"><code style="cursor:pointer;user-select:all" onclick="navigator.clipboard.writeText(this.textContent).then(()=>this.style.color='#0f0').catch(()=>{})">${d.config}</code></div>`;
    detHistory.appendChild(card);
  });
}

// ─── Controls ───────────────────────────────────
document.getElementById('btn-start').addEventListener('click', async () => {
  if (running) return;
  await startCamera();
  running = true;
  streakCandidate = null;
  streakCount = 0;
  lastPlate = null;
  lastPlateBox = null;
  smoothBox = null;
  noDetectFrames = 0;
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  log('Pipeline iniciado', 'ok');
  loop();
});

document.getElementById('btn-stop').addEventListener('click', () => {
  running = false;
  stopCamera();
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  log('Detenido', 'warn');
});

document.getElementById('btn-clear-log').addEventListener('click', () => {
  logBox.innerHTML = '<div>[Log limpiado]</div>';
});

// ─── Init ───────────────────────────────────────
log('Inicializando...');
await Promise.all([loadModels(), loadKeys()]);
log('Listo. Presiona Iniciar', 'ok');
