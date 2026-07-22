import * as ort from 'onnxruntime-web';

let session;
let currentImg = null;

const logBox = document.getElementById('log-box');
const imgEl = document.getElementById('uploaded-img');
const overlay = document.getElementById('overlay');
const container = document.getElementById('img-container');

function log(msg, type = 'info') {
  const d = document.createElement('div'); d.className = type;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logBox.appendChild(d); logBox.scrollTop = logBox.scrollHeight;
}

async function load() {
  log('Cargando detector...');
  try {
    session = await ort.InferenceSession.create('assets/models/yolov8_plate.onnx', { executionProviders: ['wasm'] });
    log('Detector cargado', 'ok');
  } catch (e) { log('Error: ' + e.message, 'err'); }
}
await load();

function getParams() {
  return {
    conf: parseFloat(document.getElementById('param-conf').value),
    iou: parseFloat(document.getElementById('param-iou').value),
    imgsz: parseInt(document.getElementById('param-imgsz').value),
    edge: parseFloat(document.getElementById('param-edge').value),
    minH: parseInt(document.getElementById('param-minh').value),
  };
}
document.querySelectorAll('#tab-params input').forEach(el => {
  el.addEventListener('input', () => {
    const id = el.id.replace('param-', '');
    const val = document.getElementById(`val-${id}`);
    if (val) val.textContent = el.value;
  });
});

document.getElementById('file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    currentImg = ev.target.result;
    imgEl.src = currentImg;
    imgEl.style.display = 'block';
    log(`Imagen cargada: ${file.name}`, 'ok');
  };
  reader.readAsDataURL(file);
  e.target.value = '';
});

document.getElementById('btn-detect').addEventListener('click', detect);

async function detect() {
  if (!session || !currentImg) { log('Primero sube una imagen', 'warn'); return; }

  const p = getParams();
  log(`Detectando (conf=${p.conf}, imgsz=${p.imgsz})...`, 'info');

  const img = new Image();
  img.src = currentImg;
  await img.decode();

  const vw = img.naturalWidth, vh = img.naturalHeight;

  // Preprocess
  const { tensor, transform } = preprocess(img, p.imgsz);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]];
  const raw = decode(output.data, output.dims, p.imgsz);
  const nmsBoxes = nms(raw, p.iou);
  const high = nmsBoxes.filter(b => b.score >= p.conf);
  let yoloBoxes = high.map(b => toOriginal(b, transform)).filter(b => (b.x2-b.x1) >= 30 && (b.y2-b.y1) >= 10 && (b.x2-b.x1)/(b.y2-b.y1+1) < 12);

  let fallbackBox = null;
  if (yoloBoxes.length === 0) {
    fallbackBox = detectFallback(img, p);
    if (fallbackBox && fallbackBox.score) {
      log(`Fallback: ${(fallbackBox.score*100).toFixed(1)}%`, 'info');
    }
  }

  const maxConf = raw.length ? Math.max(...raw.map(b => b.score)) : 0;
  document.getElementById('stat-preds').textContent = raw.length;
  document.getElementById('stat-high').textContent = high.length;
  document.getElementById('stat-maxconf').textContent = (maxConf*100).toFixed(1) + '%';
  document.getElementById('stat-fb').textContent = fallbackBox ? (fallbackBox.score*100).toFixed(1) + '%' : '--';

  // Draw on overlay
  drawBoxes(yoloBoxes, fallbackBox, vw, vh);

  log(`YOLO: ${raw.length} predicciones, ${high.length} high, max ${(maxConf*100).toFixed(1)}%`, 'ok');
  if (yoloBoxes.length > 0) log(`✅ Placa detectada por YOLO`, 'ok');
  else if (fallbackBox) log(`⚠ Placa detectada por fallback`, 'warn');
  else log('❌ Sin detecci&oacute;n', 'err');
}

function preprocess(img, isz) {
  const vw = img.naturalWidth, vh = img.naturalHeight;
  const maxSize = Math.max(vw, vh);
  const scale = isz / maxSize;
  const ox = (isz - vw * scale) / 2, oy = (isz - vh * scale) / 2;
  const c = document.createElement('canvas');
  c.width = isz; c.height = isz;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, isz, isz);
  ctx.drawImage(img, ox, oy, vw * scale, vh * scale);
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

function decode(data, dims, isz) {
  const numPreds = dims[2];
  const raw = [];
  const maxCx = Math.max(...data.slice(0, Math.min(numPreds, 100)));
  const normalized = maxCx < 2;
  const scale = normalized ? isz : 1;
  for (let i = 0; i < numPreds; i++) {
    const cx = data[i], cy = data[numPreds + i];
    const w = data[2 * numPreds + i], h = data[3 * numPreds + i];
    const conf = data[4 * numPreds + i];
    if (conf < 0.05) continue;
    raw.push({ x1: (cx - w/2) * scale, y1: (cy - h/2) * scale, x2: (cx + w/2) * scale, y2: (cy + h/2) * scale, score: conf });
  }
  return raw;
}

function toOriginal(box, t) {
  return { x1: Math.max(0, (box.x1 - t.ox) / t.scale), y1: Math.max(0, (box.y1 - t.oy) / t.scale), x2: Math.min(t.vw, (box.x2 - t.ox) / t.scale), y2: Math.min(t.vh, (box.y2 - t.oy) / t.scale), score: box.score };
}

function nms(dets, thresh) {
  dets.sort((a, b) => b.score - a.score);
  const picked = new Array(dets.length).fill(false);
  const result = [];
  for (let i = 0; i < dets.length; i++) {
    if (picked[i]) continue; result.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (picked[j]) continue;
      const x1 = Math.max(dets[i].x1, dets[j].x1), y1 = Math.max(dets[i].y1, dets[j].y1);
      const x2 = Math.min(dets[i].x2, dets[j].x2), y2 = Math.min(dets[i].y2, dets[j].y2);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const aA = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
      const aB = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
      if (inter / (aA + aB - inter + 1e-6) >= thresh) picked[j] = true;
    }
  }
  return result;
}

function detectFallback(img, params) {
  const vw = img.naturalWidth, vh = img.naturalHeight;
  if (vw < 50 || vh < 50) return null;
  const c = document.createElement('canvas');
  c.width = vw; c.height = vh;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, vw, vh).data;
  const gray = new Uint8Array(vw * vh);
  for (let i = 0; i < vw * vh; i++) gray[i] = (data[i<<2] * 0.299 + data[(i<<2)+1] * 0.587 + data[(i<<2)+2] * 0.114) | 0;
  const grad = new Float32Array(vw * vh);
  let maxGrad = 0;
  for (let y = 0; y < vh; y++) for (let x = 0; x < vw - 1; x++) { const g = Math.abs(gray[y*vw+x] - gray[y*vw+x+1]); grad[y*vw+x] = g; if (g > maxGrad) maxGrad = g; }
  if (maxGrad < 5) return null;
  const edgeThresh = maxGrad * params.edge;
  const rowCount = new Uint16Array(vh);
  for (let y = 0; y < vh; y++) { let c = 0; for (let x = 0; x < vw; x++) { if (grad[y*vw+x] > edgeThresh) c++; } rowCount[y] = c; }
  let maxR = 0; for (let y = 0; y < vh; y++) { if (rowCount[y] > maxR) maxR = rowCount[y]; }
  if (maxR < 3) return null;
  const rowMin = Math.max(3, maxR * 0.15);
  let yS = -1, yE = -1;
  for (let y = 0; y < vh; y++) { if (rowCount[y] > rowMin) { if (yS === -1) yS = y; yE = y; } }
  if (yS === -1 || yE - yS < 3) return null;
  yS = Math.max(0, yS - 3); yE = Math.min(vh, yE + 3);
  const colCount = new Uint16Array(vw);
  for (let x = 0; x < vw; x++) { let c = 0; for (let y = yS; y < yE; y++) { if (grad[y*vw+x] > edgeThresh) c++; } colCount[x] = c; }
  let maxC = 0; for (let x = 0; x < vw; x++) { if (colCount[x] > maxC) maxC = colCount[x]; }
  if (maxC < 3) return null;
  const colMin = Math.max(3, maxC * 0.18);
  let xS = -1, xE = -1;
  for (let x = 0; x < vw; x++) { if (colCount[x] > colMin) { if (xS === -1) xS = x; xE = x; } }
  if (xS === -1 || xE - xS < 10) return null;
  xS = Math.max(0, xS - 4); xE = Math.min(vw, xE + 4);
  const pw = xE - xS, ph = yE - yS;
  if (pw < 10 || ph < params.minH) return null;
  if (pw > vw * 0.95 || ph > vh * 0.95) return null;
  const aspect = pw / ph;
  const aspectScore = Math.max(0, 1 - Math.abs(aspect - 3.5) / 5);
  const densityScore = Math.min(1, (maxR + maxC) / (vw * 0.05 + vh * 0.5));
  const score = 0.2 + 0.4 * aspectScore + 0.4 * densityScore;
  return { x1: xS, y1: yS, x2: xE, y2: yE, score: Math.min(0.7, score) };
}

function drawBoxes(yoloBoxes, fallbackBox, vw, vh) {
  // Resize overlay to match displayed image size
  const rect = imgEl.getBoundingClientRect();
  const displayW = rect.width;
  const displayH = rect.height;
  overlay.width = displayW;
  overlay.height = displayH;
  const scaleX = displayW / vw;
  const scaleY = displayH / vh;

  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // YOLO boxes
  for (const b of yoloBoxes) {
    const x = b.x1 * scaleX, y = b.y1 * scaleY;
    const w = (b.x2 - b.x1) * scaleX, h = (b.y2 - b.y1) * scaleY;
    ctx.strokeStyle = '#ffcc00'; ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = '#ffcc00'; ctx.font = 'bold 13px monospace';
    ctx.fillText(`${(b.score*100).toFixed(0)}%`, x, y - 5);
  }

  // Fallback
  if (fallbackBox) {
    const x = fallbackBox.x1 * scaleX, y = fallbackBox.y1 * scaleY;
    const w = (fallbackBox.x2 - fallbackBox.x1) * scaleX, h = (fallbackBox.y2 - fallbackBox.y1) * scaleY;
    ctx.strokeStyle = '#00aaff'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = '#00aaff'; ctx.font = 'bold 13px monospace';
    ctx.fillText(`F ${(fallbackBox.score*100).toFixed(0)}%`, x, y + h + 16);
  }
}
