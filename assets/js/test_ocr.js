import * as ort from 'onnxruntime-web';
import { validatePlate, normalizePlate } from './validator.js';
import { Corrector } from './corrector.js';

let ocrSession, ppocrKeys, corrector;
let images = []; // { id, dataUrl, file }
let selectedId = null;

const fileInput = document.getElementById('file-input');
const imageList = document.getElementById('image-list');
const selectedView = document.getElementById('selected-view');
const resultArea = document.getElementById('result-area');
const logBox = document.getElementById('log-box');
const uploadCount = document.getElementById('upload-count');

function log(msg, type = 'info') {
  const d = document.createElement('div'); d.className = type;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logBox.appendChild(d); logBox.scrollTop = logBox.scrollHeight;
}

// ─── Load models ────────────────────────────────
async function load() {
  log('Cargando OCR...');
  try {
    ocrSession = await ort.InferenceSession.create('assets/models/ppocr_rec.onnx', { executionProviders: ['wasm'] });
    log('OCR cargado', 'ok');
  } catch (e) { log('Error OCR: ' + e.message, 'err'); }
  try {
    const r = await fetch('assets/models/ppocr_keys.json');
    ppocrKeys = await r.json();
    log(`Keys: ${ppocrKeys.length} caracteres`, 'ok');
  } catch (e) { log('Error keys: ' + e.message, 'err'); }
  corrector = new Corrector();
  log('Listo', 'ok');
}
await load();

// ─── File upload ────────────────────────────────
fileInput.addEventListener('change', () => {
  for (const file of fileInput.files) {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const reader = new FileReader();
    reader.onload = (e) => {
      images.push({ id, dataUrl: e.target.result, file, result: null });
      renderList();
      selectImage(id);
    };
    reader.readAsDataURL(file);
  }
  fileInput.value = '';
});

document.getElementById('btn-clear').addEventListener('click', () => {
  images = []; selectedId = null;
  renderList();
  selectedView.innerHTML = '<div class="text-center text-secondary small py-4">Seleccion&aacute; una imagen o sube una nueva</div>';
  resultArea.innerHTML = '';
  logBox.innerHTML = '<div>[Log limpiado]</div>';
});

document.getElementById('btn-detect-ocr').addEventListener('click', () => {
  if (!selectedId) { log('Primero sube una imagen', 'warn'); return; }
  images.forEach(img => { img.result = null; });
  images.forEach(img => processImage(img.id));
});

// ─── Image list ─────────────────────────────────
function renderList() {
  imageList.innerHTML = '';
  uploadCount.textContent = `${images.length} im&aacute;genes`;
  images.forEach(img => {
    const div = document.createElement('div');
    div.className = `image-item p-1 ${img.id === selectedId ? 'active' : ''}`;
    div.innerHTML = `<img src="${img.dataUrl}" class="img-preview" style="max-height:80px;width:100%;object-fit:cover;" />`;
    div.onclick = () => selectImage(img.id);
    imageList.appendChild(div);
  });
}

function selectImage(id) {
  selectedId = id;
  renderList();
  const img = images.find(i => i.id === id);
  if (!img) return;
  selectedView.innerHTML = `<img src="${img.dataUrl}" class="img-preview" />`;
  renderResult(id);
}

// ─── Process image ──────────────────────────────
async function processImage(id) {
  const img = images.find(i => i.id === id);
  if (!img || !ocrSession) return;

  const params = {
    h: parseInt(document.getElementById('param-h').value),
    conf: parseFloat(document.getElementById('param-conf').value),
  };

  const src = new Image();
  src.src = img.dataUrl;
  await src.decode();

  const cropW = Math.min(320, src.width);
  const cropH = Math.min(48, src.height * cropW / src.width);

  const c = document.createElement('canvas');
  c.width = cropW; c.height = cropH;
  const ctx = c.getContext('2d');
  ctx.drawImage(src, 0, 0, cropW, cropH);

  const newH = params.h;
  const newW = Math.max(1, Math.round(cropW * (newH / cropH)));
  const d = document.createElement('canvas');
  d.width = newW; d.height = newH;
  d.getContext('2d').drawImage(c, 0, 0, newW, newH);
  const rgba = d.getContext('2d').getImageData(0, 0, newW, newH).data;

  const nchw = new Float32Array(3 * newH * newW);
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const si = (y * newW + x) * 4;
      const di = y * newW + x;
      nchw[di] = rgba[si] / 127.5 - 1.0;
      nchw[1 * newH * newW + di] = rgba[si + 1] / 127.5 - 1.0;
      nchw[2 * newH * newW + di] = rgba[si + 2] / 127.5 - 1.0;
    }
  }

  const tensor = new ort.Tensor('float32', nchw, [1, 3, newH, newW]);
  const results = await ocrSession.run({ x: tensor });
  const output = results[ocrSession.outputNames[0]];
  const data = output.data;
  const timeSteps = output.dims[1], vocabSize = output.dims[2];

  // Decode with character grid
  const chars = [];
  let prev = 0, text = '';
  for (let t = 0; t < timeSteps; t++) {
    let bestIdx = 0, bestVal = -Infinity;
    const offset = t * vocabSize;
    for (let v = 0; v < Math.min(vocabSize, params.conf > 0 ? vocabSize : 0); v++) {
      if (data[offset + v] > bestVal) { bestVal = data[offset + v]; bestIdx = v; }
    }
    const conf = 1 / (1 + Math.exp(-bestVal)); // sigmoid approx

    if (bestIdx !== 0 && bestIdx !== prev && conf >= params.conf) {
      const ch = ppocrKeys ? (ppocrKeys[bestIdx] || '·') : '·';
      text += ch;
      chars.push({ char: ch, conf });
    } else if (bestIdx === 0 || bestIdx === prev) {
      // blank or repeat
    }
    prev = bestIdx;
  }

  const normalized = normalizePlate(text);
  const corrected = corrector.correct(normalized);
  const valid = validatePlate(corrected) && corrected.length === 6;

  img.result = { chars, text, normalized, corrected, valid, params };
  log(`"${text}" → "${corrected}" ${valid ? '✅' : '❌'}`, valid ? 'ok' : 'info');

  if (id === selectedId) renderResult(id);
}

function renderResult(id) {
  const img = images.find(i => i.id === id);
  if (!img || !img.result) { resultArea.innerHTML = '<div class="text-secondary small mt-2">Procesando...</div>'; return; }
  const r = img.result;

  let html = '<div class="result-card">';
  html += `<div class="char-grid mb-1">`;
  for (const ch of r.chars) {
    html += `<span class="char">${ch.char}<span class="conf">${(ch.conf*100).toFixed(0)}%</span></span>`;
  }
  html += `</div>`;
  html += `<div class="d-flex justify-content-between align-items-center">
    <span class="plate ${r.valid ? 'ok' : 'fail'}">${r.corrected || '---'}</span>
    <span class="small ${r.valid ? 'text-success' : 'text-danger'}">${r.valid ? '✅ V&aacute;lida' : '❌ Inv&aacute;lida'}</span>
  </div>`;
  html += `<div class="small text-secondary mt-1">OCR crudo: "${r.text}" &rarr; normalizado: "${r.normalized}"</div>`;
  html += `<div class="small text-secondary">Params: ${JSON.stringify(r.params)}</div>`;
  html += '</div>';
  resultArea.innerHTML = html;
}

// ─── Param changes reprocess ─────────────────────
document.querySelectorAll('#param-h, #param-conf').forEach(el => {
  el.addEventListener('input', () => {
    const id = el.id.replace('param-', '');
    document.getElementById(`val-${id}`).textContent = el.value;
    // Reprocess all images
    images.forEach(img => { img.result = null; });
    images.forEach(img => processImage(img.id));
  });
});
