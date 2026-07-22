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
let streakRequired = 3;
let noDetectLimit = 20;

async function init() {
  const resolution = await getConfig('resolution', '1280x720');
  const sensitivity = await getConfig('sensitivity', 0.15);
  streakRequired = await getConfig('streak', 3);
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

  setLoading('Modelos listos', '');
  setTimeout(hideLoading, 500);

  if (!detLoaded && !ocrLoaded) setStatus('Modo demo - coloca los .onnx en assets/models/');
  else setStatus('Listo. Presiona Iniciar');

  setupEvents();
}

function setupEvents() {
  document.getElementById('btn-start').addEventListener('click', start);
  document.getElementById('btn-stop').addEventListener('click', stop);
  document.getElementById('cfg-sensitivity').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('cfg-sensitivity-val').textContent = val.toFixed(2);
    setConfig('sensitivity', val);
    if (detector) detector.confThreshold = val;
  });
  document.getElementById('cfg-streak').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('cfg-streak-val').textContent = val;
    streakRequired = val;
    setConfig('streak', val);
  });
  document.getElementById('cfg-nodetect').addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    document.getElementById('cfg-nodetect-val').textContent = val;
    noDetectLimit = val;
    setConfig('noDetect', val);
  });
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

    const crop = detector.cropPlate(video, box);
    if (crop) {
      const rawText = await ocr.recognize(crop);
      if (rawText) {
        const normalized = normalizePlate(rawText);
        const corrected = corrector.correct(normalized);

        if (validatePlate(corrected) && corrected.length === 6) {
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
        streakCandidate = null; streakCount = 0;
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
