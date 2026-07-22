import * as ort from 'onnxruntime-web';

export class PlateDetector {
  constructor(options = {}) {
    this.session = null;
    this.inputSize = options.inputSize || 640;
    this.confThreshold = options.confThreshold || 0.5;
    this.iouThreshold = options.iouThreshold || 0.45;
    this.mock = true;
    this._transform = null;
    this._smoothBox = null;
    this._debug = true;
  }

  async loadModel(modelPath) {
    try {
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu', 'webgl', 'wasm'],
      });
      this.mock = false;
    } catch (e) {
      console.warn('[Detector] ONNX load failed:', e.message);
    }
    return !this.mock;
  }

  async detect(video, canvasRef) {
    if (this.mock) {
      this._drawOverlay(canvasRef, []);
      return [];
    }

    const input = this._preprocess(video);
    const inputName = this.session.inputNames[0];
    const feeds = {};
    feeds[inputName] = input;
    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    if (this._debug) console.log(`[Detector] model input="${inputName}" output="${this.session.outputNames[0]}" dims=[${output.dims}]`);
    const rawBoxes = this._decodeAll(output.data, output.dims);
    const nmsBoxes = this._nms(rawBoxes);
    const highConf = nmsBoxes.filter(b => b.score >= this.confThreshold);
    const mapped = highConf.map(b => this._toOriginal(b)).filter(b => {
      const w = b.x2 - b.x1;
      const h = b.y2 - b.y1;
      return w >= 30 && h >= 10 && w / h < 12;
    });

    if (this._debug) {
      const confs = rawBoxes.map(b => b.score);
      const maxConf = confs.length ? Math.max(...confs) : 0;
      const over03 = confs.filter(c => c >= 0.3).length;
      console.log(`[Detector] preds=${rawBoxes.length} nms=${nmsBoxes.length} high=${highConf.length} maxConf=${(maxConf*100).toFixed(1)}% over0.3=${over03}`);
    }

    // Fallback + anti-parpadeo
    if (mapped.length === 0) {
      const fallback = this._detectFallback(video);
      if (fallback) {
        if (this._smoothBox) {
          const d = Math.hypot(fallback.x1 - this._smoothBox.x1, fallback.y1 - this._smoothBox.y1);
          const boxSz = Math.hypot(this._smoothBox.x2 - this._smoothBox.x1, this._smoothBox.y2 - this._smoothBox.y1);
          if (d > boxSz * 1.5) this._smoothBox = null;
        }
        if (this._smoothBox) {
          const alpha = 0.35;
          this._smoothBox.x1 += (fallback.x1 - this._smoothBox.x1) * alpha;
          this._smoothBox.y1 += (fallback.y1 - this._smoothBox.y1) * alpha;
          this._smoothBox.x2 += (fallback.x2 - this._smoothBox.x2) * alpha;
          this._smoothBox.y2 += (fallback.y2 - this._smoothBox.y2) * alpha;
          const t = this._transform;
          this._smoothBox.x1 = Math.max(0, this._smoothBox.x1);
          this._smoothBox.y1 = Math.max(0, this._smoothBox.y1);
          this._smoothBox.x2 = Math.min(t.vw, this._smoothBox.x2);
          this._smoothBox.y2 = Math.min(t.vh, this._smoothBox.y2);
          this._smoothBox.score = Math.max(0.5, fallback.score);
          this._smoothBox.hangCount = 0;
        } else {
          this._smoothBox = { x1: fallback.x1, y1: fallback.y1, x2: fallback.x2, y2: fallback.y2, score: fallback.score, hangCount: 0 };
        }
        mapped.push({ x1: this._smoothBox.x1, y1: this._smoothBox.y1, x2: this._smoothBox.x2, y2: this._smoothBox.y2, score: this._smoothBox.score });
        if (this._debug) console.log(`[Detector] fallback: ${(this._smoothBox.score*100).toFixed(1)}%`);
      } else if (this._smoothBox) {
        this._smoothBox.hangCount++;
        if (this._smoothBox.hangCount > 30) {
          this._smoothBox = null;
        } else {
          this._smoothBox.score = Math.max(0.2, this._smoothBox.score * 0.96);
          mapped.push({ x1: this._smoothBox.x1, y1: this._smoothBox.y1, x2: this._smoothBox.x2, y2: this._smoothBox.y2, score: this._smoothBox.score });
        }
      }
    } else {
      this._smoothBox = null;
    }

    this._drawOverlay(canvasRef, mapped, rawBoxes);
    return mapped;
  }

  _vw(v) { return v.videoWidth || v.width || 0; }
  _vh(v) { return v.videoHeight || v.height || 0; }

  _preprocess(video) {
    const vw = this._vw(video);
    const vh = this._vh(video);
    const maxSize = Math.max(vw, vh);
    const isz = this.inputSize;
    const scale = isz / maxSize;
    const ox = (isz - vw * scale) / 2;
    const oy = (isz - vh * scale) / 2;

    this._transform = { scale, ox, oy, vw, vh };

    const canvas = document.createElement('canvas');
    canvas.width = isz;
    canvas.height = isz;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, isz, isz);
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
    return new ort.Tensor('float32', data, [1, 3, isz, isz]);
  }

  _decodeAll(data, dims) {
    const numPreds = dims[2];
    const raw = [];
    const maxCx = Math.max(...data.slice(0, Math.min(numPreds, 100)));
    const normalized = maxCx < 2;
    const scale = normalized ? this.inputSize : 1;
    if (this._debug) console.log(`[Detector] decode: maxCx=${maxCx.toFixed(4)} normalized=${normalized} scale=${scale}`);
    for (let i = 0; i < numPreds; i++) {
      const cx = data[i];
      const cy = data[numPreds + i];
      const w = data[2 * numPreds + i];
      const h = data[3 * numPreds + i];
      const conf = data[4 * numPreds + i];
      if (conf < 0.05) continue;
      raw.push({
        x1: (cx - w / 2) * scale,
        y1: (cy - h / 2) * scale,
        x2: (cx + w / 2) * scale,
        y2: (cy + h / 2) * scale,
        score: conf,
      });
    }
    return raw;
  }

  _nms(detections) {
    detections.sort((a, b) => b.score - a.score);
    const picked = new Array(detections.length).fill(false);
    const result = [];
    for (let i = 0; i < detections.length; i++) {
      if (picked[i]) continue;
      result.push(detections[i]);
      for (let j = i + 1; j < detections.length; j++) {
        if (picked[j]) continue;
        if (this._iou(detections[i], detections[j]) >= this.iouThreshold) {
          picked[j] = true;
        }
      }
    }
    return result;
  }

  _iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
  }

  _toOriginal(box) {
    const t = this._transform;
    return {
      x1: Math.max(0, (box.x1 - t.ox) / t.scale),
      y1: Math.max(0, (box.y1 - t.oy) / t.scale),
      x2: Math.min(t.vw, (box.x2 - t.ox) / t.scale),
      y2: Math.min(t.vh, (box.y2 - t.oy) / t.scale),
      score: box.score,
    };
  }

  _drawOverlay(canvas, final, raw = null) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Filtered low-confidence boxes (semi-transparent)
    if (raw) {
      for (const d of raw) {
        if (d.score < 0.2) continue;
        const b = this._toOriginal(d);
        const iw = b.x2 - b.x1;
        const ih = b.y2 - b.y1;
        if (iw < 20 || ih < 8) continue;
        ctx.strokeStyle = 'rgba(255,200,0,0.25)';
        ctx.lineWidth = 1;
        ctx.strokeRect(b.x1, b.y1, iw, ih);
      }
    }

    // High-confidence boxes (solid yellow)
    for (const d of final) {
      const iw = d.x2 - d.x1;
      const ih = d.y2 - d.y1;
      if (iw < 2 || ih < 2) continue;
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 3;
      ctx.strokeRect(d.x1, d.y1, iw, ih);
      ctx.fillStyle = '#ffcc00';
      ctx.font = 'bold 13px monospace';
      ctx.fillText(`${(d.score * 100).toFixed(0)}%`, d.x1, d.y1 - 5);
    }

    // Center crosshair
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(cx - 30, cy); ctx.lineTo(cx + 30, cy);
    ctx.moveTo(cx, cy - 30); ctx.lineTo(cx, cy + 30);
    ctx.stroke();
    ctx.setLineDash([]);

    // Status
    if (final.length === 0) {
      ctx.fillStyle = 'rgba(255,50,50,0.6)';
      ctx.font = '14px sans-serif';
      ctx.fillText('Sin deteccion', 10, 20);
    }
  }

  _detectFallback(video) {
    const vw = this._vw(video);
    const vh = this._vh(video);
    if (vw < 50 || vh < 50) return null;

    const canvas = document.createElement('canvas');
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const imageData = ctx.getImageData(0, 0, vw, vh);
    const data = imageData.data;

    const gray = new Uint8Array(vw * vh);
    for (let i = 0; i < vw * vh; i++) {
      const pi = i << 2;
      gray[i] = (data[pi] * 0.299 + data[pi + 1] * 0.587 + data[pi + 2] * 0.114) | 0;
    }

    // Horizontal gradient magnitude (edge detection)
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

    const edgeThresh = maxGrad * 0.12;

    // Row edge density
    const rowCount = new Uint16Array(vh);
    for (let y = 0; y < vh; y++) {
      let c = 0;
      for (let x = 0; x < vw; x++) { if (grad[y * vw + x] > edgeThresh) c++; }
      rowCount[y] = c;
    }

    let maxR = 0;
    for (let y = 0; y < vh; y++) { if (rowCount[y] > maxR) maxR = rowCount[y]; }
    if (maxR < 3) return null;

    const rowMin = Math.max(3, maxR * 0.15);
    let yS = -1, yE = -1;
    for (let y = 0; y < vh; y++) {
      if (rowCount[y] > rowMin) { if (yS === -1) yS = y; yE = y; }
    }
    if (yS === -1 || yE - yS < 3) return null;

    yS = Math.max(0, yS - 3);
    yE = Math.min(vh, yE + 3);
    const bandH = yE - yS;

    // Column edge density within band
    const colCount = new Uint16Array(vw);
    for (let x = 0; x < vw; x++) {
      let c = 0;
      for (let y = yS; y < yE; y++) { if (grad[y * vw + x] > edgeThresh) c++; }
      colCount[x] = c;
    }

    let maxC = 0;
    for (let x = 0; x < vw; x++) { if (colCount[x] > maxC) maxC = colCount[x]; }
    if (maxC < 3) return null;

    const colMin = Math.max(3, maxC * 0.18);
    let xS = -1, xE = -1;
    for (let x = 0; x < vw; x++) {
      if (colCount[x] > colMin) { if (xS === -1) xS = x; xE = x; }
    }
    if (xS === -1 || xE - xS < 10) return null;

    xS = Math.max(0, xS - 4);
    xE = Math.min(vw, xE + 4);

    const pw = xE - xS;
    const ph = yE - yS;
    if (pw < 10 || ph < 3) return null;
    if (pw > vw * 0.95 || ph > vh * 0.95) return null;

    const aspect = pw / ph;
    const aspectScore = Math.max(0, 1 - Math.abs(aspect - 3.5) / 5);
    const densityScore = Math.min(1, (maxR + maxC) / (vw * 0.05 + vh * 0.5));
    const score = 0.2 + 0.4 * aspectScore + 0.4 * densityScore;

    return { x1: xS, y1: yS, x2: xE, y2: yE, score: Math.min(0.7, score) };
  }

  cropPlate(video, box) {
    const w = Math.round(box.x2 - box.x1);
    const h = Math.round(box.y2 - box.y1);
    if (w < 4 || h < 4) return null;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, box.x1, box.y1, w, h, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  }
}
