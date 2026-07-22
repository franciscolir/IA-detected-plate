import * as ort from 'onnxruntime-web';

export class PlateOCR {
  constructor() {
    this.session = null;
    this.chars = [''];
    this.inputHeight = 48;
    this.mock = true;
  }

  async loadModel(modelPath) {
    try {
      const [session, chars] = await Promise.all([
        ort.InferenceSession.create(modelPath, {
          executionProviders: ['webgpu', 'webgl', 'wasm'],
        }),
        fetch('assets/models/ppocr_keys.json').then(r => r.json()),
      ]);
      this.session = session;
      this.chars = chars;
      this.mock = false;
    } catch (e) {
      console.warn('[OCR] ONNX load failed:', e.message);
      this.mock = true;
    }
    return !this.mock;
  }

  async recognize(imageData) {
    if (this.mock) return '';

    const { data, shape } = this._preprocess(imageData);
    const ortTensor = new ort.Tensor('float32', data, shape);
    const results = await this.session.run({ x: ortTensor });
    const output = results[this.session.outputNames[0]];
    return this._ctcGreedyDecode(output.data, output.dims);
  }

  _preprocess(imageData) {
    const oldW = imageData.width;
    const oldH = imageData.height;
    const newH = this.inputHeight;
    const newW = Math.max(1, Math.round(oldW * (newH / oldH)));

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = oldW;
    srcCanvas.height = oldH;
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

    const dstCanvas = document.createElement('canvas');
    dstCanvas.width = newW;
    dstCanvas.height = newH;
    dstCanvas.getContext('2d').drawImage(srcCanvas, 0, 0, newW, newH);

    const resized = dstCanvas.getContext('2d').getImageData(0, 0, newW, newH);
    const pixels = resized.data;

    const nchw = new Float32Array(3 * newH * newW);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const srcIdx = (y * newW + x) * 4;
        const dstIdx = y * newW + x;
        nchw[dstIdx] = pixels[srcIdx] / 127.5 - 1.0;
        nchw[1 * newH * newW + dstIdx] = pixels[srcIdx + 1] / 127.5 - 1.0;
        nchw[2 * newH * newW + dstIdx] = pixels[srcIdx + 2] / 127.5 - 1.0;
      }
    }

    return { data: nchw, shape: [1, 3, newH, newW] };
  }

  _ctcGreedyDecode(data, dims) {
    const timeSteps = dims[1];
    const vocabSize = dims[2];
    let prev = 0;
    let result = '';
    for (let t = 0; t < timeSteps; t++) {
      let bestIdx = 0;
      let bestVal = -Infinity;
      const offset = t * vocabSize;
      for (let v = 0; v < vocabSize; v++) {
        if (data[offset + v] > bestVal) {
          bestVal = data[offset + v];
          bestIdx = v;
        }
      }
      if (bestIdx !== 0 && bestIdx !== prev) {
        result += this.chars[bestIdx] || '';
      }
      prev = bestIdx;
    }
    return result;
  }

  dispose() {
    if (this.session) {
      this.session = null;
    }
  }
}
