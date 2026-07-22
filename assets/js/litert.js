// litert.js - Runtime LiteRT con carga ESM desde CDN + fallback mock
let litertCore = null;
let litertInterop = null;
let litertInitialized = false;
const BACKEND_ORDER = ['webgpu', 'webnn', 'wasm', 'cpu'];
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';

let _loadPromise = null;
let _tf = null;

async function ensureTf() {
  if (_tf) return _tf;
  _tf = await import('@tensorflow/tfjs');
  await import('@tensorflow/tfjs-backend-webgpu');
  return _tf;
}

async function ensureLiteRt() {
  if (litertInitialized) return { core: litertCore, interop: litertInterop, tf: _tf };
  if (_loadPromise) return _loadPromise;
  _loadPromise = _initLiteRt();
  return _loadPromise;
}

async function _initLiteRt() {
  const tf = await ensureTf();
  const urls = [
    'https://esm.sh/@litertjs/core@2.5.3',
    'https://esm.sh/@litertjs/tfjs-interop@2.5.3'
  ];
  try {
    [litertCore, litertInterop] = await Promise.all(urls.map(u => import(u)));
    litertInitialized = true;
    return { core: litertCore, interop: litertInterop, tf };
  } catch (e) {
    console.warn('[LiteRT] CDN ESM load failed:', e.message);
    litertCore = null;
    litertInterop = null;
    litertInitialized = true;
    return { core: null, interop: null, tf };
  }
}

export class LiteRT {
  constructor() {
    this.net = null;
    this.backend = null;
    this.modelPath = null;
    this.inputShape = null;
    this.useMock = false;
  }

  async loadModel(modelPath) {
    this.modelPath = modelPath;
    const { core, interop, tf } = await ensureLiteRt();

    if (core && interop) {
      return await this._realLoad(modelPath, core, interop, tf);
    }
    return this._mockLoad();
  }

  async _realLoad(modelPath, core, interop, tf) {
    try {
      for (const backend of BACKEND_ORDER) {
        try {
          if (backend === 'webgpu') {
            await tf.setBackend('webgpu');
            await core.loadLiteRt(WASM_BASE);
            const device = core.getWebGpuDevice();
            tf.removeBackend('webgpu');
            const { WebGPUBackend } = await import('@tensorflow/tfjs-backend-webgpu');
            tf.registerBackend('webgpu', () => new WebGPUBackend(device, device.adapterInfo));
            await tf.setBackend('webgpu');
          } else if (backend === 'wasm') {
            await tf.setBackend('wasm');
          } else {
            await tf.setBackend('cpu');
          }

          const resp = await fetch(modelPath);
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const bytes = new Uint8Array(await resp.arrayBuffer());
          this.net = await core.loadAndCompile(bytes, { accelerator: backend === 'webgpu' ? 'webgpu' : 'cpu' });
          const inputInfo = this.net.getInputDetails()[0];
          this.inputShape = Array.from(inputInfo.shape);
          this.backend = backend;

          const dummy = tf.ones(this.inputShape);
          const [warmResult] = await interop.runWithTfjsTensors(this.net, [dummy]);
          tf.dispose([warmResult, dummy]);

          console.log(`[LiteRT] Loaded ${modelPath} on ${backend}`);
          return true;
        } catch (e) {
          console.warn(`[LiteRT] Backend ${backend} failed:`, e.message);
        }
      }
      throw new Error('No backend available');
    } catch (e) {
      console.warn('[LiteRT] Real inference unavailable:', e.message);
      this.useMock = true;
      return this._mockLoad();
    }
  }

  _mockLoad() {
    this.useMock = true;
    this.backend = 'MOCK';
    if (this.modelPath.includes('yolox')) {
      this.inputShape = [1, 3, 416, 416];
    }
    console.log(`[LiteRT] Mock mode for ${this.modelPath}`);
    return true;
  }

  async warmup() {
    if (this.useMock) return;
    const { interop, tf } = await ensureLiteRt();
    if (this.net && interop) {
      const dummy = tf.ones(this.inputShape);
      const [warmResult] = await interop.runWithTfjsTensors(this.net, [dummy]);
      tf.dispose([warmResult, dummy]);
    }
  }

  async runInference(inputTensor) {
    if (this.useMock) return this._mockOutput();
    const { interop, tf } = await ensureLiteRt();
    if (!this.net || !interop) throw new Error('Model not loaded');
    const [result] = await interop.runWithTfjsTensors(this.net, [inputTensor]);
    return result;
  }

  _mockOutput() {
    return null;
  }

  dispose() {
    if (this.net) {
      this.net = null;
      this.backend = null;
      this.inputShape = null;
    }
  }
}

export async function createLiteRT() {
  return new LiteRT();
}