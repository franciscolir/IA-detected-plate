# AGENTS.md - IA Detected Plate

## Comandos de verificacion

```bash
# Servidor local
npx serve .

# o con Python
python -m http.server 8000

# Verificar JS syntax (todos los modulos)
node -c assets/js/app.js
node -c assets/js/detector.js
node -c assets/js/ocr.js
node -c assets/js/corrector.js
node -c assets/js/validator.js
node -c assets/js/camera.js
node -c assets/js/database.js
node -c assets/js/test_camera.js
node -c assets/js/test_detector.js
node -c assets/js/test_ocr.js
```

## Dependencias CDN

| Paquete | Version | Motivo |
|---------|---------|--------|
| onnxruntime-web | 1.27.0 | Motor de inferencia ONNX (detector + OCR) |
| Bootstrap | 5.3.3 | UI (grid, tabs, offcanvas, botones) |
| Dexie.js | 4.0.8 | IndexedDB (persistencia de configuracion) |

## Estructura del proyecto

```
index.html              Pantalla principal
test_camera.html        Test completo con camara (todos los parametros ajustables)
test_ocr.html           Test OCR con subida de imagenes
test_detector.html      Test detector con subida de imagenes
sw.js                   Service Worker (cache offline)
manifest.json           PWA manifest
assets/css/             CSS por pagina (app, index, test_camera, test_detector, test_ocr)
assets/js/              JS modular
assets/models/          Modelos ONNX + diccionario CTC (gitignored)
training/               Notebook Colab + script local de entrenamiento
```

## Modulos JS

| Archivo | Clase/Funcion | Descripcion |
|---------|---------------|-------------|
| app.js | - | Orquestador principal: loop, streak, sys panel, display |
| detector.js | PlateDetector | YOLOv8 + fallback por proyeccion + frame skip + canvas reuse |
| ocr.js | PlateOCR | PP-OCRv4 + preprocessing canvas + CTC decode + warmup |
| corrector.js | Corrector | Correccion por zona (letras vs numeros) |
| validator.js | validatePlate, normalizePlate | Validacion de formatos de patentes chilenas |
| camera.js | Camera | getUserMedia + torch |
| database.js | getConfig, setConfig | Dexie/IndexedDB |
| test_camera.js | - | Pipeline completo autocontenido para test |
| test_detector.js | - | Solo detector para test con imagenes |
| test_ocr.js | - | Solo OCR para test con imagenes |

## Modelos

- `assets/models/yolov8_plate.onnx` - YOLOv8n entrenado con dataset de patentes chilenas (640x640, 6.1 MB)
- `assets/models/ppocr_rec.onnx` - PP-OCRv4 SVTR-LCNet para OCR (10.8 MB)
- `assets/models/ppocr_keys.json` - Diccionario CTC de 6625 caracteres

Sin modelos, el sistema funciona en modo MOCK (demo UI).

## Optimizaciones de rendimiento

| Optimizacion | Desktop | Movil |
|--------------|---------|-------|
| YOLO frame skip | cada 2 | cada 3 |
| OCR skip (box estable) | cada 3 | cada 2 |
| Resolucion camara | 1280x720 | 640x480 |
| Canvas reuse | si | si |
| Warmup JIT | si (timeout 8s) | si (timeout 8s) |

## Entrenamiento

Ver `training/colab_train.ipynb` para entrenar en Google Colab.
Ver `training/train_yolo.py` para entrenar localmente.

```bash
cd training
pip install -r requirements.txt
python train_yolo.py
```