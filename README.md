# IA Detected Plate

Reconocimiento de patentes chilenas en tiempo real desde el navegador, sin servidor, sin backend, 100% offline con PWA.

## Demo

**[https://franciscolir.github.io/IA-detected-plate/](https://franciscolir.github.io/IA-detected-plate/)**

## Cómo funciona

| M&oacute;dulo | Tecnolog&iacute;a |
|--------------|-------------------|
| Detecci&oacute;n | YOLOv8n (ONNX) via onnxruntime-web |
| OCR | PP-OCRv4 SVTR-LCNet (ONNX) via onnxruntime-web |
| Fallback | Detecci&oacute;n por proyecci&oacute;n de bordes (cuando YOLO no detecta) |
| C&aacute;mara | getUsermedia 1280x720 |
| Almacenamiento | IndexedDB via Dexie.js (solo configuraci&oacute;n) |

### Pipeline

```
Cámara → YOLOv8 (640x640) → NMS → Crop → PP-OCRv4 → Corrector → Validación
                        ↓ (si no detecta)
                  Fallback por bordes → Crop → PP-OCRv4 → Corrector → Validación
```

## P&aacute;ginas

| P&aacute;gina | Descripci&oacute;n |
|--------------|--------------------|
| `index.html` | Pantalla principal: c&aacute;mara + placa CSS + inicio/detenci&oacute;n |
| `test_camera.html` | Test completo con par&aacute;metros ajustables y logs |
| `test_ocr.html` | Test espec&iacute;fico de OCR con grid de caracteres |
| `test_detector.html` | Test de detector con estad&iacute;sticas YOLO y fallback |

## Modelos

- `assets/models/yolov8_plate.onnx` — YOLOv8n entrenado con dataset de patentes chilenas a 640x640
- `assets/models/ppocr_rec.onnx` — PP-OCRv4 SVTR-LCNet para reconocimiento de texto
- `assets/models/ppocr_keys.json` — Diccionario CTC de 6625 caracteres

## Entrenamiento

Ver `training/colab_train.ipynb` para entrenar en Google Colab.

```bash
cd training
pip install -r requirements.txt
python train_yolo.py
```

## Desarrollo local

```bash
npx serve .
# o
python -m http.server 8000
# Abrir http://localhost:3000
```

## Tecnolog&iacute;as

- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) — 1.27.0
- [Bootstrap](https://getbootstrap.com/) — 5.3.3
- [Dexie.js](https://dexie.org/) — 4.0.8

## Licencia

MIT
