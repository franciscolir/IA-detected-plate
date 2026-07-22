# IA Detected Plate

Reconocimiento de patentes chilenas en tiempo real desde el navegador, sin servidor, sin backend, 100% offline con PWA.

**[https://franciscolir.github.io/IA-detected-plate/](https://franciscolir.github.io/IA-detected-plate/)**

## Pipeline

```
Cámara → YOLOv8 (640x640) → NMS → Crop → PP-OCRv4 → Corrector → Validación → Placa
                        ↓ (si YOLO no detecta)
                  Fallback por bordes → Crop → PP-OCRv4 → Corrector → Validación
```

| M&oacute;dulo | Tecnolog&iacute;a |
|--------------|-------------------|
| Detecci&oacute;n | YOLOv8n (ONNX) via onnxruntime-web |
| OCR | PP-OCRv4 SVTR-LCNet (ONNX) via onnxruntime-web |
| Fallback | Detecci&oacute;n por proyecci&oacute;n de bordes |
| C&aacute;mara | getUserMedia 1280x720, zoom digital 1-4x |
| Almacenamiento | IndexedDB via Dexie.js (solo configuraci&oacute;n) |

## P&aacute;ginas

| P&aacute;gina | Descripci&oacute;n |
|--------------|--------------------|
| [`index.html`](https://franciscolir.github.io/IA-detected-plate/) | Pantalla principal: c&aacute;mara + placa CSS chilena + detecci&oacute;n en vivo |
| [`test_camera.html`](https://franciscolir.github.io/IA-detected-plate/test_camera.html) | Test completo con c&aacute;mara: todos los par&aacute;metros ajustables, logs en vivo, historial de detecciones con configuraci&oacute;n exportable |
| [`test_ocr.html`](https://franciscolir.github.io/IA-detected-plate/test_ocr.html) | Test OCR con subida de im&aacute;genes: sub&iacute; placas recortadas, muestra caracteres con % de confianza, ajuste de input height y min confidence |
| [`test_detector.html`](https://franciscolir.github.io/IA-detected-plate/test_detector.html) | Test detector con subida de im&aacute;genes: sub&iacute; fotos de autos, YOLO + fallback marcan la placa, estad&iacute;sticas de detecci&oacute;n |

## Par&aacute;metros ajustables

| Par&aacute;metro | Rango | Default | Descripci&oacute;n |
|----------------|-------|---------|-------------------|
| Sensibilidad | 0.05-0.50 | 0.15 | Confianza m&iacute;nima para aceptar detecciones YOLO |
| Streak | 1-10 | 3 | Lecturas consecutivas id&eacute;nticas para confirmar placa |
| No-detect frames | 5-60 | 20 | Frames sin detecci&oacute;n antes de limpiar la placa |
| IoU NMS | 0.10-0.90 | 0.45 | Intersecci&oacute;n sobre uni&oacute;n para filtrar cajas duplicadas |
| Input size | 320-640 | 640 | Tama&ntilde;o de entrada del detector YOLO |
| Fallback edge | 0.03-0.30 | 0.12 | Umbral de gradiente para detecci&oacute;n por bordes |

## Modelos

- `assets/models/yolov8_plate.onnx` — YOLOv8n entrenado con dataset de patentes chilenas a 640x640
- `assets/models/ppocr_rec.onnx` — PP-OCRv4 SVTR-LCNet para reconocimiento de texto
- `assets/models/ppocr_keys.json` — Diccionario CTC de 6625 caracteres

## Placa CSS

La placa chilena se renderiza con CSS puro:
- Fondo blanco con acabado met&aacute;lico (degradado + brillo diagonal)
- Borde negro 4px, esquinas redondeadas 10px
- Tipograf&iacute;a Arial Black, peso 900
- Formato `ABCD · 12` o `AB · 1234` con punto medio separador
- Animaci&oacute;n al detectar

## Entrenamiento

### Google Colab (recomendado)

```bash
# Abrir training/colab_train.ipynb en colab.research.google.com
# Subir datasets ZIP, ejecutar celdas en orden
```

### Local

```bash
cd training
pip install -r requirements.txt
python train_yolo.py
```

El notebook soporta:
- M&uacute;ltiples datasets ZIP (se unifican autom&aacute;ticamente)
- Continuar entrenamiento desde `best.pt` anterior
- Export a ONNX a 640x640
- Descarga autom&aacute;tica de `best.onnx` + `best.pt`

## Desarrollo local

```bash
npx serve .
# o python -m http.server 8000
# Abrir http://localhost:3000
```

## Tecnolog&iacute;as

- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) — 1.27.0
- [Bootstrap](https://getbootstrap.com/) — 5.3.3
- [Dexie.js](https://dexie.org/) — 4.0.8

## Licencia

MIT
