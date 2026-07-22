# AGENTS.md - IA Detected Plate

## Comandos de verificacion

```bash
# Servidor local
npx serve .

# o con Python
python -m http.server 8000

# Verificar JS syntax
node -c assets/js/app.js
node -c assets/js/detector.js
```

## Dependencias CDN

| Paquete                         | Version | Motivo                        |
| ------------------------------- | ------- | ----------------------------- |
| onnxruntime-web                 | 1.27.0  | Motor inferencia ONNX (detector + OCR) |
| Bootstrap                       | 5.3.3   | UI                            |

## Paginas

| Pagina           | Descripcion                          |
| ---------------- | ------------------------------------ |
| index.html       | Camara + placa CSS (principal)       |
| test_camera.html | Test completo con todos los parametros |
| test_ocr.html    | Test especifico de OCR                |
| test_detector.html | Test de detector YOLO + fallback     |

## Modelos

- `assets/models/yolov8_plate.onnx` - YOLOv8 para deteccion de patentes
- `assets/models/ppocr_rec.onnx` - PP-OCRv4 SVTR-LCNet para OCR
- `assets/models/ppocr_keys.json` - Diccionario de caracteres CTC (6625 chars)

Sin modelos, el sistema funciona en modo MOCK (demo UI).

## Entrenamiento

Ver `training/colab_train.ipynb` para entrenar en Google Colab.
Ver `training/train_yolo.py` para entrenar localmente.

```bash
cd training
pip install -r requirements.txt
python train_yolo.py
```