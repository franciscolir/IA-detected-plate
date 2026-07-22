# Modelos IA

Este directorio debe contener los modelos `.tflite` para inferencia.

## Modelos requeridos

### 1. YOLOX-Tiny (Detector de patentes)
- **Archivo**: `yolox_plate.tflite`
- **Tamaño**: <10 MB
- **Input**: 416x416x3 (RGB, normalizado 0-1)
- **Output**: Boxes, scores, classes

**Obtención**:
- Entrenar YOLOX-Tiny en dataset de patentes chilenas
- Exportar a TFLite con TensorFlow Lite Converter
- Dataset sugerido: CCPD,CustomLabel (mínimo 5000 imágenes)

### 2. PP-OCRv4 Recognition (OCR)
- **Archivo**: `ppocr_rec.tflite`
- **Tamaño**: ~10 MB
- **Input**: 48x320x3 (RGB, normalizado 0-1)
- **Output**: Probabilidades CTC

**Obtención**:
- Descargar modelo oficial PaddleOCR v4
- Convertir Paddle → ONNX → TFLite
- Herramienta: `paddle2onnx` + `onnx-tf`

## Fallback

Si los modelos no están presentes, el sistema usa **mock inference** para desarrollo UI.

## Referencia

Se descargó `ch_PP-OCRv4_rec_infer.tar` (formato Paddle, no TFLite) como referencia.