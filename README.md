# IA Detected Plate

Reconocimiento de patentes chilenas en tiempo real desde el navegador, sin servidor, sin backend, 100% offline.

**[https://franciscolir.github.io/IA-detected-plate/](https://franciscolir.github.io/IA-detected-plate/)**

## Pipeline

```
Camara -> YOLOv8 (640x640) -> NMS -> Crop -> PP-OCRv4 -> Corrector -> Validacion -> Placa
                         |
                         v (si YOLO no detecta)
                   Fallback por bordes -> Crop -> PP-OCRv4 -> Corrector -> Validacion
```

Optimizaciones de rendimiento activas:

- **Frame skip YOLO**: el detector corre cada 2 frames (desktop) o cada 3 (movil). Entre frames se reutiliza la ultima caja detectada.
- **OCR skip**: cuando el box es estable (no se mueve mas del 15%), el OCR corre cada 3 frames (desktop) o cada 2 (movil).
- **Canvas reuse**: los canvases de preprocessing, fallback y crop se reutilizan entre frames para reducir garbage collection.
- **Warmup**: inferencia dummy al cargar para forzar JIT y asignacion de memoria antes del primer frame real.

## Modo movil

La app detecta el dispositivo automaticamente y ajusta parametros:

| Parametro | Desktop | Movil |
|-----------|---------|-------|
| Resolucion camara | 1280x720 | 640x480 |
| YOLO frame skip | 2 | 3 |
| OCR skip interval | 3 | 2 |

El badge `[DESK]` o `[MOVIL]` en el titulo indica el modo activo.

## Modulos

| Modulo | Tecnologia |
|--------|------------|
| Deteccion | YOLOv8n (ONNX) via onnxruntime-web |
| OCR | PP-OCRv4 SVTR-LCNet (ONNX) via onnxruntime-web |
| Fallback | Deteccion por proyeccion de bordes (Sobel horizontal) |
| Camara | getUserMedia, facingMode environment |
| Almacenamiento | IndexedDB via Dexie.js (solo configuracion) |
| UI | Bootstrap 5.3.3, CSS puro para placa chilena |

## Paginas

| Pagina | Descripcion |
|--------|-------------|
| `index.html` | Pantalla principal: camara + placa CSS chilena + deteccion en vivo + panel de diagnostico |
| `test_camera.html` | Test completo con camara: todos los parametros ajustables, logs en vivo, historial de detecciones |
| `test_ocr.html` | Test OCR con subida de imagenes: placas recortadas, caracteres con % de confianza |
| `test_detector.html` | Test detector con subida de imagenes: YOLO + fallback marcan la placa, estadisticas |

## Parametros

| Parametro | Rango | Default | Descripcion |
|-----------|-------|---------|-------------|
| Sensibilidad | 0.05-0.50 | 0.15 | Confianza minima para aceptar detecciones YOLO |
| Streak | 1-10 | 2 | Lecturas consecutivas identicas para confirmar placa |
| No-detect frames | 5-60 | 20 | Frames sin deteccion antes de limpiar la placa |
| IoU NMS | 0.10-0.90 | 0.45 | Interseccion sobre union para filtrar cajas duplicadas |
| Fallback edge | 0.03-0.30 | 0.08 | Umbral de gradiente para deteccion por bordes |
| Fallback row factor | 0.05-0.50 | 0.10 | % del maximo de bordes por fila |
| Fallback col factor | 0.05-0.50 | 0.12 | % del maximo de bordes por columna |
| Fallback min height | 2-20 | 4 | Altura minima en pixels para aceptar una deteccion del fallback |

Los parametros se ajustan desde `test_camera.html`. La pantalla principal usa los valores por defecto.

## Modelos

- `assets/models/yolov8_plate.onnx` -- YOLOv8n entrenado con dataset de patentes chilenas a 640x640
- `assets/models/ppocr_rec.onnx` -- PP-OCRv4 SVTR-LCNet para reconocimiento de texto
- `assets/models/ppocr_keys.json` -- Diccionario CTC de 6625 caracteres

Sin modelos, el sistema funciona en modo MOCK (demo UI).

## Panel de diagnostico

La pantalla principal muestra 3 cards con informacion del sistema:

**App / Modelos** -- estado del detector, OCR, predicciones, confianza, texto OCR crudo, placa corregida, streak, FPS.

**Dispositivo** -- RAM, nucleos CPU, GPU (WebGL renderer), plataforma.

**Navegador** -- User Agent, soporte WebGL, WebGPU, WASM, resolucion de pantalla, device pixel ratio.

## Placa CSS

La placa chilena se renderiza con CSS puro:
- Fondo blanco con acabado metalico (degradado + brillo diagonal)
- Borde negro 4px, esquinas redondeadas 10px
- Tipografia Arial Black, peso 900
- Formato `ABCD 12` o `AB 1234` con espacio separador
- Animacion al detectar

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
- Multiples datasets ZIP (se unifican automaticamente)
- Continuar entrenamiento desde `best.pt` anterior
- Export a ONNX a 640x640
- Descarga automatica de `best.onnx` + `best.pt`

## Desarrollo local

```bash
npx serve .
# o
python -m http.server 8000
```

## Estructura del proyecto

```
index.html              Pantalla principal
test_camera.html        Test completo con camara
test_ocr.html           Test OCR con imagenes
test_detector.html      Test detector con imagenes
assets/css/             CSS por pagina
assets/js/              JS modular (app, detector, ocr, corrector, validator, camera, database)
assets/models/          Modelos ONNX + diccionario CTC
training/               Notebook Colab + script local
```

## Tecnologias

- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.27.0
- [Bootstrap](https://getbootstrap.com/) 5.3.3
- [Dexie.js](https://dexie.org/) 4.0.8

## Licencia

MIT