import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

MODEL_ONNX = Path(__file__).parent.parent / "assets" / "models" / "yolov8_plate.onnx"

def test_with_ultralytics():
    model = YOLO(str(MODEL_ONNX))
    results = model("test_images/", imgsz=416)
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = b.conf[0].item()
                print(f"  [{r.path}] ({x1},{y1})-({x2},{y2}) conf={conf:.3f}")

def test_with_onnxruntime():
    session = ort.InferenceSession(str(MODEL_ONNX))
    input_name = session.get_inputs()[0].name
    img = cv2.imread("test.jpg")
    if img is None:
        print("  Pon una imagen test.jpg en training/")
        return
    resized = cv2.resize(img, (416, 416))
    blob = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    outputs = session.run(None, {input_name: blob})
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output[0]: {outputs[0][0, :, :5]}")

if __name__ == "__main__":
    if not MODEL_ONNX.exists():
        print(f"[!] No se encuentra {MODEL_ONNX}")
        print("    Entrena primero con train_yolo.py o copia un .onnx")
        sys.exit(1)

    print("=== Test con Ultralytics ===")
    test_with_ultralytics()

    print("\n=== Test con ONNX Runtime (como en el browser) ===")
    test_with_onnxruntime()
