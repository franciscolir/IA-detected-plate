import os
import zipfile
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────
ZIP_FILE = "Patentes Chile.v4i.yolov8.zip"
MODEL_SIZE = "yolov8n.pt"        # nano (menos precision, mas rapido)
# MODEL_SIZE = "yolov8s.pt"      # small (mejor precision, mas lento)
IMGSZ = 640
EPOCHS = 50
BATCH = 16
PATIENCE = 15
DEVICE = "cuda" if os.system("python -c 'import torch; print(torch.cuda.is_available())'") else "cpu"
# ──────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"

def extract_dataset():
    zip_path = ROOT / ZIP_FILE
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encuentra {zip_path}")
    print(f"[1/4] Extrayendo {ZIP_FILE}...")
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(DATASET_DIR))
    # Fix data.yaml paths (zip tiene relative paths incorrectos)
    data_yaml = DATASET_DIR / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    data["train"] = str((DATASET_DIR / "train" / "images").resolve())
    data["val"] = str((DATASET_DIR / "valid" / "images").resolve())
    data["test"] = str((DATASET_DIR / "test" / "images").resolve())
    data_yaml.write_text(yaml.dump(data))
    return data_yaml

def train(data_yaml):
    print(f"[2/4] Entrenando (device={DEVICE})...")
    model = YOLO(MODEL_SIZE)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        device=DEVICE,
        workers=4,
        amp=True,
        name="plate_detector",
        exist_ok=True,
    )
    return ROOT / "runs" / "detect" / "plate_detector" / "weights" / "best.pt"

def export_onnx(weights_path):
    print(f"[3/4] Exportando a ONNX (imgsz={IMGSZ}, half=True)...")
    model = YOLO(str(weights_path))
    model.export(format="onnx", imgsz=IMGSZ, half=True)
    return weights_path.parent / "best.onnx"

def deploy(onnx_path):
    dest = ROOT.parent / "assets" / "models" / "yolov8_plate.onnx"
    shutil.copy2(str(onnx_path), str(dest))
    print(f"[4/4] Copiado a: {dest}")

def count_images(data_yaml):
    data = yaml.safe_load(data_yaml.read_text())
    for split in ("train", "val", "test"):
        img_dir = Path(data[split])
        count = len(list(img_dir.glob("*.*")))
        print(f"  {split}: {count} imagenes")

if __name__ == "__main__":
    data_yaml = extract_dataset()
    count_images(data_yaml)
    best_pt = train(data_yaml)
    onnx_path = export_onnx(best_pt)
    deploy(onnx_path)
    print("[LISTO] Modelo entrenado y desplegado en assets/models/")
