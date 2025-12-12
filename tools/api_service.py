"""
FastAPI service for YOLO inference.

Run:
  ./venv/bin/python -m uvicorn tools.api_service:app --host 0.0.0.0 --port 8000

Env vars:
  YOLO_WEIGHTS=path/to/best.pt
  YOLO_IMGSZ=1280
  YOLO_CONF=0.25
  YOLO_DEVICE=0
"""
import io
import os
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO


WEIGHTS = os.getenv("YOLO_WEIGHTS", "runs/detect/sanpo_yolo/weights/best.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
CONF = float(os.getenv("YOLO_CONF", "0.25"))
DEVICE = os.getenv("YOLO_DEVICE", "0")

model = YOLO(WEIGHTS)
class_names = model.names

app = FastAPI(title="YOLO Inference API", version="1.0.0")


def format_response(result) -> List[dict]:
    preds = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0]) if box.cls is not None else -1
        score = float(box.conf[0]) if box.conf is not None else 0.0
        label = class_names.get(cls, str(cls)) if isinstance(class_names, dict) else (
            class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        )
        preds.append({"cls": cls, "label": label, "conf": score, "bbox": [x1, y1, x2, y2]})
    return preds


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    result = model.predict(image, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
    return JSONResponse({"predictions": format_response(result)})


if __name__ == "__main__":
    uvicorn.run("tools.api_service:app", host="0.0.0.0", port=8000, reload=False)
