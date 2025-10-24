from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # nano model to start

model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,          # M1 GPU RAM is limited; use 4–8
    workers=2,        # macOS: keep workers low
    device="mps",     # <<--- use Apple Metal GPU
    amp=True          # mixed precision; faster on MPS
)
