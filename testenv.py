from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')

# Run inference on an image
results = model('data/coco2017/val2017/000000000139.jpg')

# Each item in `results` is a Result object, so take the first one
result = results[0]

# Display the detection results visually
result.show()

# Optional: print detection summary
for box in result.boxes:
    cls = model.names[int(box.cls)]
    conf = float(box.conf)
    xyxy = box.xyxy[0].tolist()
    print(f"{cls:<15} conf={conf:.2f} box={xyxy}")
