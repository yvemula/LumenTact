## COCO/YOLO Setup Guide from Scratch

Get coco2017 dataset from the internet (big download).

1. Ensure you have Python3.11 since you need to have PyTorch. PyTorch doesnt work on python 3.13+/newer versions at the time of creating this doc. Search the python version compatable with PyTorch at the time.

2. Create a Virtual Environment for Python3.11

3. Get PyTorch, pip install all the torch stuff

4. Ensure successful download ``` python -c "import torch; print(torch.__version__)"  ```

5. Get Ultralytics ``` pip install ultralyitcs ```

6. Check if everything working with ``` python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded!')" ```. Should see the success message in terminal. If you get any incompatabilites just force-install downgrade the version to what is compatible.

### Run This to Test Success
```
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
```

This should output an image with bounding boxes and labels and output a summary of confidence levels/other numbers




