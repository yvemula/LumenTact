from ultralytics import YOLO

HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "chair", "bench", "traffic light", "stop sign"
}

class TinyDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35, imgsz=640):
        self.model = YOLO(model_name)
        self.conf  = conf
        self.imgsz = imgsz
    def __call__(self, frame_bgr):
        # returns list of dict {cls, conf, bbox(cx,cy,w,h)}
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        dets=[]
        for r in results:
            for b in r.boxes:
                cls_id   = int(b.cls.item())
                cls_name = r.names[cls_id]
                if cls_name not in HAZARD_CLASSES: continue
                conf     = float(b.conf.item())
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cx = (x1+x2)/2.0 / r.orig_shape[1]
                cy = (y1+y2)/2.0 / r.orig_shape[0]
                w  = (x2-x1)/r.orig_shape[1]
                h  = (y2-y1)/r.orig_shape[0]
                dets.append({"cls":cls_name, "conf":conf, "bbox":(cx,cy,w,h)})
        return dets
