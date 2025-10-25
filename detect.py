import argparse
from ultralytics import YOLO
from .haptics import (
    process_frame_detections,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image or video")
    ap.add_argument("--conf", type=float, default=0.25, help="object confidence threshold for filtering detections (default: 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for Non-Max Suppression (NMS) (default: 0.7)")
    args = ap.parse_args()
    model = YOLO("yolov8n.pt")

    results_generator = model(
        args.input, 
        stream=True, 
        show=True, 
        save=True,
        conf=args.conf, 
        iou=args.iou     
    )

    for r in results_generator:
        frame_detections = []
        
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                box_data = box.xywhn.cpu().tolist()[0]
                
                frame_detections.append({
                    'class_name': class_name,
                    'box_data': box_data
                })

        process_frame_detections(frame_detections)


if __name__ == "__main__":
    main()