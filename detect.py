import argparse
from ultralytics import YOLO
from .haptics import (
    process_frame_detections,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image or video")
    # New Arguments for NMS control
    ap.add_argument("--conf", type=float, default=0.25, help="object confidence threshold for filtering detections (default: 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for Non-Max Suppression (NMS) (default: 0.7)")
    args = ap.parse_args()

    # Load the pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Pass the NMS arguments to the model call to filter the detection results
    results_generator = model(
        args.input, 
        stream=True, 
        show=True, 
        save=True,
        conf=args.conf, # Confidence threshold
        iou=args.iou     # NMS IoU threshold
    )

    for r in results_generator:
        frame_detections = []
        
        # r is a Results object for a single frame/image
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                # box_data is [x_center, y_center, width, height], all normalized [0, 1]
                box_data = box.xywhn.cpu().tolist()[0]
                
                # Collect all detections in the current frame
                frame_detections.append({
                    'class_name': class_name,
                    'box_data': box_data
                })

        # Process all detections for the frame and generate a single, prioritized haptic signal
        process_frame_detections(frame_detections)


if __name__ == "__main__":
    main()