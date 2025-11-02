import argparse
from ultralytics import YOLO
from .haptics import (
    generate_haptic_feedback,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="path to input video, camera index (e.g., 0), or RTSP stream URL",
    )
    args = ap.parse_args()

    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    print(f"[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print(f"[INFO] Starting detection on source: {source}")

    results_generator = model(source, stream=True, show=True, save=False)

    for r in results_generator:

        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                # box_data is (x_center, y_center, width, height) normalized
                box_data = box.xywhn.cpu().tolist()[0]

                # This function now also needs to handle direction
                generate_haptic_feedback(class_name, box_data)


if __name__ == "__main__":
    main()
