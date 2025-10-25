import argparse
from ultralytics import YOLO
from .haptics import (
    generate_haptic_feedback,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image or video")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")

    results_generator = model(args.input, stream=True, show=True, save=True)

    for r in results_generator:

        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]

                box_data = box.xywhn.cpu().tolist()[0]

                generate_haptic_feedback(class_name, box_data)


if __name__ == "__main__":
    main()