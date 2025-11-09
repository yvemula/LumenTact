# LumenTact/detect.py
import argparse
from ultralytics import YOLO
from .haptics import (
    process_frame_detections,
)
from . import hardware  # <-- IMPORT HARDWARE
from .utils import setup_logger # <-- IMPORT LOGGER

log = setup_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="path to input image/video, or camera index (e.g., '0')",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="object confidence threshold (default: 0.25)",
    )
    ap.add_argument(
        "--iou", type=float, default=0.7, help="IoU threshold for NMS (default: 0.7)"
    )
    ap.add_argument(
        "-m",
        "--model",
        default="yolov8n.pt",
        help="path to YOLOv8 model weights (e.g., yolov8n.pt or runs/detect/lumtact_run/weights/best.pt)",
    )
    args = ap.parse_args()

    # --- NEW: Initialize Haptic Controller ---
    log.info("Initializing haptic controller...")
    controller = hardware.get_controller()
    if not controller.connect():
        log.error("Failed to connect to haptic controller. Exiting.")
        return
    # --- END NEW ---

    try:
        # --- Handle camera input ---
        try:
            source = int(args.input)
            log.info(f"Using camera index: {source}")
        except ValueError:
            source = args.input
            log.info(f"Using video/image source: {source}")

        log.info(f"Loading model: {args.model}...")
        model = YOLO(args.model)

        results_generator = model(
            source,
            stream=True,
            show=True,  # Display the video feed
            save=False,  # Don't save video clips by default
            conf=args.conf,
            iou=args.iou,
        )

        for r in results_generator:
            frame_detections = []

            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    class_name = model.names[cls_id]

                    # Get normalized (x_center, y_center, width, height)
                    box_data = box.xywhn.cpu().tolist()[0]

                    frame_detections.append(
                        {"class_name": class_name, "box_data": box_data}
                    )

            # Process all detections for this frame
            # --- UPDATED: Pass the controller ---
            process_frame_detections(frame_detections, controller)

    except KeyboardInterrupt:
        log.info("Detection stopped by user.")
    except Exception as e:
        log.error(f"An error occurred during detection: {e}")
    finally:
        # --- NEW: Ensure controller is disconnected ---
        log.info("Disconnecting haptic controller.")
        controller.disconnect()
        log.info("Shutdown complete.")
        # --- END NEW ---


if __name__ == "__main__":
    main()