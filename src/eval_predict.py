from ultralytics import YOLO
import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--src", default="dataset/images/val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.val(data=args.data, device=args.device)  # mAP/PR/confusion

    model.predict(source=args.src, device=args.device, save=args.save, imgsz=640, conf=0.25)

if __name__ == "__main__":
    main()
