import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="path to input image or video")
    args = ap.parse_args()

    model = YOLO("yolov8n.pt")

    results = model(args.input, show=True, save=True)


if __name__ == "__main__":
    main()
