import argparse
import pathlib
import random
from typing import List, Tuple

from PIL import Image, ImageDraw


def load_labels(label_path: pathlib.Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = parts
        boxes.append((int(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def xywh_to_xyxy(box, w_img, h_img):
    _, cx, cy, w, h = box
    x1 = (cx - w / 2.0) * w_img
    y1 = (cy - h / 2.0) * h_img
    x2 = (cx + w / 2.0) * w_img
    y2 = (cy + h / 2.0) * h_img
    return x1, y1, x2, y2


def choose_color(idx: int):
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]
    return palette[idx % len(palette)]


def draw_boxes(img_path: pathlib.Path, lbl_path: pathlib.Path, out_path: pathlib.Path, class_names: List[str]):
    boxes = load_labels(lbl_path)
    if not boxes:
        return False
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cls, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = xywh_to_xyxy((cls, cx, cy, w, h), img.width, img.height)
        color = choose_color(cls)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name = class_names[cls] if cls < len(class_names) else str(cls)
        draw.text((x1 + 2, y1 + 2), name, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def read_names(data_yaml: pathlib.Path) -> List[str]:
    names = []
    for line in data_yaml.read_text().splitlines():
        if line.strip().startswith("names:"):
            try:
                raw = line.split(":", 1)[1].strip()
                # crude parse for simple JSON-ish list
                if raw.startswith("[") and raw.endswith("]"):
                    raw = raw[1:-1]
                names = [x.strip().strip("'\"") for x in raw.split(",") if x.strip()]
            except Exception:
                pass
            break
    if not names:
        names = ["class0"]
    return names


def main():
    ap = argparse.ArgumentParser(description="Render YOLO label previews for quick sanity checks.")
    ap.add_argument("--data", default="dataset_yolo_train", help="YOLO dataset root containing images/ and labels/")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to sample from")
    ap.add_argument("--count", type=int, default=8, help="Number of images to preview")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--out", default="runs/label_previews", help="Output directory for preview images")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data)
    img_dir = data_root / "images" / args.split
    lbl_dir = data_root / "labels" / args.split
    data_yaml = data_root / "data.yaml"

    if not img_dir.exists():
        raise SystemExit(f"Image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise SystemExit(f"Label dir not found: {lbl_dir}")
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    class_names = read_names(data_yaml)
    imgs = sorted(img_dir.glob("*"))
    random.seed(args.seed)
    random.shuffle(imgs)
    imgs = imgs[: args.count]

    saved = 0
    for img_path in imgs:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        out_path = pathlib.Path(args.out) / args.split / f"{stem}.jpg"
        if draw_boxes(img_path, lbl_path, out_path, class_names):
            saved += 1
    print(f"Saved {saved} previews to {args.out}")


if __name__ == "__main__":
    main()
