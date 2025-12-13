"""
Generate pseudo-labels using a teacher YOLO model (for distillation/semi-supervised training).

Example:
  ./venv/bin/python tools/pseudo_labels.py \
    --teacher runs/detect/sanpo_yolo/weights/best.pt \
    --data dataset_yolo_train \
    --out dataset_yolo_pseudo \
    --conf 0.25 --imgsz 1280 --device 0 --symlink
"""
import argparse
import pathlib
from typing import Iterable, List

import yaml
from ultralytics import YOLO
from PIL import Image


def list_images(p: pathlib.Path) -> List[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts])


def norm_box(x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def write_labels(out_path: pathlib.Path, boxes):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c, cx, cy, w, h in boxes:
            f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def extract_names(model_names) -> List[str]:
    if isinstance(model_names, dict):
        return [model_names[k] for k in sorted(model_names.keys())]
    return [str(n) for n in model_names]


def main():
    ap = argparse.ArgumentParser(description="Generate pseudo-labels from a teacher YOLO model.")
    ap.add_argument("--teacher", required=True, help="Path to teacher weights")
    ap.add_argument("--data", default="dataset_yolo_train", help="Source dataset root with images/{train,val}")
    ap.add_argument("--out", default="dataset_yolo_pseudo", help="Output dataset root with pseudo labels")
    ap.add_argument("--splits", default="train", help="Comma-separated splits to process (train,val)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for teacher predictions")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference size for teacher")
    ap.add_argument("--device", default="0", help="Device id or 'cpu'")
    ap.add_argument("--max", type=int, default=0, help="Limit number of images per split (0 = all)")
    ap.add_argument("--symlink", action="store_true", help="Symlink images into the pseudo dataset (else copy)")
    ap.add_argument("--overwrite-labels", action="store_true", help="Overwrite existing labels in output")
    args = ap.parse_args()

    src_root = pathlib.Path(args.data)
    out_root = pathlib.Path(args.out)
    teacher = YOLO(args.teacher)
    names = extract_names(teacher.names)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        if not src_img_dir.exists():
            print(f"Skip split '{split}': no images at {src_img_dir}")
            continue
        imgs = list_images(src_img_dir)
        if args.max > 0:
            imgs = imgs[: args.max]
        print(f"[{split}] Generating pseudo-labels for {len(imgs)} images from {src_img_dir}")

        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            stem = img_path.stem
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / f"{stem}.txt"
            if out_lbl.exists() and not args.overwrite_labels:
                continue

            # link/copy image
            if args.symlink:
                if out_img.exists():
                    out_img.unlink()
                out_img.symlink_to(img_path.resolve())
            else:
                if not out_img.exists():
                    out_img.write_bytes(img_path.read_bytes())

            # teacher predict
            res = teacher.predict(img_path, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            H, W = res.orig_shape
            boxes = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0]) if box.cls is not None else 0
                cx, cy, w, h = norm_box(x1, y1, x2, y2, W, H)
                boxes.append((cls, cx, cy, w, h))
            write_labels(out_lbl, boxes)

    # write data.yaml for pseudo dataset
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str((out_root / "images/train").resolve()),
                "val": str((out_root / "images/val").resolve()),
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        )
    )
    print(f"Wrote pseudo dataset yaml: {data_yaml}")


if __name__ == "__main__":
    main()
