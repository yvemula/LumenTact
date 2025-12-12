"""
Preflight checks before training/inference.

Usage:
  ./venv/bin/python tools/preflight_check.py --data dataset_yolo_train/data.yaml
"""
import argparse
import pathlib
import yaml


def main():
    ap = argparse.ArgumentParser(description="Run basic dataset sanity checks.")
    ap.add_argument("--data", default="dataset_yolo_train/data.yaml", help="Path to data.yaml")
    args = ap.parse_args()

    data_yaml = pathlib.Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    cfg = yaml.safe_load(data_yaml.read_text())
    train = pathlib.Path(cfg.get("train", ""))
    val = pathlib.Path(cfg.get("val", ""))
    names = cfg.get("names", [])

    missing = []
    for p in [train, val]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit(f"Missing paths: {missing}")

    def count_split(base: pathlib.Path):
        imgs = list((base).glob("*"))
        lbls = list((base.parent.parent / "labels" / base.name).glob("*.txt"))
        return len(imgs), len(lbls)

    train_imgs, train_lbls = count_split(train)
    val_imgs, val_lbls = count_split(val)

    print(f"Train imgs={train_imgs}, labels={train_lbls} | Val imgs={val_imgs}, labels={val_lbls}")
    if train_imgs == 0 or train_lbls == 0:
        print("Warning: train split empty or missing labels.")
    if val_imgs == 0 or val_lbls == 0:
        print("Warning: val split empty or missing labels.")

    print(f"Classes ({len(names)}): {names}")
    print("Preflight check complete.")


if __name__ == "__main__":
    main()
