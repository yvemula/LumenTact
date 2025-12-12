import argparse
import pathlib
import yaml


def parse_names_from_lumendata(lumendata_path: pathlib.Path):
    if not lumendata_path.exists():
        return []
    try:
        data = yaml.safe_load(lumendata_path.read_text())
        names = data.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception:
        pass
    return []


def gather_counts(img_dir: pathlib.Path, lbl_dir: pathlib.Path):
    imgs = sorted(img_dir.glob("*"))
    lbls = sorted(lbl_dir.glob("*.txt"))
    return len(imgs), len(lbls)


def main():
    ap = argparse.ArgumentParser(description="Ensure YOLO data.yaml exists; emit basic split counts.")
    ap.add_argument("--data", default="dataset_yolo_train", help="Dataset root with images/ and labels/ subfolders")
    ap.add_argument("--names", default="", help="Comma-separated class names (overrides lumendata.yaml/defaults)")
    ap.add_argument("--lumendata", default="lumendata.yaml", help="Optional source of names if --names not set")
    ap.add_argument("--force", action="store_true", help="Overwrite data.yaml if it exists")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data)
    data_yaml = data_root / "data.yaml"

    train_img = data_root / "images/train"
    val_img = data_root / "images/val"
    train_lbl = data_root / "labels/train"
    val_lbl = data_root / "labels/val"

    for p in [train_img, val_img, train_lbl, val_lbl]:
        if not p.exists():
            raise SystemExit(f"Missing expected path: {p}")

    train_imgs, train_lbls = gather_counts(train_img, train_lbl)
    val_imgs, val_lbls = gather_counts(val_img, val_lbl)

    if args.names.strip():
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = parse_names_from_lumendata(pathlib.Path(args.lumendata))
    if not names:
        names = ["obstacle"]

    print(f"Train imgs={train_imgs}, labels={train_lbls} | Val imgs={val_imgs}, labels={val_lbls}")
    if data_yaml.exists() and not args.force:
        print(f"{data_yaml} already exists; use --force to overwrite.")
        return

    content = {
        "train": str((train_img).resolve()),
        "val": str((val_img).resolve()),
        "nc": len(names),
        "names": names,
    }
    data_yaml.write_text(yaml.safe_dump(content, sort_keys=False))
    print(f"Wrote {data_yaml} with {len(names)} class names.")


if __name__ == "__main__":
    main()
