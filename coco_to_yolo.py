

import argparse, json, os, shutil, pathlib, random, sys

def ensure(path, kind):
    if not os.path.exists(path):
        sys.exit(f"[ERR] {kind} not found: {path}")
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to COCO instances_*.json")
    ap.add_argument("--images", required=True, help="folder with images (train2017 or val2017)")
    ap.add_argument("--out", default="dataset", help="output root")
    # accept both --img-sub and --img_sub
    ap.add_argument("--img-sub", "--img_sub", dest="img_sub", default="images/train")
    ap.add_argument("--lab-sub", "--lab_sub", dest="lab_sub", default="labels/train")
    ap.add_argument("--keep", default="person,bicycle,car,traffic light,stop sign,bench",
                    help="comma-separated class names to keep")
    ap.add_argument("--max-images", type=int, default=0, help="optional limit for a quick first run")
    args = ap.parse_args()

    # sanity checks
    ensure(args.ann, "annotations JSON")
    ensure(args.images, "images folder")

    keep = [s.strip() for s in args.keep.split(",") if s.strip()]
    out_img = os.path.join(args.out, args.img_sub)
    out_lab = os.path.join(args.out, args.lab_sub)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    print(f"[INFO] ann={args.ann}")
    print(f"[INFO] images={args.images}")
    print(f"[INFO] out_img={out_img}")
    print(f"[INFO] out_lab={out_lab}")
    print(f"[INFO] keep={keep}  max_images={args.max_images or 'ALL'}")

    with open(args.ann, "r") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    name_to_yolo   = {name: i for i, name in enumerate(keep)}

    imgs = {im["id"]: im for im in data.get("images", [])}
    anns_by_img = {}
    dropped_unknown = 0

    for a in data.get("annotations", []):
        cname = cat_id_to_name.get(a.get("category_id"))
        if cname not in keep or a.get("iscrowd", 0) == 1:
            continue
        if a.get("bbox") is None:
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    img_ids = [i for i in imgs if i in anns_by_img]
    if not img_ids:
        sys.exit("[ERR] No images left after filtering; check --keep class names match COCO labels exactly.")
    random.seed(42)
    random.shuffle(img_ids)
    if args.max_images and args.max_images < len(img_ids):
        img_ids = img_ids[:args.max_images]

    copied = 0
    for img_id in img_ids:
        im = imgs[img_id]
        src = os.path.join(args.images, im["file_name"])
        if not os.path.exists(src):
            # skip missing files but warn once in a while
            if copied % 500 == 0:
                print(f"[WARN] Missing image, skipping: {src}")
            continue

        dst = os.path.join(out_img, im["file_name"])
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        W, H = im["width"], im["height"]
        lab_path = os.path.join(out_lab, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
        lines = []
        for a in anns_by_img[img_id]:
            cname = cat_id_to_name[a["category_id"]]
            if cname not in name_to_yolo:
                dropped_unknown += 1
                continue
            x, y, w, h = a["bbox"]  # COCO bbox in pixels (x_min, y_min, w, h)
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            ww = w / W
            hh = h / H
            # clip
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)
            cls = name_to_yolo[cname]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        with open(lab_path, "w") as f:
            f.write("\n".join(lines))
        copied += 1
        if copied % 1000 == 0:
            print(f"[INFO] Copied {copied} images...")

    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        for name in keep:
            f.write(name + "\n")

    print(f"[DONE] {copied} images → {out_img} and {out_lab}")
    print(f"[INFO] Classes: {keep}")
    if dropped_unknown:
        print(f"[INFO] Dropped {dropped_unknown} boxes with unknown classes")

if __name__ == "__main__":
    main()
