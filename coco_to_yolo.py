import argparse, json, os, shutil, pathlib, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to COCO instances_*.json")
    ap.add_argument("--images", required=True, help="folder with images (train2017 or val2017)")
    ap.add_argument("--out", default="dataset", help="output root")
    ap.add_argument("--img-sub", default="images/train")
    ap.add_argument("--lab-sub", default="labels/train")
    ap.add_argument("--keep", default="person,bicycle,car,traffic light,stop sign,bench",
                    help="comma-separated class names to keep")
    ap.add_argument("--max-images", type=int, default=0, help="optional limit for a quick first run")
    args = ap.parse_args()

    keep = [s.strip() for s in args.keep.split(",") if s.strip()]
    out_img = os.path.join(args.out, args.img-sub)
    out_lab = os.path.join(args.out, args.lab-sub)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    with open(args.ann, "r") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    name_to_yolo = {name: i for i, name in enumerate(keep)}

    imgs = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        cname = cat_id_to_name.get(a["category_id"])
        if cname in keep and a.get("iscrowd", 0) == 0:
            anns_by_img.setdefault(a["image_id"], []).append(a)

    img_ids = [i for i in imgs.keys() if i in anns_by_img]
    random.seed(42)
    random.shuffle(img_ids)
    if args.max_images and args.max_images < len(img_ids):
        img_ids = img_ids[:args.max_images]

    copied = 0
    for img_id in img_ids:
        im = imgs[img_id]
        src = os.path.join(args.images, im["file_name"])
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_img, im["file_name"])
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        W, H = im["width"], im["height"]
        lab_path = os.path.join(out_lab, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
        lines = []
        for a in anns_by_img[img_id]:
            x, y, w, h = a["bbox"]  # COCO: top-left x,y,width,height (pixels)
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            ww = w / W
            hh = h / H
            cls = name_to_yolo[cat_id_to_name[a["category_id"]]]
            # clip to [0,1] just in case
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
        with open(lab_path, "w") as f:
            f.write("\n".join(lines))
        copied += 1

    # write class list so you can build data.yaml easily
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        for name in keep:
            f.write(name + "\n")
    print(f"Done: {copied} images → {out_img} and {out_lab}")
    print("Classes:", keep)

if __name__ == "__main__":
    main()
