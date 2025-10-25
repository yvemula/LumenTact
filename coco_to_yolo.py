import argparse, json, os, shutil, pathlib, random

def process_subset(subset_name, img_ids, imgs, anns_by_img, cat_id_to_name, name_to_yolo, args, out_root):
    """Handles copying images and generating labels for a given subset (train/val)."""
    
    # Define output paths using the subset name (e.g., 'images/train', 'labels/train')
    out_img_sub = os.path.join(args.img_sub_root, subset_name)
    out_lab_sub = os.path.join(args.lab_sub_root, subset_name)
    out_img = os.path.join(out_root, out_img_sub)
    out_lab = os.path.join(out_root, out_lab_sub)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    copied = 0
    for img_id in img_ids:
        im = imgs[img_id]
        src = os.path.join(args.images, im["file_name"])
        if not os.path.exists(src):
            continue
            
        # 1. Copy Image
        dst = os.path.join(out_img, im["file_name"])
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        # 2. Generate YOLO Labels
        W, H = im["width"], im["height"]
        lab_path = os.path.join(out_lab, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
        lines = []
        for a in anns_by_img[img_id]:
            x, y, w, h = a["bbox"]  # COCO: top-left x,y,width,height (pixels)
            
            # Convert COCO bbox to normalized YOLO format (x_center, y_center, width, height)
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
    
    print(f"  {copied} images copied to {out_img_sub} and labels to {out_lab_sub}")
    return copied

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to COCO instances_*.json")
    ap.add_argument("--images", required=True, help="folder with images (train2017 or val2017)")
    ap.add_argument("--out", default="dataset", help="output root")
    ap.add_argument("--img-sub-root", default="images", help="output image subdirectory root")
    ap.add_argument("--lab-sub-root", default="labels", help="output label subdirectory root")
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraction of images to reserve for validation set.")
    ap.add_argument("--keep", default="person,bicycle,car,traffic light,stop sign,bench",
                    help="comma-separated class names to keep")
    ap.add_argument("--max-images", type=int, default=0, help="optional limit for a quick first run")
    args = ap.parse_args()

    # --- Data Loading and Filtering ---
    keep = [s.strip() for s in args.keep.split(",") if s.strip()]
    
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
    
    # --- Split Data ---
    random.seed(42)
    random.shuffle(img_ids)
    
    if args.max_images and args.max_images < len(img_ids):
        img_ids = img_ids[:args.max_images]

    val_count = int(len(img_ids) * args.val_split)
    
    val_ids = img_ids[:val_count]
    train_ids = img_ids[val_count:]
    
    print(f"Total relevant images: {len(img_ids)}")
    print(f"Splitting: {len(train_ids)} for training, {len(val_ids)} for validation ({args.val_split*100:.1f}%)")

    # --- Process Subsets ---
    total_copied = 0
    
    print("Processing 'train' subset...")
    copied_train = process_subset("train", train_ids, imgs, anns_by_img, cat_id_to_name, name_to_yolo, args,