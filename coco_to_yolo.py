# LumenTact/coco_to_yolo.py
import argparse
import json
import os
import shutil
import pathlib
import random
import yaml

def process_subset(subset_name, img_ids, imgs, anns_by_img, cat_id_to_name, name_to_yolo, args, out_root):
    """Handles copying images and generating labels for a given subset (train/val)."""
    
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
            # print(f"Warning: Image file not found {src}")
            continue
            
        dst = os.path.join(out_img, im["file_name"])
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        W, H = im["width"], im["height"]
        lab_path = os.path.join(out_lab, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
        lines = []
        if img_id in anns_by_img: # Ensure image has annotations
            for a in anns_by_img[img_id]:
                x, y, w, h = a["bbox"]  
                
                # Convert COCO (top-left x, y, width, height) to YOLO (center x, y, width, height) normalized
                xc = (x + w / 2) / W
                yc = (y + h / 2) / H
                ww = w / W
                hh = h / H
                cls = name_to_yolo[cat_id_to_name[a["category_id"]]]
                
                # Clamp values to [0.0, 1.0] to fix potential bad annotations
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
    ap.add_argument("--images", required=True, help="folder with images (e.g., coco2017/train2017)")
    ap.add_argument("--out", default="yolo-dataset", help="output root")
    ap.add_argument("--img-sub-root", default="images", help="output image subdirectory root")
    ap.add_argument("--lab-sub-root", default="labels", help="output label subdirectory root")
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraction of images to reserve for validation set.")
    ap.add_argument("--keep", default="person,bicycle,car,traffic light,stop sign,bench",
                    help="comma-separated class names to keep")
    ap.add_argument("--max-images", type=int, default=0, help="optional limit for a quick first run")
    args = ap.parse_args()

    keep = [s.strip() for s in args.keep.split(",") if s.strip()]
    
    print(f"Loading annotations from: {args.ann}...")
    with open(args.ann, "r") as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    name_to_yolo = {name: i for i, name in enumerate(keep)}

    print(f"Keeping {len(keep)} classes: {keep}")

    imgs = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    print("Processing annotations...")
    total_anns = 0
    for a in data["annotations"]:
        cname = cat_id_to_name.get(a["category_id"])
        # Keep annotation if it's in our list and not a "crowd" annotation
        if cname in keep and a.get("iscrowd", 0) == 0:
            anns_by_img.setdefault(a["image_id"], []).append(a)
            total_anns += 1

    # Filter image IDs to only include those that have at least one annotation we care about
    img_ids = [i for i in imgs.keys() if i in anns_by_img]
    print(f"Found {len(img_ids)} images with {total_anns} relevant annotations.")
    
    random.seed(42) # for reproducible splits
    random.shuffle(img_ids)
    
    if args.max_images and args.max_images < len(img_ids):
        print(f"Limiting to {args.max_images} images (from {len(img_ids)}).")
        img_ids = img_ids[:args.max_images]

    val_count = int(len(img_ids) * args.val_split)
    train_ids = img_ids[val_count:]
    val_ids = img_ids[:val_count]
    
    print(f"\nTotal relevant images: {len(img_ids)}")
    print(f"Splitting: {len(train_ids)} for training, {len(val_ids)} for validation ({args.val_split*100:.1f}%)")

    # Create output root directory
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    
    total_copied = 0
    
    print("\nProcessing 'train' subset...")
    copied_train = process_subset("train", train_ids, imgs, anns_by_img, cat_id_to_name, name_to_yolo, args, out_root)
    total_copied += copied_train

    print("Processing 'val' subset...")
    copied_val = process_subset("val", val_ids, imgs, anns_by_img, cat_id_to_name, name_to_yolo, args, out_root)
    total_copied += copied_val

    # --- Create data.yaml file ---
    yaml_path = os.path.join(out_root, "data.yaml")
    
    # Get relative paths from the data.yaml file
    train_img_path = os.path.join(args.img_sub_root, "train")
    val_img_path = os.path.join(args.img_sub_root, "val")
    
    yaml_data = {
        "train": train_img_path,
        "val": val_img_path,
        "nc": len(keep),
        "names": keep
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print(f"\nSuccessfully created {total_copied} images and labels in: {out_root}")
    print(f"Training config file created at: {yaml_path}")
    print("Conversion complete.")


if __name__ == "__main__":
    main()