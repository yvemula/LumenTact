import argparse, json, os, shutil, pathlib, random, math, sys
from collections import defaultdict, Counter

def iou(a, b):
    # optional duplicate suppression helper (not enabled by default)
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def seg_to_tight_box(seg):
    xs, ys = [], []
    for poly in seg:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs or not ys: return None
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

def clip_box(x, y, w, h, W, H):
    x2, y2 = x + w, y + h
    x1c = max(0.0, min(float(W), x))
    y1c = max(0.0, min(float(H), y))
    x2c = max(0.0, min(float(W), x2))
    y2c = max(0.0, min(float(H), y2))
    wc, hc = max(0.0, x2c - x1c), max(0.0, y2c - y1c)
    return x1c, y1c, wc, hc

def to_yolo_line(cls, x, y, w, h, W, H):
    xc = (x + w / 2.0) / W
    yc = (y + h / 2.0) / H
    ww = w / W
    hh = h / H
    # final clamp
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    ww = min(max(ww, 0.0), 1.0)
    hh = min(max(hh, 0.0), 1.0)
    return f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"

def parse_kv_renames(s):
    # format: "traffic light=signal,stop sign=stop_sign"
    out = {}
    if not s: return out
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        if "=" not in kv:
            # allow simple passthrough tokens; they’ll be kept as-is
            out[kv] = kv
        else:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to COCO instances_*.json")
    ap.add_argument("--images", required=True, help="folder with images")
    ap.add_argument("--out", default="dataset_yolo", help="output root")
    ap.add_argument("--img-sub", dest="img_sub", default="images/train")
    ap.add_argument("--lab-sub", dest="lab_sub", default="labels/train")
    ap.add_argument("--keep", default="person,bicycle,car,traffic light,stop sign,bench",
                    help="comma-separated class names to keep (after rename if provided)")
    ap.add_argument("--rename", default="", help="optional renames/merges: 'traffic light=signal,stop sign=sign_stop'")
    ap.add_argument("--min-area", type=float, default=0.0, help="drop boxes smaller than this FRACTION of image area (e.g., 0.0005)")
    ap.add_argument("--min-w", type=float, default=0.0, help="drop boxes smaller than this FRACTION of image width")
    ap.add_argument("--min-h", type=float, default=0.0, help="drop boxes smaller than this FRACTION of image height")
    ap.add_argument("--use-seg", action="store_true", help="prefer segmentation-derived tight bbox if available")
    ap.add_argument("--allow-crowd", action="store_true", help="include iscrowd==1 annotations")
    ap.add_argument("--max-images", type=int, default=0, help="limit for quick dry runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.0, help="if >0, perform stratified split into train/val under images/ and labels/")
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--write-yaml", action="store_true", help="write data.yaml alongside classes.txt")
    ap.add_argument("--yaml-name", default="lumendata.yaml", help="name of YAML to write when --write-yaml")
    args = ap.parse_args()

    random.seed(args.seed)

    keep_names = [s.strip() for s in args.keep.split(",") if s.strip()]
    renames = parse_kv_renames(args.rename)

    with open(args.ann, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}

    # rename/merge map applied to COCO names
    def rename(name):
        return renames.get(name, name)

    # final class list after rename, filtered by keep
    final_names = []
    seen = set()
    for n in keep_names:
        rn = rename(n)
        if rn not in seen:
            final_names.append(rn); seen.add(rn)

    name_to_yolo = {n: i for i, n in enumerate(final_names)}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if (not args.allow_crowd) and a.get("iscrowd", 0) == 1:
            continue
        cname_raw = cat_id_to_name.get(a["category_id"])
        cname = rename(cname_raw)
        if cname in final_names:
            anns_by_img[a["image_id"]].append(a)

    # keep only images with at least one kept annotation
    img_ids = [i for i in imgs.keys() if len(anns_by_img[i]) > 0]
    random.shuffle(img_ids)
    if args.max_images and args.max_images < len(img_ids):
        img_ids = img_ids[:args.max_images]

    # prepare output dirs (train for now; we'll move to val if split)
    out_img = os.path.join(args.out, args.img_sub)
    out_lab = os.path.join(args.out, args.lab_sub)
    pathlib.Path(out_img).mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_lab).mkdir(parents=True, exist_ok=True)

    stats = {
        "images_total": len(imgs),
        "images_kept": 0,
        "boxes_kept": 0,
        "boxes_dropped_tiny": 0,
        "boxes_dropped_invalid": 0,
        "boxes_dropped_oob": 0,
        "per_class_counts": Counter(),
    }

    def write_one(image_id, dest_img_dir, dest_lab_dir):
        im = imgs[image_id]
        src = os.path.join(args.images, im["file_name"])
        if not os.path.exists(src):
            return False, 0
        W, H = float(im["width"]), float(im["height"])
        # copy/symlink
        dst = os.path.join(dest_img_dir, im["file_name"])
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if args.symlink:
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        # labels
        lab_path = os.path.join(dest_lab_dir, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
        lines = []
        kept_boxes = 0
        for a in anns_by_img[image_id]:
            cname = rename(cat_id_to_name[a["category_id"]])
            cls = name_to_yolo[cname]
            # choose bbox
            bx = a.get("bbox", None)
            if args.use_seg and a.get("segmentation"):
                tight = seg_to_tight_box(a["segmentation"])
                if tight: bx = tight
            if not bx or len(bx) != 4:
                stats["boxes_dropped_invalid"] += 1
                continue
            x, y, w, h = bx
            # clip to image bounds
            x, y, w, h = clip_box(x, y, w, h, W, H)
            if w <= 0 or h <= 0:
                stats["boxes_dropped_invalid"] += 1
                continue
            # size filters (fractions)
            if (args.min_area and (w * h) < args.min_area * (W * H)) or \
               (args.min_w and (w < args.min_w * W)) or \
               (args.min_h and (h < args.min_h * H)):
                stats["boxes_dropped_tiny"] += 1
                continue
            lines.append(to_yolo_line(cls, x, y, w, h, W, H))
            kept_boxes += 1
            stats["per_class_counts"][cname] += 1
        if kept_boxes == 0:
            # remove copied image if no labels kept
            try:
                if not args.symlink and os.path.exists(dst): os.remove(dst)
            except Exception:
                pass
            return False, 0
        with open(lab_path, "w") as f:
            f.write("\n".join(lines))
        return True, kept_boxes

    kept_images = []
    for iid in img_ids:
        ok, k = write_one(iid, out_img, out_lab)
        if ok:
            kept_images.append(iid)
            stats["images_kept"] += 1
            stats["boxes_kept"] += k

    # optional stratified split
    if args.val_ratio > 0.0:
        val_img_dir = os.path.join(args.out, "images/val")
        val_lab_dir = os.path.join(args.out, "labels/val")
        pathlib.Path(val_img_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(val_lab_dir).mkdir(parents=True, exist_ok=True)
        # naive stratification: group by dominant class per image
        img_dominant = []
        for iid in kept_images:
            # choose the most frequent class in that image’s annotations
            counts = Counter(rename(cat_id_to_name[a["category_id"]]) for a in anns_by_img[iid]
                             if rename(cat_id_to_name[a["category_id"]]) in final_names)
            dom = counts.most_common(1)[0][0] if counts else None
            img_dominant.append((iid, dom))
        by_class = defaultdict(list)
        for iid, dom in img_dominant:
            by_class[dom].append(iid)
        val_ids = set()
        for c, arr in by_class.items():
            random.shuffle(arr)
            take = max(1, int(len(arr) * args.val_ratio))
            val_ids.update(arr[:take])
        # move files into val
        for iid in val_ids:
            im = imgs[iid]
            # move image
            src_img = os.path.join(out_img, im["file_name"])
            dst_img = os.path.join(val_img_dir, im["file_name"])
            pathlib.Path(dst_img).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_img, dst_img)
            # move label
            src_lab = os.path.join(out_lab, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
            dst_lab = os.path.join(val_lab_dir, pathlib.Path(im["file_name"]).with_suffix(".txt").name)
            pathlib.Path(dst_lab).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_lab, dst_lab)

    # write classes + optional YAML
    classes_txt = os.path.join(args.out, "classes.txt")
    with open(classes_txt, "w") as f:
        for n in final_names: f.write(n + "\n")

    if args.write_yaml:
        yaml_path = os.path.join(args.out, args.yaml_name)
        rel = "."
        with open(yaml_path, "w") as f:
            f.write(f"path: {rel}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n")
            f.write("names: [" + ", ".join(final_names) + "]\n")

    # stats file
    stats_path = os.path.join(args.out, "convert_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=int)

    # console summary
    print("\n=== COCO → YOLO CONVERSION SUMMARY ===")
    print(f"Images total:   {stats['images_total']}")
    print(f"Images kept:    {stats['images_kept']}")
    print(f"Boxes kept:     {stats['boxes_kept']}")
    print(f"Dropped tiny:   {stats['boxes_dropped_tiny']}")
    print(f"Dropped invalid:{stats['boxes_dropped_invalid']}")
    print("Per-class counts:", dict(stats['per_class_counts']))
    print("Classes (index order):", final_names)
    print("Wrote:", classes_txt)
    if args.write_yaml:
        print("Wrote YAML:", os.path.join(args.out, args.yaml_name))

if __name__ == "__main__":
    main()
