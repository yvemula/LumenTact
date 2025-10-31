import argparse, pathlib, shutil, random, re
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm

# -----------------------------
# Helpers
# -----------------------------
def load_mask(path: pathlib.Path) -> np.ndarray:
    """Load mask as integer ID map. Supports npy/npz or image files.
       If RGB panoptic, compress to a 24-bit ID."""
    if path.suffix.lower() in (".npy", ".npz"):
        arr = np.load(path)
        if hasattr(arr, "files"):  # npz: take first array
            arr = arr[list(arr.files)[0]]
        return arr
    arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] >= 3:  # RGB panoptic
        arr = (arr[:, :, 0].astype(np.uint32) << 16) | \
              (arr[:, :, 1].astype(np.uint32) << 8)  | \
               arr[:, :, 2].astype(np.uint32)
    return arr

def norm_box(xmin, ymin, xmax, ymax, W, H):
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return cx, cy, w, h

def session_key(sess: pathlib.Path, src_root: pathlib.Path) -> str:
    """Stable, filesystem-safe prefix per session to avoid name collisions."""
    rel = sess.relative_to(src_root)
    return re.sub(r"[^A-Za-z0-9]+", "_", str(rel))

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build a YOLO dataset from SANPO masks (no COCO JSON needed).")
    ap.add_argument("--src", default="sanpo_subset_auto", help="Root of downloaded SANPO subset")
    ap.add_argument("--out", default="dataset_yolo", help="Output YOLO dataset root")
    ap.add_argument("--ids", default="", help="Comma-separated mask IDs to keep (order defines class IDs). Empty=any nonzero → class 0")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction (split by session)")
    ap.add_argument("--min-px", type=int, default=8, help="Drop boxes smaller than this many pixels on a side")
    ap.add_argument("--min-frac", type=float, default=0.0, help="Drop boxes smaller than this fraction of image area")
    ap.add_argument("--symlink", action="store_true", help="Symlink images instead of copying (saves disk)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for session split")
    args = ap.parse_args()

    random.seed(args.seed)

    SRC = pathlib.Path(args.src)
    OUT = pathlib.Path(args.out)
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUT / p).mkdir(parents=True, exist_ok=True)

    IDS = None
    if args.ids.strip():
        IDS = [int(x) for x in args.ids.split(",") if x.strip()]

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    MSK_EXTS = IMG_EXTS | {".npy", ".npz"}

    IMG_HINT = re.compile(r"(left|cam|camera|rgb|image|frame|stereo)", re.I)
    MSK_HINT = re.compile(r"(mask|seg|panoptic|semantic|label|annot|anno)", re.I)

    # -----------------------------------
    # Discover session directories
    # -----------------------------------
    session_dirs = set()
    for p in SRC.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            parts = p.parts
            if "sessions" in parts:
                idx = parts.index("sessions")
                if idx + 1 < len(parts):
                    session_dirs.add(pathlib.Path(*parts[:idx + 2]))
            else:
                # Fallback: go up two levels
                session_dirs.add(p.parent.parent if len(parts) >= 2 else p.parent)
    session_dirs = sorted(session_dirs)

    # -----------------------------------
    # Build image↔mask pairs per session
    # -----------------------------------
    pairs = []
    for sess in session_dirs:
        img_map, img_score = {}, {}
        msk_map, msk_score = {}, {}

        for f in sess.rglob("*"):
            if f.is_file():
                suf = f.suffix.lower()
                if suf in IMG_EXTS:
                    stem = f.stem
                    img_map[stem] = f
                    img_score[stem] = 1 + (1 if IMG_HINT.search(str(f)) else 0)
                elif suf in MSK_EXTS:
                    stem = f.stem
                    msk_map[stem] = f
                    msk_score[stem] = 1 + (1 if MSK_HINT.search(str(f)) else 0)

        shared = sorted(set(img_map) & set(msk_map))
        for stem in shared:
            score = img_score.get(stem, 1) + msk_score.get(stem, 1)
            pairs.append((sess, img_map[stem], msk_map[stem], score))

    # Prefer likely-good folders (left/seg)
    pairs.sort(key=lambda x: x[3], reverse=True)

    # -----------------------------------
    # Session-level split
    # -----------------------------------
    sess_list = sorted({s for s, *_ in pairs})
    random.shuffle(sess_list)
    cut = max(1, int((1.0 - args.val_ratio) * len(sess_list)))
    train_sessions = set(sess_list[:cut])

    # -----------------------------------
    # Convert
    # -----------------------------------
    n_empty = 0
    total_pairs = 0
    for sess, img_p, msk_p, _score in tqdm(pairs, desc="Converting"):
        # Determine split
        split = "train" if sess in train_sessions else "val"

        # Load image
        try:
            img = Image.open(img_p).convert("RGB")
        except Exception:
            continue
        W, H = img.size

        # Load mask
        try:
            arr = load_mask(msk_p)
        except Exception:
            continue

        # Candidate class IDs present in this mask
        cand_ids = np.unique(arr)
        cand_ids = cand_ids[cand_ids != 0]  # drop background
        if IDS is not None:
            cand_ids = [i for i in cand_ids if int(i) in IDS]

        boxes = []
        for mid in cand_ids:
            mask = (arr == int(mid))
            lab = label(mask.astype(np.uint8), connectivity=1)
            for r in regionprops(lab):
                ymin, xmin, ymax, xmax = r.bbox
                wpx, hpx = (xmax - xmin), (ymax - ymin)
                # size thresholds
                if wpx < args.min_px or hpx < args.min_px:
                    continue
                if args.min_frac and (wpx * hpx) < args.min_frac * (W * H):
                    continue
                cx, cy, w, h = norm_box(xmin, ymin, xmax, ymax, W, H)
                # class index
                if IDS is None:
                    cls = 0  # single-class "obstacle"
                else:
                    # class index = position in IDS list
                    cls = IDS.index(int(mid))
                boxes.append((cls, cx, cy, w, h))

        # Build unique output names (avoid collisions across sessions)
        skey = session_key(sess, SRC)
        stem = f"{skey}__{img_p.stem}"
        out_img = OUT / f"images/{split}/{stem}{img_p.suffix.lower()}"
        out_lbl = OUT / f"labels/{split}/{stem}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)

        # Save/symlink image
        if args.symlink:
            if out_img.exists():
                out_img.unlink()
            out_img.symlink_to(img_p.resolve())
        else:
            shutil.copy2(img_p, out_img)

        # Write labels only if we have boxes
        if boxes:
            with open(out_lbl, "w") as f:
                for c, cx, cy, w, h in boxes:
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        else:
            n_empty += 1
            # intentionally no empty txt file (Ultralytics treats missing label as background)

        total_pairs += 1

    print(f"Done. Sessions: {len(sess_list)}  Pairs processed: {total_pairs}  Empty-label images: {n_empty}")
    print(f"Output: {OUT}/images/{{train,val}}  and  {OUT}/labels/{{train,val}}")
    if IDS is None:
        print("Class mapping: single-class (0=obstacle, any nonzero mask)")
    else:
        print(f"Class mapping (index → mask id): { {i: mid for i, mid in enumerate(IDS)} }")
        print("Make sure your lumendata.yaml `names:[...]` matches this order.")
        
if __name__ == "__main__":
    main()
