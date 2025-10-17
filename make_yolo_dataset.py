import sys, pathlib, shutil, numpy as np, random, re
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm

random.seed(42)

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path("sanpo_subset_auto")
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else pathlib.Path("dataset_yolo")
OUT.mkdir(parents=True, exist_ok=True)
for p in ["images/train","images/val","labels/train","labels/val"]:
    (OUT/p).mkdir(parents=True, exist_ok=True)

# Optional: comma-separated list of mask IDs to keep (else: any non-zero is obstacle)
IDS = set(map(int, sys.argv[3].split(","))) if len(sys.argv)>3 else None

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
MSK_EXTS = IMG_EXTS | {".npy",".npz"}

# Regex hints (broad, but harmless)
IMG_HINT = re.compile(r"(left|cam|camera|rgb|image|frame|stereo)", re.I)
MSK_HINT = re.compile(r"(mask|seg|panoptic|semantic|label|annot|anno)", re.I)

def load_mask(path):
    if path.suffix.lower() in (".npy",".npz"):
        arr = np.load(path)
        if hasattr(arr,"files"):
            arr = arr[list(arr.files)[0]]
        return arr
    arr = np.array(Image.open(path))
    # If RGB panoptic, compress to 24-bit id
    if arr.ndim==3 and arr.shape[2] >= 3:
        arr = (arr[:,:,0].astype(np.uint32)<<16) | (arr[:,:,1].astype(np.uint32)<<8) | arr[:,:,2].astype(np.uint32)
    return arr

def norm_box(xmin,ymin,xmax,ymax,W,H):
    cx = (xmin+xmax)/2.0 / W
    cy = (ymin+ymax)/2.0 / H
    w  = (xmax-xmin)/W
    h  = (ymax-ymin)/H
    return cx,cy,w,h

# Treat each session dir under SRC (depth=3 works with our downloader, but be robust)
# We'll consider any directory containing at least one image file a "session".
session_dirs = set()
for p in SRC.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        # session root = two levels up from the file (…/sessions/<ID>/…)
        # fallback: the deepest directory that contains both images and masks later
        parts = p.parts
        # find the index of "sessions" if present to anchor the session root
        if "sessions" in parts:
            idx = parts.index("sessions")
            if idx+1 < len(parts):
                session_dirs.add(pathlib.Path(*parts[:idx+2]))
        else:
            # fallback: go up 2-3 levels
            session_dirs.add(p.parent.parent if len(parts)>=2 else p.parent)

session_dirs = sorted(session_dirs)

# Random split by session
random.shuffle(session_dirs)
cut = max(1, int(0.8*len(session_dirs)))
train_sessions = set(session_dirs[:cut])

pairs = []
for sess in session_dirs:
    # Build stem->path maps (prefer hinted paths by weighting later)
    img_map = {}
    img_score = {}
    for f in sess.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            stem = f.stem
            img_map[stem] = f
            img_score[stem] = 1 + (1 if IMG_HINT.search(str(f)) else 0)

    msk_map = {}
    msk_score = {}
    for f in sess.rglob("*"):
        if f.is_file() and f.suffix.lower() in MSK_EXTS:
            stem = f.stem
            msk_map[stem] = f
            msk_score[stem] = 1 + (1 if MSK_HINT.search(str(f)) else 0)

    shared = sorted(set(img_map) & set(msk_map))
    if not shared:
        continue

    for stem in shared:
        pairs.append((sess, img_map[stem], msk_map[stem], img_score.get(stem,1)+msk_score.get(stem,1)))

# Sort by “quality” (hint hits first) to bias toward left/seg folders when present
pairs.sort(key=lambda x: x[3], reverse=True)

n_empty = 0
for sess, img_p, msk_p, _score in tqdm(pairs, desc="Converting"):
    try:
        img = Image.open(img_p).convert("RGB")
    except Exception:
        continue
    W,H = img.size
    arr = load_mask(msk_p)

    if IDS is None:
        fg = (arr != 0)
    else:
        fg = np.isin(arr, np.array(sorted(IDS), dtype=arr.dtype))

    lab = label(fg.astype(np.uint8), connectivity=1)
    props = regionprops(lab)
    boxes = []
    for r in props:
        ymin,xmin,ymax,xmax = r.bbox
        if (xmax-xmin)<8 or (ymax-ymin)<8:
            continue
        cx,cy,w,h = norm_box(xmin,ymin,xmax,ymax,W,H)
        boxes.append((0,cx,cy,w,h))  # class 0 = obstacle

    split = "train" if sess in train_sessions else "val"
    out_img = OUT/f"images/{split}/{img_p.stem}{img_p.suffix.lower()}"
    out_lbl = OUT/f"labels/{split}/{img_p.stem}.txt"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_p, out_img)
    if boxes:
        with open(out_lbl,"w") as f:
            for c,cx,cy,w,h in boxes:
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    else:
        n_empty += 1
        open(out_lbl,"w").close()

print(f"Done. Sessions: {len(session_dirs)}  Pairs: {len(pairs)}  Empty-label images: {n_empty}")
