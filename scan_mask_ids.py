# scan_mask_ids.py
import sys, pathlib, numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path("sanpo_subset_auto")
mask_files = list(ROOT.rglob("masks/**/*"))
mask_files = [p for p in mask_files if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".webp",".npy",".npz"}]

counts = Counter()
for p in tqdm(mask_files, desc="Scanning"):
    if p.suffix.lower() in (".npy",".npz"):
        arr = np.load(p)
        if hasattr(arr, "files"): arr = arr[list(arr.files)[0]]
    else:
        arr = np.array(Image.open(p))
    if arr.ndim==3:  # RGB panoptic → compress to 24-bit ID
        arr = arr[:,:,0].astype(np.uint32)<<16 | arr[:,:,1].astype(np.uint32)<<8 | arr[:,:,2].astype(np.uint32)
    ids, c = np.unique(arr, return_counts=True)
    counts.update(dict(zip(ids.tolist(), c.tolist())))

top = counts.most_common(30)
for val, c in top:
    print(f"id={val} count={c}")
