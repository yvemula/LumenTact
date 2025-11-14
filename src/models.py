from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


# -----------------------------
# Depth: MiDaS (DPT*) via torch.hub
# -----------------------------
class MonoDepth:
    """Monocular depth via MiDaS. Returns float32 depth in [0,1] (larger ≈ farther)."""
    def __init__(self, model_type: str = "DPT_Large", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.tf = self.transforms.dpt_transform if "DPT" in model_type else self.transforms.small_transform

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.tf(rgb).to(self.device)
        pred = self.model(inp)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze(1)[0]
        depth = pred.float().cpu().numpy()
        dmin, dmax = np.percentile(depth, 1), np.percentile(depth, 99)
        depth01 = (depth - dmin) / max(1e-6, (dmax - dmin))
        return np.clip(depth01, 0, 1).astype(np.float32)


# -----------------------------
# Traversable segmentation using YOLOv12
# - Prefer YOLOv12-seg if weights present
# - Otherwise use YOLOv12 detect and build masks from boxes
# -----------------------------
class SanpoTraversableSeg:
    """
    Steps:
      1) YOLOv12 (seg if available; else detect) → obstacle regions
      2) Depth percentile → 'open' regions
      3) Remove/dilate obstacles from 'open'
      4) Keep largest component touching bottom (assume ground)
    Returns uint8 mask {0,1}.
    """
    # Common obstacle classes from COCO; adjust as needed.
    OBSTACLE_NAMES = {
        "person","bicycle","car","motorcycle","bus","truck","train",
        "bench","chair","couch","bed","dining table","potted plant",
        "tv","laptop","keyboard","mouse","bottle","cup","backpack","suitcase",
        "boat","umbrella","skateboard","surfboard","snowboard","sports ball",
        "dog","cat","bird","horse","sheep","cow"
    }

    def __init__(
        self,
        yolo_weights: str = "yolo12n.pt",   # detection model (available)
        yolo_seg_weights: str | None = None,  # e.g. "yolo12n-seg.pt" if you have one
        conf: float = 0.25,
        iou: float = 0.5,
        far_quantile: float = 0.55,
        obst_dilate_px: int = 9,
        morph_ksize: int = 5
    ):
        # Try to load seg model if provided; otherwise detection
        self.seg_model = YOLO(yolo_seg_weights) if yolo_seg_weights else None
        self.det_model = None if self.seg_model else YOLO(yolo_weights)
        self.conf = conf
        self.iou = iou
        self.far_q = far_quantile
        self.obst_dilate_px = obst_dilate_px
        self.morph_ksize = morph_ksize

        model = self.seg_model or self.det_model
        names = model.names
        self.class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
        self.obstacle_ids = {i for i, n in enumerate(self.class_names) if n in self.OBSTACLE_NAMES}

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray, depth01: np.ndarray) -> np.ndarray:
        h, w = depth01.shape[:2]
        imgsz = max(512, int(max(h, w) / 2))

        # 1) YOLOv12 pass
        if self.seg_model is not None:
            res = self.seg_model.predict(source=bgr, conf=self.conf, iou=self.iou, verbose=False, imgsz=imgsz)
            obstacle_mask = np.zeros((h, w), np.uint8)
            if len(res):
                r0 = res[0]
                if getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None:
                    masks = r0.masks.data  # (N,Hm, Wm)
                    clses = r0.boxes.cls.int().cpu().numpy().tolist()
                    for i, cls_id in enumerate(clses):
                        if cls_id in self.obstacle_ids:
                            m = masks[i].cpu().numpy().astype(np.uint8)
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                            obstacle_mask = np.maximum(obstacle_mask, m)
        else:
            # detection-only → build obstacle mask from boxes
            res = self.det_model.predict(source=bgr, conf=self.conf, iou=self.iou, verbose=False, imgsz=imgsz)
            obstacle_mask = np.zeros((h, w), np.uint8)
            if len(res):
                r0 = res[0]
                if getattr(r0, "boxes", None) is not None and getattr(r0.boxes, "xyxy", None) is not None:
                    boxes = r0.boxes.xyxy.cpu().numpy()
                    clses = r0.boxes.cls.int().cpu().numpy().tolist()
                    for (x1, y1, x2, y2), cls_id in zip(boxes, clses):
                        if cls_id in self.obstacle_ids:
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
                            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
                            obstacle_mask[y1:y2, x1:x2] = 1

        if self.obst_dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.obst_dilate_px, self.obst_dilate_px))
            obstacle_mask = cv2.dilate(obstacle_mask, k, iterations=1)

        # 2) depth → open mask
        thr = float(np.quantile(depth01, self.far_q))
        open_mask = (depth01 >= thr).astype(np.uint8)

        # 3) remove obstacles
        base = cv2.bitwise_and(open_mask, (1 - obstacle_mask))

        # 4) cleanup & keep largest component touching bottom
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_ksize, self.morph_ksize))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, k2, iterations=1)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k2, iterations=2)

        cnts, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(base)
        best_area, best_idx = 0, -1
        for i, c in enumerate(cnts):
            x, y, ww, hh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            touches_bottom = (y + hh) >= (h - 2)
            if touches_bottom and area > best_area:
                best_area, best_idx = area, i
        if best_idx >= 0:
            cv2.drawContours(keep, cnts, best_idx, 1, thickness=-1)
        else:
            keep = base

        return keep.astype(np.uint8)


def overlay_mask(bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.5, color=(0, 255, 0)) -> np.ndarray:
    """Overlay 0/1 mask in a chosen color over the BGR image."""
    h, w = mask01.shape
    color_img = np.zeros((h, w, 3), np.uint8)
    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]
    mask255 = (mask01 * 255).astype(np.uint8)
    overlay = cv2.addWeighted(bgr, 1.0, color_img, alpha, 0.0)
    out = np.where(mask255[..., None] > 0, overlay, bgr)
    return out
