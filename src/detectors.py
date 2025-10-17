from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- Detection (GuideDog) via Ultralytics YOLOv12 ---
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[WARN] Ultralytics not available; detection will be disabled.", e)

GUIDEDOG_RELEVANT = {
    # Collapse to nav-relevant classes; extend as your weights support
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "pole", "bench", "chair",
    "door", "stair", "traffic light", "stop sign", "crosswalk", "curb"
}

class GuideDogDetector:
    """
    Wraps a YOLOv12 checkpoint fine-tuned on GuideDog (or compatible).
    Falls back to yolo12n COCO if a custom model isn't found, but filters classes.
    """
    def __init__(self, model_path: str | None = None, conf: float = 0.35, imgsz: int = 640):
        if YOLO is None:
            self.model = None
        else:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # fallback to a general model (still filtered to relevant classes)
                self.model = YOLO("yolo12n.pt")
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame_bgr) -> List[Dict]:
        if self.model is None:
            return []
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        out = []
        for r in results:
            H, W = r.orig_shape
            for b in r.boxes:
                cls_id   = int(b.cls.item())
                cls_name = r.names.get(cls_id, str(cls_id))
                if cls_name not in GUIDEDOG_RELEVANT:
                    continue
                conf     = float(b.conf.item())
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cx = (x1+x2)/(2.0*W); cy = (y1+y2)/(2.0*H)
                w  = (x2-x1)/W;       h  = (y2-y1)/H
                out.append({"cls":cls_name, "conf":conf, "bbox":(cx,cy,w,h)})
        return out


# --- SANPO traversable segmentation (ONNX) ---
class SanpoTraversableSeg:
    """
    Expect an ONNX model that outputs per-pixel class logits or a 1-channel traversable mask.
    Put it at models/sanpo_traversable.onnx (export your SANPO head accordingly).
    If not present, we synthesize a naive free-space mask (lower half) as a safe fallback.
    """
    def __init__(self, onnx_path: str = "models/sanpo_traversable.onnx", input_size: Tuple[int,int]=(512,288)):
        self.onnx_path = onnx_path
        self.input_size = input_size  # (W, H)
        self.session = None
        try:
            import onnxruntime as ort
            if Path(onnx_path).exists():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(onnx_path, providers=providers)
            else:
                print(f"[WARN] SANPO ONNX not found at {onnx_path}. Using heuristic mask fallback.")
        except Exception as e:
            print("[WARN] onnxruntime not available; using heuristic mask fallback.", e)

    def __call__(self, frame_bgr) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        if self.session is None:
            # Heuristic: bottom 40% as traversable placeholder (very conservative)
            mask = np.zeros((H, W), np.uint8)
            mask[int(H*0.6):, :] = 255
            return mask

        # Preprocess
        iw, ih = self.input_size
        img = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(img, (2, 0, 1))[None, ...]  # 1x3xH xW

        inputs = {self.session.get_inputs()[0].name: x}
        out = self.session.run(None, inputs)[0]  # expect 1x1xH xW or 1xCxH xW

        if out.ndim == 4 and out.shape[1] == 1:
            prob = out[0,0]  # H x W
        elif out.ndim == 4 and out.shape[1] > 1:
            prob = out[0].argmax(0).astype(np.float32)  # class map
            # Assume class 'traversable' id==1 (adjust to your export!)
            prob = (prob == 1).astype(np.float32)
        else:
            # Unknown output; safest fallback
            prob = np.zeros((ih, iw), np.float32)

        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (prob > 0.5).astype(np.uint8) * 255
        return mask


# --- Optional monocular depth (MiDaS small) ---
class MonoDepth:
    """
    Light wrapper around MiDaS small via torch.hub (runs on CPU/GPU).
    Used for curb/step gradient & rough distance gating.
    """
    def __init__(self, model_type="MiDaS_small"):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device).eval()
            self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.tf = self.transforms.small_transform
        except Exception as e:
            print("[WARN] Could not load MiDaS; depth disabled.", e)
            self.model = None

    def __call__(self, frame_bgr) -> Optional[np.ndarray]:
        if self.model is None: return None
        import torch
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.tf(img).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        # MiDaS outputs inverse depth-ish; normalize
        d = pred.astype(np.float32)
        d = (d - d.min()) / max(1e-6, (d.max() - d.min()))
        return d
