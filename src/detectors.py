from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

class SanpoTraversableSeg:
    """
    ONNX model that outputs either a 1-channel traversable probability or multi-class logits.
    If missing, fall back to a conservative heuristic mask (bottom 40% = traversable).
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
            mask = np.zeros((H, W), np.uint8)
            mask[int(H*0.6):, :] = 255
            return mask

        iw, ih = self.input_size
        img = cv2.resize(frame_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(img, (2, 0, 1))[None, ...]  # 1x3xH xW

        inputs = {self.session.get_inputs()[0].name: x}
        out = self.session.run(None, inputs)[0]  # expect 1x1xHxW or 1xCxHxW

        if out.ndim == 4 and out.shape[1] == 1:
            prob = out[0,0]  # HxW
        elif out.ndim == 4 and out.shape[1] > 1:
            # Assume class id==1 is traversable (adjust during export)
            cls_map = out[0].argmax(0).astype(np.float32)
            prob = (cls_map == 1).astype(np.float32)
        else:
            prob = np.zeros((ih, iw), np.float32)

        prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (prob > 0.5).astype(np.uint8) * 255
        return mask


class MonoDepth:
    """
    MiDaS small via torch.hub; optional. Returns normalized inverse-depth-ish map in [0..1].
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
        if self.model is None:
            return None
        import torch
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.tf(img).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()
        d = pred.astype(np.float32)
        d = (d - d.min()) / max(1e-6, (d.max() - d.min()))
        return d
