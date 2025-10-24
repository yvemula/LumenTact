import cv2
import numpy as np

def overlay_traversable(frame, trav_mask):
    if trav_mask is None:
        return frame
    overlay = frame.copy()
    col = np.zeros_like(frame)
    col[...,1] = 180  # green-ish
    alpha = (trav_mask.astype(np.float32)/255.0)[...,None]*0.35
    out = cv2.convertScaleAbs(overlay*(1-alpha) + col*alpha)
    return out

def put_hud(frame, action, fps):
    cv2.putText(frame, f"Action: {action}", (8,24), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (8,52), cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2,cv2.LINE_AA)
    return frame
