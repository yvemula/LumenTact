import cv2
import numpy as np

def draw_dets(frame, dets):
    h,w = frame.shape[:2]
    for d in dets:
        cx,cy,bw,bh = d["bbox"]
        x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
        x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f"{d['cls']} {d['conf']:.2f}", (x1,max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
    # front arc
    cv2.rectangle(frame,(int(0.3*w),0),(int(0.7*w),int(0.7*h)),(255,255,0),1)
    return frame

def overlay_traversable(frame, trav_mask):
    if trav_mask is None: return frame
    overlay = frame.copy()
    col = np.zeros_like(frame)
    col[...,1] = 180  # green-ish
    alpha = (trav_mask.astype(np.float32)/255.0)[...,None]*0.35
    out = cv2.convertScaleAbs(overlay*(1-alpha) + col*alpha)
    return out

def put_hud(frame, action, fps):
    cv2.putText(frame, f"Action: {action}", (8,24),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2,cv2.LINE_AA)
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (8,52),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2,cv2.LINE_AA)
    return frame
