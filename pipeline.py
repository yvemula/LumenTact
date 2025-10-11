import time, cv2
from detectors import TinyDetector
from decision import decide_action
from haptics import play
from viz import draw_dets, put_hud

def run(video_path:str, save_out:str|None=None, show=True):
    det = TinyDetector()
    cap = cv2.VideoCapture(video_path)
    fps_ema, alpha = None, 0.2
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        t0 = time.time()

        dets = det(frame)                  # detection
        action, strength = decide_action(dets, fps_ema or 15)  # decision
        play(action, strength)             # haptics emulator (non-blocking patterns kept short)

        out = draw_dets(frame.copy(), dets)
        out = put_hud(out, action, fps_ema)

        if show:
            cv2.imshow("WearableNav (software MVP)", out)
            if cv2.waitKey(1) & 0xFF == 27: break   # ESC quits

        if save_out:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_out, fourcc, 20.0, (out.shape[1], out.shape[0]))
            writer.write(out)

        dt = max(1e-3, time.time()-t0)
        inst_fps = 1.0/dt
        fps_ema = inst_fps if fps_ema is None else (1-alpha)*fps_ema + alpha*inst_fps

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
