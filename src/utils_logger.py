# src/utils_logger.py
import csv, time, cv2

class CSVLogger:
    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["ts","action","n_dets","names","confs"])
    def write(self, action, dets):
        ts = time.time()
        names = "|".join([d[0] for d in dets])
        confs = "|".join([f"{d[1]:.2f}" for d in dets])
        self.w.writerow([ts, action, len(dets), names, confs])
    def close(self):
        self.f.close()

class VideoWriter:
    def __init__(self, path, width, height, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vw = cv2.VideoWriter(path, fourcc, fps, (width, height))
    def write(self, frame): self.vw.write(frame)
    def close(self): self.vw.release()
