import argparse
from pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--save", default=None, help="Optional path to save annotated MP4")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--sanpo-onnx", default="models/sanpo_traversable.onnx")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--log", default="runs/last_run.csv")
    ap.add_argument("--haptic-cooldown-s", type=float, default=0.5)
    ap.add_argument("--start-paused", action="store_true")
    args = ap.parse_args()

    run(
        video_path=args.video,
        save_out=args.save,
        show=not args.no_show,
        sanpo_onnx=args.sanpo_onnx,
        use_depth=not args.no_depth,
        log_csv=args.log,
        haptic_cooldown_s=args.haptic_cooldown_s,
        start_paused=args.start_paused
    )
