import argparse
from pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--save", default=None, help="Optional path to save annotated MP4")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    run(args.video, save_out=args.save, show=not args.no_show)
