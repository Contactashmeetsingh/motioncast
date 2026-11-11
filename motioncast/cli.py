"""Command line: `python -m motioncast download` and `python -m motioncast run VIDEO`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from motioncast.config import MODELS_DIR, VIDEOS_DIR, PipelineConfig

# Roboflow `sports` pretrained checkpoints and sample clips (public Google Drive links).
MODELS = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
}
SAMPLE_VIDEOS = {
    "0bfacc_0.mp4": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
    "2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
    "08fd33_0.mp4": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
    "573e61_0.mp4": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
    "121364_0.mp4": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu",
}


def download(all_videos: bool = False) -> None:
    import gdown

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    videos = SAMPLE_VIDEOS if all_videos else dict(list(SAMPLE_VIDEOS.items())[:1])
    for folder, files in ((MODELS_DIR, MODELS), (VIDEOS_DIR, videos)):
        for name, file_id in files.items():
            dest = folder / name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"ok      {dest}")
                continue
            print(f"fetch   {dest}")
            if gdown.download(id=file_id, output=str(dest), quiet=True) is None:
                sys.exit(f"Download failed for {name} (Google Drive quota?). Retry later.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="motioncast", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch pretrained models and a sample clip")
    d.add_argument("--all-videos", action="store_true", help="fetch all 5 sample clips")

    r = sub.add_parser("run", help="run the full pipeline on a video")
    r.add_argument("video", type=Path)
    r.add_argument("--out", type=Path, default=None, help="output dir (default outputs/<video name>)")
    r.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    r.add_argument("--start", type=int, default=0, help="first frame")
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--pitch-every", type=int, default=1, help="run pitch keypoints every N frames")
    r.add_argument("--fast-ball", action="store_true",
                   help="use the player model's ball class instead of the dedicated ball model")
    r.add_argument("--no-render", action="store_true", help="skip the annotated video")
    r.add_argument("--reuse", action="store_true",
                   help="reuse <out>/analysis.pkl from a previous run (skips model inference)")

    args = p.parse_args(argv)
    if args.cmd == "download":
        download(args.all_videos)
        return

    from motioncast.pipeline import run

    if not args.video.exists():
        sys.exit(f"No such video: {args.video}")
    cfg = PipelineConfig(device=args.device, start_frame=args.start, max_frames=args.max_frames,
                         pitch_every=args.pitch_every, use_ball_model=not args.fast_ball,
                         render=not args.no_render)
    out = args.out or Path("outputs") / args.video.stem
    paths = run(args.video, out, cfg, reuse_analysis=args.reuse)
    stats = json.loads(paths["stats"].read_text())
    print(f"\nWrote {out}/")
    for k, v in paths.items():
        print(f"  {k:9s} {v}")
    print(f"\n{stats['frames']} frames, {len(stats['players'])} tracks, "
          f"possession {stats['possession']['team_pct']}, coverage {stats['coverage']}")
    print(f"timings {stats['timings']}")
