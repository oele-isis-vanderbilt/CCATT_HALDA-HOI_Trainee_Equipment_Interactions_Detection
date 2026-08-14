#!/usr/bin/env python3
"""
Copy a few sample images per video directory for quick visualization.

Usage:
  python 2_viz_copy_samples.py --input_root /path/to/frames_root --output_root /path/to/viz_root --count 3
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy a few images per video directory for visualization.")
    ap.add_argument("--input_root", required=True, type=Path, help="Root with per-video image folders.")
    ap.add_argument("--output_root", required=True, type=Path, help="Root to write copied samples.")
    ap.add_argument("--count", type=int, default=3, help="Number of images to copy per video (default: 3).")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for video_dir in sorted(args.input_root.iterdir()):
        if not video_dir.is_dir():
            continue
        imgs = sorted(video_dir.glob("*.jpg"))[: args.count]
        if not imgs:
            continue
        out_dir = args.output_root / video_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out_dir / img.name)
            print(f"[Viz] copied {img} -> {out_dir/img.name}")


if __name__ == "__main__":
    main()
