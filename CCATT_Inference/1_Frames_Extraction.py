#!/usr/bin/env python3
"""
1_Frames_Extraction.py
----------------------
Extract (sample) frames from videos in a directory.

Outputs:
  <output>/<video_stem>/frame_<frame_idx:06d>.jpg

Notes:
- Uses OpenCV VideoCapture (fast, no ffmpeg dependency).
- Supports either (a) saving every Nth frame or (b) approximate FPS sampling.
"""

import os
import argparse
import cv2
from pathlib import Path
from typing import Optional


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def extract_frames_every_n(video_path: str, out_dir: str, every_n: int,
                           resize_w: Optional[int], resize_h: Optional[int],
                           jpg_quality: int) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "status": "fail_open", "saved": 0}

    saved = 0
    frame_idx = 0

    # jpg quality param
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if every_n <= 1 or (frame_idx % every_n == 0):
            if resize_w is not None and resize_h is not None:
                frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, encode_params)
            saved += 1

        frame_idx += 1

    cap.release()
    return {"video": video_path, "status": "ok", "saved": saved}


def extract_frames_by_fps(video_path: str, out_dir: str, target_fps: float,
                          resize_w: Optional[int], resize_h: Optional[int],
                          jpg_quality: int) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "status": "fail_open", "saved": 0}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        # fall back to every frame
        src_fps = 0.0

    # compute stride: save approximately target_fps frames per second
    if src_fps > 0:
        stride = max(1, int(round(src_fps / target_fps)))
    else:
        stride = 1

    cap.release()
    return extract_frames_every_n(video_path, out_dir, stride, resize_w, resize_h, jpg_quality)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos in a directory.")
    parser.add_argument("--input", required=True, help="Directory containing videos (.mp4/.3gp).")
    parser.add_argument("--output", required=True, help="Directory to write extracted frames.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--every_n", type=int, default=1,
                       help="Save every Nth frame (default=1 => save all frames).")
    group.add_argument("--fps", type=float, default=None,
                       help="Approximate sampling rate in frames/sec (e.g., 2.0).")

    parser.add_argument("--resize_w", type=int, default=None, help="Resize width (optional).")
    parser.add_argument("--resize_h", type=int, default=None, help="Resize height (optional).")
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality (0-100).")

    args = parser.parse_args()

    in_dir = Path(args.input)
    out_root = Path(args.output)
    ensure_dir(str(out_root))

    video_paths = []
    for ext in ("*.mp4", "*.3gp", "*.mov", "*.mkv", "*.avi"):
        video_paths.extend(in_dir.glob(ext))

    if not video_paths:
        raise SystemExit(f"No videos found in: {in_dir}")

    resize_w = args.resize_w
    resize_h = args.resize_h
    if (resize_w is None) ^ (resize_h is None):
        raise SystemExit("If you set --resize_w you must also set --resize_h (and vice versa).")

    print(f"[Frames] input={in_dir}")
    print(f"[Frames] output={out_root}")
    print(f"[Frames] videos={len(video_paths)}")

    for vp in sorted(video_paths):
        video_stem = vp.stem
        out_dir = out_root / video_stem
        ensure_dir(str(out_dir))

        if args.fps is not None:
            result = extract_frames_by_fps(str(vp), str(out_dir), args.fps,
                                           resize_w, resize_h, args.jpg_quality)
        else:
            result = extract_frames_every_n(str(vp), str(out_dir), args.every_n,
                                            resize_w, resize_h, args.jpg_quality)

        print(f"[Frames] {video_stem}: status={result['status']} saved={result['saved']}")


if __name__ == "__main__":
    main()
