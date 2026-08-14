#!/usr/bin/env python3
"""
Filter out blurry frames (Laplacian variance) and optionally sample N frames.

Example:
  python 2_filter_and_sample_frames.py \\
    --input_dir /path/to/frames/video123 \\
    --output_dir /path/to/clear_frames/video123 \\
    --threshold 100 \\
    --sample 2000 \\
    --sample_out /path/to/sampled_frames/video123 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import random
import os
import shutil
from pathlib import Path

import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


def laplacian_var(image_path: Path) -> float:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def random_sample_paths(paths: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if n <= 0 or n >= len(paths):
        return paths
    paths_copy = paths[:]
    rng.shuffle(paths_copy)
    return paths_copy[:n]


def filter_non_blurry_images(
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    recursive: bool = False,
) -> int:
    ensure_dir(output_dir)
    images: list[Path] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(Path(root) / fn)
    else:
        images = list_images(input_dir)
    kept = 0
    for img_path in images:
        if laplacian_var(img_path) >= threshold:
            dest = output_dir / img_path.name
            ensure_dir(dest.parent)
            shutil.copy2(img_path, dest)
            kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove blurry images and optionally sample N frames.")
    ap.add_argument("--input_dir", required=True, type=Path, help="Directory of input frames.")
    ap.add_argument("--output_dir", required=True, type=Path, help="Directory to write non-blurry frames.")
    ap.add_argument("--threshold", type=float, default=100.0, help="Laplacian variance threshold (lower = blurrier).")
    ap.add_argument("--sample", type=int, default=0, help="Sample N frames (0 disables sampling).")
    ap.add_argument("--sample_out", type=Path, default=None, help="Output dir for sampled frames (defaults to output_dir).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    args = ap.parse_args()

    kept = filter_non_blurry_images(args.input_dir, args.output_dir, threshold=args.threshold, recursive=False)
    print(f"[BlurFilter] kept={kept} -> {args.output_dir}")

    if args.sample and args.sample > 0:
        sample_out = args.sample_out or args.output_dir
        ensure_dir(sample_out)
        imgs = list_images(args.output_dir)
        selected = random_sample_paths(imgs, args.sample, args.seed)
        for p in selected:
            shutil.copy2(p, sample_out / p.name)
        print(f"[Sample] selected={len(selected)} from {len(imgs)} -> {sample_out}")


if __name__ == "__main__":
    main()
