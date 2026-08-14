#!/usr/bin/env python3
"""
Remove blurry frames and sample a fixed number per video directory.

Usage:
  python 2_sample_nonblurry.py \
    --input_root /path/to/frames_root \
    --output_root /path/to/clear_frames_root \
    --sample_out_root /path/to/sampled_frames_root \
    --threshold 150 \
    --sample 500 \
    --seed 42 \
    --roi_config camera_rois.json \
    --camera_view 1-4 \
    --equipment_types "IV Pump" MV \
    --roi_threshold 200

If --sample_out_root is omitted, sampled frames are written under output_root.
Handles either a flat directory of images or multiple subdirectories (one per video).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_rois(config_path: Path, camera_view: str, equipment_types: list[str] | None) -> list[list[float]]:
    cfg = json.loads(config_path.read_text())
    view_cfg = cfg.get(camera_view, {})
    if not view_cfg:
        raise SystemExit(f"No camera view '{camera_view}' in ROI config {config_path}")
    eq_types = equipment_types or list(view_cfg.keys())
    rois: list[list[float]] = []
    for eq in eq_types:
        if eq not in view_cfg:
            print(f"[WARN] Equipment '{eq}' missing for camera view '{camera_view}' in {config_path}; skipping ROI check for it.")
            continue
        coords = [float(v) for v in view_cfg[eq]]
        if len(coords) != 4:
            raise SystemExit(f"ROI for '{eq}' must have 4 values, got {coords}")
        rois.append(coords)
    return rois


def list_images(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])


# --- Camera view heuristics -------------------------------------------------
VIEW_ALIAS_TO_ROI = {
    "view2_v1": "old_cam",
    "view2_v2": "new_cam",
    "view1_v1": "old_pan",
    "view1_v2": "new_pan",
}


def get_camera_view(fname: str):
    fname = (fname or "").lower()
    if "old_cam" in fname:
        return "view2_v1"
    elif "new_cam" in fname:
        return "view2_v2"
    elif "old_pan" in fname:
        return "view1_v1"
    elif "new_pan" in fname:
        return "view1_v2"
    # Additional heuristics for names like "cam16_v2", "pan_v1", "pan2", etc.
    if "cam" in fname:
        if "v2" in fname or "_2" in fname:
            return "view2_v2"
        if "v1" in fname or "_1" in fname:
            return "view2_v1"
    if "pan" in fname:
        if "v2" in fname or "_2" in fname:
            return "view1_v2"
        if "v1" in fname or "_1" in fname:
            return "view1_v1"
    return None


def resolve_roi_view(name: str) -> str | None:
    view = get_camera_view(name)
    if view is None:
        return None
    return VIEW_ALIAS_TO_ROI.get(view, view)


def laplacian_var_gray(gray_img: np.ndarray, bbox: list[float] | None = None) -> float:
    """Compute Laplacian variance on the full image or a cropped ROI."""
    if gray_img is None or gray_img.size == 0:
        return 0.0
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray_img.shape[1], x2)
        y2 = min(gray_img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        gray_img = gray_img[y1:y2, x1:x2]
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def random_sample_paths(paths: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if n <= 0 or n >= len(paths):
        return paths
    paths_copy = paths[:]
    rng.shuffle(paths_copy)
    return paths_copy[:n]


def check_one_blur_job(job):
    img_path, thr, rois, roi_thr = job
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return ("skip", img_path)
    global_score = laplacian_var_gray(gray)
    if global_score < thr:
        return ("reject_global", img_path)
    if rois:
        for roi in rois:
            score = laplacian_var_gray(gray, roi)
            if score < roi_thr:
                return ("reject_roi", img_path)
    return ("keep", img_path)


def filter_non_blurry_images(
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    roi_boxes: list[list[float]] | None,
    roi_threshold: float,
    workers: int,
) -> dict:
    ensure_dir(output_dir)
    images = list_images(input_dir)
    kept = 0
    rejected_global = 0
    rejected_roi = 0
    if not images:
        return {"kept": 0, "rejected_global": 0, "rejected_roi": 0, "total": 0}

    jobs = [(str(p), threshold, roi_boxes, roi_threshold) for p in images]

    if workers <= 1:
        results = map(check_one_blur_job, jobs)
    else:
        with Pool(processes=workers) as pool:
            results = pool.imap_unordered(check_one_blur_job, jobs, chunksize=64)

    for status, img_path in results:
        if status == "reject_global":
            rejected_global += 1
            continue
        if status == "reject_roi":
            rejected_roi += 1
            continue
        if status != "keep":
            continue
        src = Path(img_path)
        dest = output_dir / src.name
        ensure_dir(dest.parent)
        shutil.copy2(src, dest)
        kept += 1
    return {"kept": kept, "rejected_global": rejected_global, "rejected_roi": rejected_roi, "total": len(images)}


def process_video_dir(
    input_root: Path,
    video_dir: Path,
    clear_root: Path,
    sample_root: Path,
    threshold: float,
    roi_boxes: list[list[float]] | None,
    roi_threshold: float,
    sample: int,
    seed: int,
    workers: int,
) -> None:
    # Preserve full relative path (e.g., CAM16/<video>, PAN/<video>) to avoid
    # cross-camera collisions when different source branches share basename.
    try:
        rel_dir = video_dir.relative_to(input_root)
    except ValueError:
        rel_dir = Path(video_dir.name)
    clear_dir = clear_root / rel_dir
    sample_dir = sample_root / rel_dir

    stats = filter_non_blurry_images(video_dir, clear_dir, threshold, roi_boxes, roi_threshold, workers)
    print(
        f"[BlurFilter] {video_dir.name}: kept={stats['kept']}/{stats['total']} "
        f"(global_rejects={stats['rejected_global']}, roi_rejects={stats['rejected_roi']}) -> {clear_dir}"
    )

    if sample and sample > 0:
        ensure_dir(sample_dir)
        imgs = list_images(clear_dir)
        selected = random_sample_paths(imgs, sample, seed)
        for p in selected:
            shutil.copy2(p, sample_dir / p.name)
        print(f"[Sample] {video_dir.name}: selected={len(selected)} from {len(imgs)} -> {sample_dir}")


def discover_video_dirs(root: Path) -> list[Path]:
    """Return all subdirectories that actually contain images (recurses)."""
    video_dirs: list[Path] = []
    if list_images(root):
        video_dirs.append(root)
    for p in root.rglob("*"):
        if p.is_dir() and list_images(p):
            video_dirs.append(p)
    # Remove duplicates and sort for stability
    uniq = sorted({p for p in video_dirs})
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove blurry frames and sample N frames per video directory.")
    ap.add_argument("--input_root", required=True, type=Path, help="Root containing per-video frame directories (or flat images).")
    ap.add_argument("--output_root", required=True, type=Path, help="Root to write non-blurry frames.")
    ap.add_argument("--sample_out_root", type=Path, default=None, help="Root for sampled frames (default: same as output_root).")
    ap.add_argument("--threshold", type=float, default=400.0, help="Laplacian variance threshold (lower = blurrier).")
    ap.add_argument("--roi_config", type=Path, help="Optional: camera_rois.json to apply ROI-only blur check.")
    ap.add_argument("--camera_view", type=str, help="Camera view key in ROI config (e.g., new_cam/old_cam/new_pan/old_pan). If omitted, view is inferred from directory name.")
    ap.add_argument("--equipment_types", nargs="+", help="Equipment names to use from ROI config (default: all for view).")
    ap.add_argument("--roi_threshold", type=float, default=400.0, help="Laplacian variance threshold inside each ROI (aggressive).")
    ap.add_argument("--sample", type=int, default=0, help="Number of frames to sample per video (0 disables).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    ap.add_argument("--workers", type=int, default=4, help="Number of worker processes for blur filtering.")
    args = ap.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    sample_root = args.sample_out_root or output_root

    ensure_dir(output_root)
    ensure_dir(sample_root)

    video_dirs = discover_video_dirs(input_root)
    if not video_dirs:
        raise SystemExit(f"No images or subdirectories found in {input_root}")

    # Cache of loaded ROI boxes per view to avoid repeated file reads.
    roi_cache: dict[str, list[list[float]]] = {}

    for vid_dir in video_dirs:
        roi_boxes = None
        view_key = args.camera_view

        if not view_key and args.roi_config:
            view_key = resolve_roi_view(vid_dir.name)

        if view_key:
            if view_key not in roi_cache:
                try:
                    roi_cache[view_key] = load_rois(args.roi_config, view_key, args.equipment_types)
                except SystemExit as e:
                    print(f"[WARN] ROI view '{view_key}' not found in {args.roi_config}; skipping ROI checks for {vid_dir.name}")
                    roi_cache[view_key] = []
            roi_boxes = roi_cache.get(view_key) or None

        process_video_dir(
            input_root=input_root,
            video_dir=vid_dir,
            clear_root=output_root,
            sample_root=sample_root,
            threshold=args.threshold,
            roi_boxes=roi_boxes,
            roi_threshold=args.roi_threshold,
            sample=args.sample,
            seed=args.seed,
            workers=max(1, int(args.workers)),
        )


if __name__ == "__main__":
    main()
