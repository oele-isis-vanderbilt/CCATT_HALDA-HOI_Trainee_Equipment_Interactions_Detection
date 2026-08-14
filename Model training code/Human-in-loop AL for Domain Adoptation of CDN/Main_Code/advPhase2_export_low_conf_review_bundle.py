#!/usr/bin/env python3
"""
Create a review bundle for manually checking sampled low-confidence frames.

Outputs:
- sampled_frames.txt
- sampled_all_labels.csv
- sampled_high_conf_labels.csv
- sampled_uncertain_labels.csv
- images/<relative frame path>
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import pandas as pd


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def parse_box(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def copy_frame(clear_root: Path, rel_frame: str, images_dir: Path) -> bool:
    src = clear_root / rel_frame
    dst = images_dir / rel_frame
    if not src.exists():
        # Fallback for callers that stored only basename in frame_file.
        matches = list(clear_root.rglob(Path(rel_frame).name))
        if len(matches) != 1:
            return False
        src = matches[0]
        dst = images_dir / src.relative_to(clear_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export sampled low-confidence review bundle.")
    ap.add_argument("--clear_root", type=Path, required=True, help="Root directory containing frame images.")
    ap.add_argument("--low_conf_frames", type=Path, required=True, help="Path to low_conf_frames.txt.")
    ap.add_argument("--labels_csv", type=Path, required=True, help="Path to labels CSV (typically cdn_labels.csv).")
    ap.add_argument("--output_dir", type=Path, required=True, help="Bundle output directory.")
    ap.add_argument("--sample_size", type=int, default=800, help="Number of low-confidence frames to sample.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--pos_thr", type=float, default=0.90, help="Verb score >= pos_thr -> high confidence.")
    ap.add_argument("--neg_thr", type=float, default=0.10, help="Verb score <= neg_thr -> high confidence negative.")
    ap.add_argument("--dist_thr", type=float, default=150.0, help="Distance threshold for high-confidence negative.")
    ap.add_argument("--subj_thresh", type=float, default=0.5, help="Min subject score.")
    ap.add_argument("--obj_thresh", type=float, default=0.5, help="Min object score.")
    args = ap.parse_args()

    if not args.low_conf_frames.exists():
        raise SystemExit(f"low_conf_frames file not found: {args.low_conf_frames}")
    if not args.labels_csv.exists():
        raise SystemExit(f"labels_csv not found: {args.labels_csv}")

    rng = random.Random(args.seed)
    low_list = [line.strip() for line in args.low_conf_frames.read_text().splitlines() if line.strip()]
    if not low_list:
        raise SystemExit("No frames listed in low_conf_frames file.")

    sampled = low_list if len(low_list) <= args.sample_size else rng.sample(low_list, args.sample_size)

    df = pd.read_csv(args.labels_csv)
    if df.empty:
        raise SystemExit(f"Labels CSV is empty: {args.labels_csv}")

    for col in ("verb_score", "object_score", "subject_score"):
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            raise SystemExit(f"Missing required column {col} in {args.labels_csv}")

    df = df[df["frame_file"].isin(sampled)].copy()

    def is_high(row):
        if row.subject_score < args.subj_thresh or row.object_score < args.obj_thresh:
            return False
        verb_score = row.verb_score
        if verb_score >= args.pos_thr or verb_score <= args.neg_thr:
            return True
        subj = parse_box(row.subject_box)
        obj = parse_box(row.object_box)
        cx1, cy1 = center(subj)
        cx2, cy2 = center(obj)
        dist = math.hypot(cx1 - cx2, cy1 - cy2)
        return dist > args.dist_thr

    high_mask = df.apply(is_high, axis=1)
    high_df = df[high_mask].copy()
    uncertain_df = df[~high_mask].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    copied = 0
    missing = 0
    for frame in sorted(sampled):
        ok = copy_frame(args.clear_root, frame, images_dir)
        if ok:
            copied += 1
        else:
            missing += 1

    (args.output_dir / "sampled_frames.txt").write_text("\n".join(sorted(sampled)))
    df.to_csv(args.output_dir / "sampled_all_labels.csv", index=False)
    high_df.to_csv(args.output_dir / "sampled_high_conf_labels.csv", index=False)
    uncertain_df.to_csv(args.output_dir / "sampled_uncertain_labels.csv", index=False)

    print(f"[ReviewBundle] low_total={len(low_list)} sampled={len(sampled)} copied={copied} missing={missing}")
    print(f"[ReviewBundle] all_rows={len(df)} -> {args.output_dir / 'sampled_all_labels.csv'}")
    print(f"[ReviewBundle] high_rows={len(high_df)} -> {args.output_dir / 'sampled_high_conf_labels.csv'}")
    print(f"[ReviewBundle] uncertain_rows={len(uncertain_df)} -> {args.output_dir / 'sampled_uncertain_labels.csv'}")
    print(f"[ReviewBundle] images -> {images_dir}")


if __name__ == "__main__":
    main()
