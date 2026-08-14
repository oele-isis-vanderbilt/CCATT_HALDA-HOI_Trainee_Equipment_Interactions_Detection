#!/usr/bin/env python3
"""
Sample low-confidence frames and split HOI labels into high-confidence vs uncertain.

Rules:
1) Sample N frames uniformly at random from low_conf_frames.txt (default 800, or all if fewer).
2) From CDN labels, keep only entries whose frame is in the sampled set.
   - Require subject_score >= subj_thresh and object_score >= obj_thresh.
   - High-confidence HOI if verb_score >= pos_thr OR verb_score <= neg_thr OR center distance(subject, object) > dist_thresh.
   - Otherwise, mark as uncertain (still must meet subject/object score thresholds).
3) Save:
   - sampled_frames.txt
   - sampled_high_conf_labels.csv
   - sampled_uncertain_labels.csv
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd


def center(box):
    x1, y1, x2, y2 = box
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


def parse_box(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def main():
    ap = argparse.ArgumentParser(description="Sample low-confidence frames and split HOIs.")
    ap.add_argument("--low_conf_frames", type=Path, required=True, help="Path to low_conf_frames.txt produced earlier.")
    ap.add_argument("--labels_csv", type=Path, required=True, help="cdn_labels.csv from advPhase2 autolabel.")
    ap.add_argument("--output_dir", type=Path, required=True, help="Directory to write outputs.")
    ap.add_argument("--sample_size", type=int, default=800, help="Number of frames to sample (use all if fewer).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--pos_thr", type=float, default=0.90, help="Verb score >= pos_thr -> high confidence verb 117.")
    ap.add_argument("--neg_thr", type=float, default=0.10, help="Verb score <= neg_thr -> high confidence negative (57).")
    ap.add_argument("--dist_thr", type=float, default=150.0, help="If center distance > dist_thr -> treat as high-confidence negative.")
    ap.add_argument("--subj_thresh", type=float, default=0.5, help="Min subject score.")
    ap.add_argument("--obj_thresh", type=float, default=0.5, help="Min object score.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    low_list = [l.strip() for l in args.low_conf_frames.read_text().splitlines() if l.strip()]
    if not low_list:
        raise SystemExit("No frames listed in low_conf_frames file.")

    if len(low_list) <= args.sample_size:
        sampled = low_list
    else:
        sampled = rng.sample(low_list, args.sample_size)

    df = pd.read_csv(args.labels_csv)
    if df.empty:
        raise SystemExit("Labels CSV is empty.")

    df["verb_score"] = df["verb_score"].astype(float)
    df["object_score"] = df["object_score"].astype(float)
    df["subject_score"] = df["subject_score"].astype(float)

    df = df[df["frame_file"].isin(sampled)]

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
    (args.output_dir / "sampled_frames.txt").write_text("\n".join(sorted(sampled)))
    high_df.to_csv(args.output_dir / "sampled_high_conf_labels.csv", index=False)
    uncertain_df.to_csv(args.output_dir / "sampled_uncertain_labels.csv", index=False)

    print(f"[Sample] frames total low={len(low_list)} sampled={len(sampled)}")
    print(f"[Sample] high_conf HOIs={len(high_df)} -> {args.output_dir/'sampled_high_conf_labels.csv'}")
    print(f"[Sample] uncertain HOIs={len(uncertain_df)} -> {args.output_dir/'sampled_uncertain_labels.csv'}")


if __name__ == "__main__":
    main()
