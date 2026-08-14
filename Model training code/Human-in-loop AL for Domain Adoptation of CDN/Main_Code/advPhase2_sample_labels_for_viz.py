#!/usr/bin/env python3
"""
Sample a small number of high-confidence frames and keep only labels from those
frames for visualization before/after NMS.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample labels for visualization from high-confidence frames.")
    ap.add_argument("--labels_csv", type=Path, required=True, help="Input labels CSV.")
    ap.add_argument("--frames_txt", type=Path, required=True, help="Frame list file (one frame path per line).")
    ap.add_argument("--output_csv", type=Path, required=True, help="Output sampled labels CSV.")
    ap.add_argument("--output_frames_txt", type=Path, help="Optional output sampled frame list.")
    ap.add_argument("--samples", type=int, default=50, help="Number of frames to sample.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument(
        "--min-positive-frames",
        type=int,
        default=20,
        help="Minimum sampled frames that must contain at least one positive verb.",
    )
    ap.add_argument(
        "--positive-verbs",
        type=int,
        nargs="+",
        default=[117, 118],
        help="Verb ids considered positive for frame selection.",
    )
    args = ap.parse_args()

    if not args.labels_csv.exists():
        raise SystemExit(f"Labels CSV not found: {args.labels_csv}")
    if not args.frames_txt.exists():
        raise SystemExit(f"Frames file not found: {args.frames_txt}")

    frames = [line.strip() for line in args.frames_txt.read_text().splitlines() if line.strip()]
    if not frames:
        raise SystemExit(f"No frames found in {args.frames_txt}")

    df = pd.read_csv(args.labels_csv)
    if df.empty:
        raise SystemExit(f"Labels CSV is empty: {args.labels_csv}")
    if "frame_file" not in df.columns:
        raise SystemExit(f"Missing frame_file column in {args.labels_csv}")
    if "verb_id" not in df.columns:
        raise SystemExit(f"Missing verb_id column in {args.labels_csv}")

    rng = random.Random(args.seed)

    # Restrict frame candidates to those listed in frames_txt.
    frame_set = set(frames)
    df_in = df[df["frame_file"].isin(frame_set)].copy()
    if df_in.empty:
        raise SystemExit("No label rows match frames listed in frames_txt.")

    sample_n = min(args.samples, len(frames))
    positive_verbs = {int(v) for v in args.positive_verbs}
    positive_rows = df_in[df_in["verb_id"].astype(int).isin(positive_verbs)]
    positive_frames = list(set(positive_rows["frame_file"].astype(str).tolist()))
    all_frames = list(frame_set)

    min_pos = max(0, min(args.min_positive_frames, sample_n))
    n_pos = min(min_pos, len(positive_frames))
    chosen_pos = rng.sample(positive_frames, n_pos) if n_pos > 0 else []

    remaining_pool = list(set(all_frames) - set(chosen_pos))
    need_rest = sample_n - len(chosen_pos)
    chosen_rest = rng.sample(remaining_pool, need_rest) if need_rest > 0 else []
    sampled_frames = chosen_pos + chosen_rest

    out_df = df[df["frame_file"].isin(sampled_frames)].copy()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    if args.output_frames_txt:
        args.output_frames_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_frames_txt.write_text("\n".join(sorted(sampled_frames)))

    print(
        f"[VizSample] sampled_frames={len(sampled_frames)} positive_frames_in_sample={len(chosen_pos)} "
        f"matched_rows={len(out_df)} -> {args.output_csv}"
    )


if __name__ == "__main__":
    main()
