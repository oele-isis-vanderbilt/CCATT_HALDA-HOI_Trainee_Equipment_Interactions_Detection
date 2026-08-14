#!/usr/bin/env python3
"""
Append high-confidence pseudo labels into a cumulative CSV shared across
active-learning cycles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Append high-confidence pseudo labels to a cumulative CSV.")
    ap.add_argument("--source", type=Path, required=True, help="Current cycle high_conf_labels.csv")
    ap.add_argument("--high_conf_frames", type=Path, required=True, help="Current cycle high_conf_frames.txt")
    ap.add_argument("--dest", type=Path, required=True, help="Cumulative pseudo labels CSV")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"High-confidence labels not found: {args.source}")
    if not args.high_conf_frames.exists():
        raise SystemExit(f"High-confidence frames not found: {args.high_conf_frames}")

    df = pd.read_csv(args.source)
    if df.empty:
        print(f"[Pseudo] source empty, nothing appended: {args.source}")
        return

    high_conf_frames = {
        line.strip() for line in args.high_conf_frames.read_text().splitlines() if line.strip()
    }
    if not high_conf_frames:
        print(f"[Pseudo] no frames listed in {args.high_conf_frames}, nothing appended")
        return

    before_rows = len(df)
    df = df[df["frame_file"].isin(high_conf_frames)].copy()
    if df.empty:
        print(
            f"[Pseudo] no rows matched frames from {args.high_conf_frames}; "
            f"source rows={before_rows}"
        )
        return

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    header = not args.dest.exists()
    df.to_csv(args.dest, mode="a", index=False, header=header)
    print(
        f"[Pseudo] appended {len(df)} rows from {df['frame_file'].nunique()} frames "
        f"(source rows={before_rows}) -> {args.dest}"
    )


if __name__ == "__main__":
    main()
