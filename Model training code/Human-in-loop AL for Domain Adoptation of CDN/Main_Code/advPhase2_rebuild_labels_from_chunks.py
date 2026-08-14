#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

VERB_ID = 117
NO_INTERACTION = 57


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    ab = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def parse_box(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def main():
    ap = argparse.ArgumentParser(description="Rebuild label CSVs from existing chunk_*_cdn_preds_std.csv files.")
    ap.add_argument("--state_dir", type=Path, required=True)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--subject_score_thresh", type=float, default=0.01)
    ap.add_argument("--pos_verb_thresh", type=float, default=0.8)
    ap.add_argument("--neg_verb_thresh", type=float, default=0.2)
    ap.add_argument("--dist_thresh", type=float, default=150.0)
    ap.add_argument("--out_labels", type=Path, required=True)
    ap.add_argument("--out_high_labels", type=Path, required=True)
    ap.add_argument("--out_low_frames", type=Path, required=True)
    ap.add_argument("--out_high_frames", type=Path, required=True)
    args = ap.parse_args()

    chunk_dir = args.state_dir / "chunk_outputs"
    std_files = sorted(chunk_dir.glob("chunk_*_cdn_preds_std.csv"))
    if not std_files:
        raise SystemExit(f"No std chunk CSVs found in {chunk_dir}")

    all_rows = []
    for f in std_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        for col in ("score", "object_score", "verb_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[
            (df.get("verb_id") == VERB_ID)
            & (df.get("object_score") >= args.score_thresh)
            & (df.get("score") > args.subject_score_thresh)
        ]
        if df.empty:
            continue

        records = []
        for row in df.itertuples(index=False):
            subj = parse_box(row.subject_box)
            obj = parse_box(row.object_box)
            verb_score = float(row.verb_score)
            if verb_score >= args.pos_verb_thresh:
                decision = "positive_117"
                verb_out = VERB_ID
            elif verb_score <= args.neg_verb_thresh:
                decision = "negative_low_score"
                verb_out = NO_INTERACTION
            else:
                d = math.hypot(*(a - b for a, b in zip(center(subj), center(obj))))
                if iou(subj, obj) == 0 and d > args.dist_thresh:
                    decision = "negative_far"
                    verb_out = NO_INTERACTION
                else:
                    decision = "uncertain_mid_score"
                    verb_out = VERB_ID

            records.append(
                {
                    "frame_file": row.frame_file,
                    "frame_stem": row.frame_stem,
                    "subject_box": json.dumps(subj),
                    "object_box": json.dumps(obj),
                    "verb_id": int(row.verb_id),
                    "object_id": int(row.object_id),
                    "verb_score": verb_score,
                    "object_score": float(row.object_score),
                    "subject_score": float(row.score),
                    "verb_out": verb_out,
                    "decision": decision,
                }
            )
        if records:
            chunk_labels = pd.DataFrame.from_records(records)
            all_rows.append(chunk_labels)
            # Keep chunk labels in sync too.
            chunk_id = f.name.replace("_cdn_preds_std.csv", "")
            chunk_labels.to_csv(chunk_dir / f"{chunk_id}_cdn_labels.csv", index=False)
            chunk_labels[chunk_labels["decision"] != "uncertain_mid_score"].to_csv(
                chunk_dir / f"{chunk_id}_high_conf_labels.csv", index=False
            )

    if not all_rows:
        raise SystemExit("No rows after filtering from existing chunk std CSVs.")

    labels = pd.concat(all_rows, ignore_index=True)
    high_labels = labels[labels["decision"] != "uncertain_mid_score"].copy()

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out_labels, index=False)
    high_labels.to_csv(args.out_high_labels, index=False)

    frame_scores = labels.groupby("frame_file")["verb_score"].apply(list)
    low_frames = []
    high_frames = []
    for frame, scores in frame_scores.items():
        if any(args.neg_verb_thresh < float(s) < args.pos_verb_thresh for s in scores):
            low_frames.append(frame)
        else:
            high_frames.append(frame)

    args.out_low_frames.write_text("\n".join(sorted(low_frames)))
    args.out_high_frames.write_text("\n".join(sorted(high_frames)))

    print(f"[REBUILD] std chunks: {len(std_files)}")
    print(f"[REBUILD] labels rows: {len(labels)} -> {args.out_labels}")
    print(f"[REBUILD] high rows: {len(high_labels)} -> {args.out_high_labels}")
    print(f"[REBUILD] low frames: {len(low_frames)} -> {args.out_low_frames}")
    print(f"[REBUILD] high frames: {len(high_frames)} -> {args.out_high_frames}")


if __name__ == "__main__":
    main()
