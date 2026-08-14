#!/usr/bin/env python3
"""
Prepare advPhase2 high-confidence labels for 5.5_remove_hoi_duplicates.py.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def is_valid_box(raw) -> tuple[bool, str]:
    if pd.isna(raw):
        return False, "missing"
    try:
        vals = ast.literal_eval(str(raw))
    except Exception as exc:
        return False, f"parse_error:{exc}"
    ok = (
        isinstance(vals, (list, tuple))
        and len(vals) == 4
        and all(isinstance(v, (int, float)) for v in vals)
    )
    if not ok:
        return False, f"bad_shape:{vals!r}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Phase 2 labels for HOI duplicate removal.")
    ap.add_argument("--input_csv", type=Path, required=True, help="Input high_conf_labels.csv")
    ap.add_argument("--frames_txt", type=Path, help="Optional high_conf_frames.txt; keep only rows from these frames.")
    ap.add_argument("--output_csv", type=Path, required=True, help="Output CSV for 5.5_remove_hoi_duplicates.py")
    args = ap.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise SystemExit(f"Input CSV is empty: {args.input_csv}")

    if "frame_file" not in df.columns or "subject_box" not in df.columns or "object_box" not in df.columns:
        raise SystemExit("Input CSV must contain frame_file, subject_box, and object_box columns.")

    if args.frames_txt:
        if not args.frames_txt.exists():
            raise SystemExit(f"Frames file not found: {args.frames_txt}")
        keep_frames = {
            line.strip() for line in args.frames_txt.read_text().splitlines() if line.strip()
        }
        if not keep_frames:
            raise SystemExit(f"No frames found in {args.frames_txt}")
        before_rows = len(df)
        df = df[df["frame_file"].isin(keep_frames)].copy()
        print(
            f"[NMSPrep] filtered to high_conf_frames rows={len(df)} "
            f"(from {before_rows}) using {args.frames_txt}"
        )
        if df.empty:
            raise SystemExit("No rows remained after filtering by high_conf_frames.")

    df = df.copy()
    subject_checks = df["subject_box"].apply(is_valid_box)
    object_checks = df["object_box"].apply(is_valid_box)
    subj_ok = subject_checks.apply(lambda x: x[0])
    obj_ok = object_checks.apply(lambda x: x[0])
    invalid_mask = ~(subj_ok & obj_ok)
    before_box_filter = len(df)
    if invalid_mask.any():
        bad = df.loc[invalid_mask, ["frame_file", "subject_box", "object_box"]].copy()
        bad["subject_box_error"] = subject_checks[invalid_mask].apply(lambda x: x[1])
        bad["object_box_error"] = object_checks[invalid_mask].apply(lambda x: x[1])
        print("[NMSPrep] skipping rows with invalid boxes:")
        for row in bad.head(20).to_dict(orient="records"):
            print(json.dumps(row, ensure_ascii=True))
        if len(bad) > 20:
            print(f"[NMSPrep] ... and {len(bad) - 20} more invalid rows")

    df = df.loc[~invalid_mask].copy()
    dropped = before_box_filter - len(df)
    if dropped:
        print(f"[NMSPrep] dropped rows with missing/invalid boxes: {dropped}")
    if df.empty:
        raise SystemExit("No rows remained after dropping invalid subject/object boxes.")

    df.insert(0, "candidate_id", [f"cand_{i:09d}" for i in range(len(df))])
    df["video"] = df.get("video", "")
    df["frame_path"] = df.get("frame_path", df["frame_file"])
    df["equipment_type"] = df.get("equipment_type", "phase2")
    df["person_bbox"] = df["subject_box"]
    df["roi_bbox"] = df["object_box"]
    if "subject_score" in df.columns:
        df["person_score"] = df["subject_score"]
    else:
        df["person_score"] = 0.0
    if "verb_out" in df.columns:
        df["verb_id"] = df["verb_out"]
    elif "verb_id" not in df.columns:
        df["verb_id"] = 117
    if "object_id" not in df.columns:
        df["object_id"] = 0
    if "cdn_score" not in df.columns:
        if "verb_score" in df.columns:
            df["cdn_score"] = df["verb_score"]
        else:
            df["cdn_score"] = 0.0

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[NMSPrep] rows={len(df)} -> {args.output_csv}")


if __name__ == "__main__":
    main()
