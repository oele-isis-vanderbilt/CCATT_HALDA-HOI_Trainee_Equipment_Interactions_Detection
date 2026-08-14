#!/usr/bin/env python3
"""
Merge Phase-2 annotation sources into one CSV.

Sources merged (strict, single-file defaults):
1) High-confidence labels: high_conf_labels_nms.csv
2) Low-confidence auto-labeled subset: low_conf_review_bundle/sampled_uncertain_nms.csv
3) Low-confidence manually corrected subset: low_conf_review_bundle/sampled_uncertain_nms_labeled.csv

Behavior:
- All three files are required (unless overridden explicitly with --high-conf/--low-auto/--low-manual).
- Rows are concatenated in this order: high_conf -> low_conf_auto -> low_conf_manual.
- By default, duplicates are removed using keys:
  [candidate_id, frame_file, object_id, verb_id]
  keeping the first occurrence (so manual rows replace overlapping auto rows naturally
  when they have different verb_id; exact duplicates are collapsed).
- Use --skip-dedup to keep raw concatenation.

Typical usage:
python Main_Code/5.600merge_annotations.py \
  --state-dir "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4" \
  --high-conf "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/high_conf_labels_nms.csv" \
  --low-manual "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_nms_labeled.csv" \
  --output "labels_combined_three_sources.csv"
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def align_columns(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    cols: List[str] = []
    seen = set()
    for df in frames:
        for c in df.columns:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return [df.reindex(columns=cols) for df in frames]


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge high-conf + low-conf(auto/manual) Phase-2 labels.")
    ap.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Phase-2 state directory (contains high_conf_labels*.csv and low_conf_review_bundle).",
    )
    ap.add_argument(
        "--high-conf",
        type=Path,
        help="Optional explicit high-confidence CSV path (overrides default resolution).",
    )
    ap.add_argument(
        "--low-auto",
        type=Path,
        help="Optional explicit low-confidence auto-labeled CSV path.",
    )
    ap.add_argument(
        "--low-manual",
        type=Path,
        help="Optional explicit low-confidence manually-labeled CSV path.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="labels_combined_three_sources.csv",
        help="Output file name (written inside --state-dir unless absolute path).",
    )
    ap.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Do not drop duplicates across sources.",
    )
    args = ap.parse_args()

    state_dir = args.state_dir.expanduser().resolve()
    if not state_dir.is_dir():
        raise SystemExit(f"State dir not found: {state_dir}")

    review_dir = state_dir / "low_conf_review_bundle"

    high_conf_path = (
        args.high_conf.expanduser().resolve()
        if args.high_conf
        else (state_dir / "high_conf_labels_nms.csv")
    )
    low_auto_path = (
        args.low_auto.expanduser().resolve()
        if args.low_auto
        else (review_dir / "sampled_uncertain_nms.csv")
    )
    low_manual_path = (
        args.low_manual.expanduser().resolve()
        if args.low_manual
        else (review_dir / "sampled_uncertain_nms_labeled.csv")
    )

    required = [
        ("high_conf", high_conf_path),
        ("low_conf_auto", low_auto_path),
        ("low_conf_manual", low_manual_path),
    ]
    missing = [f"{tag}: {p}" for tag, p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required input file(s):\n- " + "\n- ".join(missing))

    sources: List[tuple[str, Path]] = required

    frames: List[pd.DataFrame] = []
    print("Using sources:")
    for tag, p in sources:
        df = read_csv(p)
        df["merge_source"] = tag
        frames.append(df)
        print(f"- {tag}: {p} (rows={len(df)})")

    frames = align_columns(frames)
    combined = pd.concat(frames, ignore_index=True)

    if not args.skip_dedup:
        dedup_keys = [k for k in ["candidate_id", "frame_file", "object_id", "verb_id"] if k in combined.columns]
        if dedup_keys:
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_keys, keep="first")
            print(f"Dedup keys={dedup_keys} removed={before - len(combined)}")
        else:
            print("Dedup skipped: no standard keys present.")

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = state_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Saved merged CSV -> {output_path}")
    print(f"Total rows -> {len(combined)}")
    if "merge_source" in combined.columns:
        print("Rows by source:")
        print(combined["merge_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
