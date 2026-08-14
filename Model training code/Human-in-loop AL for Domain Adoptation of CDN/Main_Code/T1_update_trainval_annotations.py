#!/usr/bin/env python3
"""
Update trainval_hico.json by dropping entries listed in a mask file.

Intended to remove IV frames marked for masking after balancing counts across classes.

Example:
  python T1_update_trainval_annotations.py \
    --input Semiautomaticdata/video_smaples/train/phase1/merged_hico_annotations.json \
    --output Semiautomaticdata/Phase1_Training/Annotations/trainval_hico.json \
    --drop_csv Semiautomaticdata/Phase1_Training/propac_samples/iv_mask.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Set


def load_drop_list(csv_paths: Iterable[Path], column: str = "file_name") -> Set[str]:
    drops: Set[str] = set()
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if column not in reader.fieldnames:
                continue
            for row in reader:
                name = row.get(column, "")
                if name:
                    drops.add(name)
    return drops


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop masked entries from trainval_hico.json.")
    ap.add_argument("--input", type=Path, required=True, help="Source merged HICO annotations JSON.")
    ap.add_argument("--output", type=Path, required=True, help="Output path for updated trainval_hico.json.")
    ap.add_argument("--drop_csv", type=Path, action="append", default=[], help="CSV(s) with file_name column to drop.")
    ap.add_argument("--drop_column", default="file_name", help="Column name in CSV to read file names from (default: file_name).")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input annotations not found: {args.input}")

    drop_names = load_drop_list(args.drop_csv, column=args.drop_column)

    with args.input.open("r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise SystemExit(f"Expected list in {args.input}")

    if drop_names:
        filtered = [e for e in entries if e.get("file_name") not in drop_names]
    else:
        filtered = entries

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Wrote {len(filtered)} entries to {args.output} (dropped {len(entries) - len(filtered)} using {len(drop_names)} mask names)")


if __name__ == "__main__":
    main()
