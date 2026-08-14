#!/usr/bin/env python3
"""
Check that every file_name in a HICO trainval JSON exists under an image root.

Example:
python3 Main_Code/5.705_check_hico_image_paths.py \
  --annotations-json "/home/mereddd/.../Annotations/annotations/trainval_hico.json" \
  --images-root "/home/mereddd/.../Annotations/images/train2015" \
  --missing-out "/home/mereddd/.../Annotations/annotations/missing_images.txt"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate HICO file_name paths against image root.")
    ap.add_argument("--annotations-json", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--missing-out", type=Path, default=None, help="Optional file to write missing file_name entries.")
    ap.add_argument("--print-limit", type=int, default=30, help="How many missing entries to print.")
    args = ap.parse_args()

    with args.annotations_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected list in {args.annotations_json}")

    missing: list[str] = []
    empty_name = 0

    for e in data:
        if not isinstance(e, dict):
            continue
        fn = str(e.get("file_name", "")).strip()
        if not fn:
            empty_name += 1
            continue
        if not (args.images_root / fn).exists():
            missing.append(fn)

    total = len(data)
    present = total - len(missing) - empty_name

    print(f"annotations_json: {args.annotations_json}")
    print(f"images_root: {args.images_root}")
    print(f"total_entries: {total}")
    print(f"present_entries: {present}")
    print(f"missing_entries: {len(missing)}")
    print(f"empty_file_name_entries: {empty_name}")

    limit = max(0, args.print_limit)
    if missing and limit > 0:
        print("sample_missing:")
        for x in missing[:limit]:
            print(x)

    if args.missing_out:
        args.missing_out.parent.mkdir(parents=True, exist_ok=True)
        args.missing_out.write_text("\n".join(missing) + ("\n" if missing else ""))
        print(f"missing_list_saved: {args.missing_out}")


if __name__ == "__main__":
    main()
