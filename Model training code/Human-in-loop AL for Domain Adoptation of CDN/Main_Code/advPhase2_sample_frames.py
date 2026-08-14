#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def list_jpgs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".jpg"
    )


def parse_excluded_relpaths(hico_json: Path) -> set[str]:
    """
    Read file_name entries from prior HICO JSON and convert to relative paths
    matching '<subdir>/<frame>.jpg' when possible.
    Supports:
    - '<subdir>/<frame>.jpg'
    - '<subdir>+<frame>.jpg' (prefixed export form)
    """
    with hico_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected list JSON in {hico_json}")

    excluded: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        fname = item.get("file_name")
        if not fname:
            continue
        s = str(fname).strip().replace("\\", "/")
        if "/" in s:
            excluded.add(s.lstrip("./"))
        elif "+" in s:
            left, right = s.split("+", 1)
            if left and right:
                excluded.add(f"{left}/{right}")
    return excluded


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample JPG frames from input root and copy to output root preserving relative paths."
    )
    ap.add_argument("--input_root", type=Path, required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--sample_size", type=int, default=2400, help="Default: 800*3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean_output", action="store_true", help="Remove output_root before copying.")
    ap.add_argument(
        "--exclude_hico_json",
        type=Path,
        help="Optional prior trainval_hico.json; frames listed there will be excluded from sampling.",
    )
    args = ap.parse_args()

    in_root = args.input_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()

    if not in_root.exists():
        raise SystemExit(f"Input root does not exist: {in_root}")

    all_imgs = list_jpgs(in_root)
    if not all_imgs:
        raise SystemExit(f"No .jpg files found under: {in_root}")

    excluded_count = 0
    if args.exclude_hico_json:
        hico_json = args.exclude_hico_json.expanduser().resolve()
        if hico_json.exists():
            excluded_rel = parse_excluded_relpaths(hico_json)
            if excluded_rel:
                kept = []
                for p in all_imgs:
                    rel = str(p.relative_to(in_root)).replace("\\", "/")
                    if rel in excluded_rel:
                        excluded_count += 1
                    else:
                        kept.append(p)
                all_imgs = kept
        else:
            print(f"[WARN] exclude_hico_json not found, ignoring: {hico_json}")

    if not all_imgs:
        raise SystemExit("No candidate images left after exclusion filter.")

    k = min(max(args.sample_size, 0), len(all_imgs))
    rng = random.Random(args.seed)
    sampled = rng.sample(all_imgs, k) if k < len(all_imgs) else all_imgs

    if args.clean_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sampled:
        rel = src.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Input images : {len(all_imgs)}")
    if args.exclude_hico_json:
        print(f"Excluded     : {excluded_count}")
    print(f"Sample size  : {k}")
    print(f"Output root  : {out_root}")
    print(f"Copied       : {copied}")


if __name__ == "__main__":
    main()
