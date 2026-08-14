#!/usr/bin/env python3
"""
Count HOI interactions per object and verb id from a HICO-style annotations JSON.

Outputs totals per object category and for specific verbs (default: interaction 118, no_interaction 58).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def main() -> None:
    ap = argparse.ArgumentParser(description="Count HOI per object/verb from HICO-style annotations.")
    ap.add_argument("--annotations", type=Path, required=True, help="Path to trainval_hico.json (HICO style).")
    ap.add_argument("--interaction_verb", type=int, default=118, help="Verb id for interaction (default 118).")
    ap.add_argument("--no_interaction_verb", type=int, default=58, help="Verb id for no-interaction (default 58).")
    args = ap.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Annotations not found: {args.annotations}")

    with args.annotations.open("r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise SystemExit(f"Expected list in {args.annotations}")

    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in entries:
        anns: List[dict] = entry.get("annotations", [])
        hois: List[dict] = entry.get("hoi_annotation", [])
        for hoi in hois:
            obj_id = hoi.get("object_id")
            verb_id = hoi.get("category_id")
            if obj_id is None or obj_id >= len(anns):
                continue
            obj_cat = anns[obj_id].get("category_id")
            if obj_cat is None:
                continue
            counts[obj_cat]["total"] += 1
            if verb_id == args.interaction_verb:
                counts[obj_cat]["interaction"] += 1
            if verb_id == args.no_interaction_verb:
                counts[obj_cat]["no_interaction"] += 1

    print("HOI counts by object category:")
    for obj_cat, c in sorted(counts.items()):
        print(
            f"  object {obj_cat}: total={c.get('total',0)} "
            f"interaction={c.get('interaction',0)} "
            f"no_interaction={c.get('no_interaction',0)}"
        )


if __name__ == "__main__":
    main()
