#!/usr/bin/env python3
"""
Normalize HOIs per human-object pair in HICO JSON.

Rules per (subject_id, object_id):
1) If the pair has any valid interaction (category_id == 118), keep exactly one 118.
2) Otherwise keep exactly one no_interaction (category_id == 58 by default).
3) If a pair is missing entirely, add one no_interaction.

Example run:
python3 Main_Code/5.703_fill_missing_no_interaction_pairs.py \
  --input-json "/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase3/Annotations/trainval_hico.json" \
  --output-json "/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase3/Annotations/trainval_hico_filled58.json" \
  --no-interaction-verb 58
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize HOIs per pair (prefer 118; else 58; fill missing).")
    ap.add_argument("--input-json", type=Path, required=True, help="Input HICO JSON")
    ap.add_argument("--output-json", type=Path, required=True, help="Output HICO JSON")
    ap.add_argument("--no-interaction-verb", type=int, default=58, help="Verb id for no_interaction")
    args = ap.parse_args()

    with args.input_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected list in {args.input_json}")

    added_total = 0
    removed_total = 0
    rewritten_pairs_total = 0
    images_changed = 0
    total_hoi = 0
    total_objects = 0
    total_subjects = 0

    for entry in data:
        anns = entry.get("annotations") or []
        hois = entry.get("hoi_annotation") or []

        human_idxs = [i for i, a in enumerate(anns) if a.get("category_id") == 1]
        object_idxs = [i for i, a in enumerate(anns) if a.get("category_id") != 1]
        total_objects += len(object_idxs)
        total_subjects += len(human_idxs)

        pair_to_hois: dict[tuple[int, int], list[dict]] = {}
        passthrough_hois: list[dict] = []
        for h in hois:
            s = h.get("subject_id")
            o = h.get("object_id")
            if isinstance(s, int) and isinstance(o, int):
                pair_to_hois.setdefault((s, o), []).append(h)
            else:
                passthrough_hois.append(h)

        added_here = 0
        removed_here = 0
        rewritten_here = 0
        new_hois: list[dict] = list(passthrough_hois)

        for s in human_idxs:
            for o in object_idxs:
                pair = (s, o)
                existing = pair_to_hois.get(pair, [])
                if not existing:
                    new_hois.append(
                        {
                            "subject_id": s,
                            "object_id": o,
                            "category_id": args.no_interaction_verb,
                        }
                    )
                    added_here += 1
                    continue

                has_118 = any(int(h.get("category_id", -1)) == 118 for h in existing)
                target = 118 if has_118 else args.no_interaction_verb
                new_hois.append(
                    {
                        "subject_id": s,
                        "object_id": o,
                        "category_id": target,
                    }
                )

                removed_here += max(0, len(existing) - 1)
                if len(existing) != 1 or int(existing[0].get("category_id", -1)) != target:
                    rewritten_here += 1

        if added_here > 0 or removed_here > 0 or rewritten_here > 0:
            entry["hoi_annotation"] = new_hois
            images_changed += 1
            added_total += added_here
            removed_total += removed_here
            rewritten_pairs_total += rewritten_here
            total_hoi += len(new_hois)
        else:
            total_hoi += len(hois)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"input_images: {len(data)}")
    print(f"images_changed: {images_changed}")
    print(f"added_no_interaction_hoi: {added_total}")
    print(f"removed_duplicate_or_conflicting_hoi: {removed_total}")
    print(f"rewritten_pairs: {rewritten_pairs_total}")
    n = len(data)
    print(f"avg_hoi_per_image: {round(total_hoi / n, 4) if n else 0}")
    print(f"avg_objects_per_image: {round(total_objects / n, 4) if n else 0}")
    print(f"avg_subjects_per_image: {round(total_subjects / n, 4) if n else 0}")
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
