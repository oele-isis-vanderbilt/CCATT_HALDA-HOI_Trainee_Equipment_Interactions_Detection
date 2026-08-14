# python3 Main_Code/5.701_stats_merged_annotations.py \
#   --input-json "/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase2/Annotations/trainval_hico.json"

#--- stats 

# {
#   "input_json": "/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase2/Annotations/trainval_hico.json",
#   "frames": 7704,
#   "annotations_total": 103198,
#   "hoi_total": 111129,
#   "object_category_counts": {
#     "91": 5251,
#     "92": 6920,
#     "93": 2333,
#     "94": 9666,
#     "95": 5027,
#     "96": 6354,
#     "97": 5392,
#     "98": 6371,
#     "99": 6366,
#     "100": 3439,
#     "101": 6678,
#     "102": 5764
#   },
#   "verb_counts": {
#     "58": 107129,
#     "118": 4000
#   },
#   "hoi_pair_counts": [
#     {
#       "object_category_id": 1,
#       "verb_id": 58,
#       "count": 1
#     },
#     {
#       "object_category_id": 91,
#       "verb_id": 58,
#       "count": 8397
#     },
#     {
#       "object_category_id": 91,
#       "verb_id": 118,
#       "count": 330
#     },
#     {
#       "object_category_id": 92,
#       "verb_id": 58,
#       "count": 9481
#     },
#     {
#       "object_category_id": 92,
#       "verb_id": 118,
#       "count": 384
#     },
#     {
#       "object_category_id": 93,
#       "verb_id": 58,
#       "count": 4955
#     },
#     {
#       "object_category_id": 93,
#       "verb_id": 118,
#       "count": 280
#     },
#     {
#       "object_category_id": 94,
#       "verb_id": 58,
#       "count": 12512
#     },
#     {
#       "object_category_id": 94,
#       "verb_id": 118,
#       "count": 401
#     },
#     {
#       "object_category_id": 95,
#       "verb_id": 58,
#       "count": 8385
#     },
#     {
#       "object_category_id": 95,
#       "verb_id": 118,
#       "count": 271
#     },
#     {
#       "object_category_id": 96,
#       "verb_id": 58,
#       "count": 11116
#     },
#     {
#       "object_category_id": 96,
#       "verb_id": 118,
#       "count": 333
#     },
#     {
#       "object_category_id": 97,
#       "verb_id": 58,
#       "count": 8558
#     },
#     {
#       "object_category_id": 97,
#       "verb_id": 118,
#       "count": 329
#     },
#     {
#       "object_category_id": 98,
#       "verb_id": 58,
#       "count": 9835
#     },
#     {
#       "object_category_id": 98,
#       "verb_id": 118,
#       "count": 378
#     },
#     {
#       "object_category_id": 99,
#       "verb_id": 58,
#       "count": 8471
#     },
#     {
#       "object_category_id": 99,
#       "verb_id": 118,
#       "count": 401
#     },
#     {
#       "object_category_id": 100,
#       "verb_id": 58,
#       "count": 5998
#     },
#     {
#       "object_category_id": 100,
#       "verb_id": 118,
#       "count": 335
#     },
#     {
#       "object_category_id": 101,
#       "verb_id": 58,
#       "count": 10440
#     },
#     {
#       "object_category_id": 101,
#       "verb_id": 118,
#       "count": 215
#     },
#     {
#       "object_category_id": 102,
#       "verb_id": 58,
#       "count": 8980
#     },
#     {
#       "object_category_id": 102,
#       "verb_id": 118,
#       "count": 343
#     }
#   ],
#   "unique_hoi_pairs": 25,
#   "validation": {
#     "bad_hoi_links": 0,
#     "subject_not_person": 0,
#     "object_is_person": 1
#   }


#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def json_stats(input_json: Path) -> dict:
    with input_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of frame entries in: {input_json}")

    frames = len(data)
    annotations_total = 0
    hoi_total = 0
    object_category_counts = Counter()
    verb_counts = Counter()
    hoi_pair_counts = Counter()

    bad_hoi_links = 0
    subject_not_person = 0
    object_is_person = 0

    for entry in data:
        anns = entry.get("annotations") or []
        hois = entry.get("hoi_annotation") or []
        annotations_total += len(anns)
        hoi_total += len(hois)

        for ann in anns:
            cid = ann.get("category_id")
            if cid != 1:
                object_category_counts[int(cid)] += 1

        for hoi in hois:
            s = hoi.get("subject_id")
            o = hoi.get("object_id")
            v = hoi.get("category_id")
            verb_counts[int(v)] += 1

            if not isinstance(s, int) or not isinstance(o, int) or s < 0 or o < 0 or s >= len(anns) or o >= len(anns):
                bad_hoi_links += 1
                continue

            s_cat = anns[s].get("category_id")
            o_cat = anns[o].get("category_id")
            if s_cat != 1:
                subject_not_person += 1
            if o_cat == 1:
                object_is_person += 1
            hoi_pair_counts[(int(o_cat), int(v))] += 1

    return {
        "input_json": str(input_json),
        "frames": frames,
        "annotations_total": annotations_total,
        "hoi_total": hoi_total,
        "object_category_counts": {str(k): int(v) for k, v in sorted(object_category_counts.items())},
        "verb_counts": {str(k): int(v) for k, v in sorted(verb_counts.items())},
        "hoi_pair_counts": [
            {"object_category_id": int(obj), "verb_id": int(verb), "count": int(cnt)}
            for (obj, verb), cnt in sorted(hoi_pair_counts.items())
        ],
        "unique_hoi_pairs": int(len(hoi_pair_counts)),
        "validation": {
            "bad_hoi_links": bad_hoi_links,
            "subject_not_person": subject_not_person,
            "object_is_person": object_is_person,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Stats for final HICO-format merged JSON.")
    ap.add_argument("--input-json", required=True, type=Path, help="Path to final HICO JSON (e.g., trainval_hico.json).")
    ap.add_argument("--output-json", type=Path, help="Optional path to save stats JSON.")
    args = ap.parse_args()

    stats = json_stats(args.input_json)
    print(json.dumps(stats, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, indent=2))
        print(f"\nSaved stats -> {args.output_json}")


if __name__ == "__main__":
    main()


