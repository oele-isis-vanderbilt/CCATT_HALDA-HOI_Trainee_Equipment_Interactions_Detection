#!/usr/bin/env python3
"""
Apply object-box NMS per image on HICO annotations and remap HOIs.

- Keeps all subject(person) annotations unchanged.
- Runs NMS on non-person object annotations (optionally per-category).
- Redirects HOI object_id from suppressed boxes -> kept representative box.
- Deduplicates HOIs after remap by (subject_id, object_id, category_id).

Example:
python3 Main_Code/5.704_apply_object_nms_hico.py \
  --input-json "/home/mereddd/.../trainval_hico_filled58.json" \
  --output-json "/home/mereddd/.../trainval_hico_filled58_objnms.json" \
  --iou-thresh 0.5 \
  --per-category
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ann_score(ann: dict, hoi_ref_count: int) -> Tuple[float, float, float]:
    # Prefer explicit score if present; else by how many HOIs reference this object; else area.
    score = float(ann.get("score", ann.get("object_score", 0.0)) or 0.0)
    bbox = ann.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = map(float, bbox[:4])
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return (score, float(hoi_ref_count), area)


def run_nms_indices(ann_idxs: List[int], anns: List[dict], hoi_obj_ref: Dict[int, int], iou_thresh: float) -> Tuple[List[int], Dict[int, int]]:
    # Returns kept indices and suppressed->kept mapping.
    ordered = sorted(
        ann_idxs,
        key=lambda idx: ann_score(anns[idx], hoi_obj_ref.get(idx, 0)),
        reverse=True,
    )
    kept: List[int] = []
    suppressed_to_kept: Dict[int, int] = {}

    for idx in ordered:
        box_i = anns[idx].get("bbox")
        if not box_i or len(box_i) < 4:
            kept.append(idx)
            continue
        matched_keep = None
        for k in kept:
            box_k = anns[k].get("bbox")
            if not box_k or len(box_k) < 4:
                continue
            if iou(box_i, box_k) >= iou_thresh:
                matched_keep = k
                break
        if matched_keep is None:
            kept.append(idx)
        else:
            suppressed_to_kept[idx] = matched_keep
    return kept, suppressed_to_kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply object NMS and remap HICO HOIs.")
    ap.add_argument("--input-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--per-category", action="store_true", help="Run NMS separately per object category_id.")
    args = ap.parse_args()

    with args.input_json.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected list in {args.input_json}")

    images_changed = 0
    total_obj_before = 0
    total_obj_after = 0
    total_obj_suppressed = 0
    hoi_dedup_removed = 0

    for entry in data:
        anns = entry.get("annotations") or []
        hois = entry.get("hoi_annotation") or []

        person_idxs = [i for i, a in enumerate(anns) if a.get("category_id") == 1]
        obj_idxs = [i for i, a in enumerate(anns) if a.get("category_id") != 1]
        total_obj_before += len(obj_idxs)

        # HOI object reference counts used as weak ranking signal.
        hoi_obj_ref: Dict[int, int] = {}
        for h in hois:
            o = h.get("object_id")
            if isinstance(o, int):
                hoi_obj_ref[o] = hoi_obj_ref.get(o, 0) + 1

        kept_obj: List[int] = []
        suppressed_to_kept: Dict[int, int] = {}

        if args.per_category:
            by_cat: Dict[int, List[int]] = {}
            for idx in obj_idxs:
                c = int(anns[idx].get("category_id", -1))
                by_cat.setdefault(c, []).append(idx)
            for idxs in by_cat.values():
                k, m = run_nms_indices(idxs, anns, hoi_obj_ref, args.iou_thresh)
                kept_obj.extend(k)
                suppressed_to_kept.update(m)
        else:
            kept_obj, suppressed_to_kept = run_nms_indices(obj_idxs, anns, hoi_obj_ref, args.iou_thresh)

        kept_obj_set = set(kept_obj)
        total_obj_after += len(kept_obj)
        total_obj_suppressed += (len(obj_idxs) - len(kept_obj))

        # Rebuild annotations list: persons first in original order, then kept objects in original order.
        kept_ann_old_idxs = sorted(person_idxs + kept_obj)
        old_to_new = {old: new for new, old in enumerate(kept_ann_old_idxs)}

        # For suppressed object idx, map to corresponding kept object's new idx.
        suppressed_old_to_new_obj = {}
        for old_sup, old_keep in suppressed_to_kept.items():
            if old_keep in old_to_new:
                suppressed_old_to_new_obj[old_sup] = old_to_new[old_keep]

        new_anns = [anns[i] for i in kept_ann_old_idxs]

        new_hois = []
        seen = set()
        for h in hois:
            s = h.get("subject_id")
            o = h.get("object_id")
            v = h.get("category_id")
            if not isinstance(s, int) or not isinstance(o, int) or not isinstance(v, int):
                continue

            # remap subject (should be kept as-is if person)
            if s not in old_to_new:
                continue
            s_new = old_to_new[s]

            # remap object: kept or suppressed->kept
            if o in old_to_new:
                o_new = old_to_new[o]
            elif o in suppressed_old_to_new_obj:
                o_new = suppressed_old_to_new_obj[o]
            else:
                continue

            key = (s_new, o_new, v)
            if key in seen:
                hoi_dedup_removed += 1
                continue
            seen.add(key)
            new_hois.append({"subject_id": s_new, "object_id": o_new, "category_id": v})

        changed = (len(new_anns) != len(anns)) or (len(new_hois) != len(hois))
        if changed:
            images_changed += 1
            entry["annotations"] = new_anns
            entry["hoi_annotation"] = new_hois

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"input_images: {len(data)}")
    print(f"images_changed: {images_changed}")
    print(f"object_boxes_before: {total_obj_before}")
    print(f"object_boxes_after: {total_obj_after}")
    print(f"object_boxes_suppressed: {total_obj_suppressed}")
    print(f"hoi_dedup_removed_after_remap: {hoi_dedup_removed}")
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
