#!/usr/bin/env python3
"""
HOI deduplication with frame-level person clustering.

Key behavior:
- Normalize coordinates (round to 2 decimals).
- Validate boxes (xmin < xmax, ymin < ymax).
- Group strictly by frame id (frame_file fallback frame_path).
- Build person clusters by person IoU.
- Overwrite person_bbox in every row of each cluster with the union person box.
- Optional row suppression with --drop-duplicates:
  suppress only duplicate HOIs within a person cluster where interaction label matches
  and ROI boxes overlap by roi_iou threshold; keep one representative with union ROI box.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def parse_bbox(raw: str) -> BBox:
    vals = ast.literal_eval(raw)
    if (
        not isinstance(vals, (list, tuple))
        or len(vals) != 4
        or not all(isinstance(v, (int, float)) for v in vals)
    ):
        raise ValueError(f"Bad bbox: {raw!r}")
    x1, y1, x2, y2 = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    # normalize ordering
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # validate strict extents
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox extents: {raw!r}")
    # round for stable matching
    return (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))


def format_bbox(box: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in box) + "]"


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def union_bbox(boxes: Sequence[BBox]) -> BBox:
    x1 = round(min(b[0] for b in boxes), 2)
    y1 = round(min(b[1] for b in boxes), 2)
    x2 = round(max(b[2] for b in boxes), 2)
    y2 = round(max(b[3] for b in boxes), 2)
    return (x1, y1, x2, y2)


def frame_id(row: Dict) -> str:
    return str(row.get("frame_file") or row.get("frame_path") or "")


def interaction_id(row: Dict) -> str:
    if row.get("verb_out") not in ("", None):
        return str(row.get("verb_out"))
    if row.get("verb_id") not in ("", None):
        return str(row.get("verb_id"))
    return ""


def load_predictions(path: Path) -> List[Dict]:
    preds: List[Dict] = []
    bad_rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                person_box = parse_bbox(row["person_bbox"])
                roi_box = parse_bbox(row["roi_bbox"])
            except Exception:
                bad_rows += 1
                continue
            preds.append(
                {
                    **row,
                    "person_bbox": person_box,
                    "roi_bbox": roi_box,
                    "person_score": float(row.get("person_score", 0.0) or 0.0),
                    "object_score": float(row.get("object_score", 0.0) or 0.0),
                    "verb_score": float(row.get("verb_score", 0.0) or 0.0),
                    "cdn_score": float(row.get("cdn_score", 0.0) or 0.0),
                }
            )
    if bad_rows:
        print(f"[NMS] skipped invalid rows: {bad_rows}")
    return preds


def build_person_clusters(rows: List[Dict], person_iou: float, score_field: str) -> List[List[Dict]]:
    ordered = sorted(rows, key=lambda r: float(r.get(score_field, 0.0)), reverse=True)
    clusters: List[List[Dict]] = []
    for row in ordered:
        matched = None
        for cluster in clusters:
            anchor = cluster[0]
            if iou(row["person_bbox"], anchor["person_bbox"]) >= person_iou:
                matched = cluster
                break
        if matched is None:
            clusters.append([row])
        else:
            matched.append(row)
    return clusters


def dedup_within_person_cluster(cluster_rows: List[Dict], roi_iou: float, score_field: str) -> List[Dict]:
    """
    Suppress only duplicate HOIs:
    same interaction label + overlapping ROI boxes (>= roi_iou).
    """
    by_interaction: Dict[str, List[Dict]] = defaultdict(list)
    for r in cluster_rows:
        by_interaction[interaction_id(r)].append(r)

    kept: List[Dict] = []
    for _, rows in by_interaction.items():
        ordered = sorted(rows, key=lambda r: float(r.get(score_field, 0.0)), reverse=True)
        roi_clusters: List[List[Dict]] = []
        for row in ordered:
            matched = None
            for c in roi_clusters:
                if iou(row["roi_bbox"], c[0]["roi_bbox"]) >= roi_iou:
                    matched = c
                    break
            if matched is None:
                roi_clusters.append([row])
            else:
                matched.append(row)

        for c in roi_clusters:
            best = max(c, key=lambda r: float(r.get(score_field, 0.0)))
            merged = dict(best)
            merged_person = union_bbox([r["person_bbox"] for r in c])
            merged_roi = union_bbox([r["roi_bbox"] for r in c])
            merged["person_bbox"] = merged_person
            merged["roi_bbox"] = merged_roi
            merged["merged_candidate_ids"] = "|".join(str(r.get("candidate_id", "")) for r in c)
            # Store one final merged box per row (no pipe-joined duplicates).
            merged["merged_person_bboxes"] = format_bbox(merged_person)
            merged["merged_roi_bboxes"] = format_bbox(merged_roi)
            merged["cluster_size"] = len(c)
            kept.append(merged)
    return kept


def write_predictions(path: Path, preds: List[Dict]) -> None:
    base_fieldnames = [
        "candidate_id",
        "video",
        "frame_file",
        "frame_path",
        "equipment_type",
        "person_bbox",
        "roi_bbox",
        "person_score",
        "object_score",
        "verb_score",
        "verb_id",
        "verb_out",
        "object_id",
        "cdn_score",
        "merged_candidate_ids",
        "merged_person_bboxes",
        "merged_roi_bboxes",
        "cluster_size",
    ]
    extra_fieldnames = sorted({k for pred in preds for k in pred.keys()} - set(base_fieldnames))
    fieldnames = base_fieldnames + extra_fieldnames
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in preds:
            row = {k: pred.get(k, "") for k in fieldnames}
            row["person_bbox"] = format_bbox(pred["person_bbox"])
            row["roi_bbox"] = format_bbox(pred["roi_bbox"])
            row["person_score"] = f"{pred.get('person_score', 0.0):.6f}"
            row["object_score"] = f"{pred.get('object_score', 0.0):.6f}"
            row["verb_score"] = f"{pred.get('verb_score', 0.0):.6f}"
            row["cdn_score"] = f"{pred.get('cdn_score', 0.0):.6f}"
            writer.writerow(row)


def run_nms(
    input_path: Path,
    output_path: Path,
    person_iou: float,
    roi_iou: float,
    score_field: str,
    drop_duplicates: bool,
) -> None:
    preds = load_predictions(input_path)
    if not preds:
        raise SystemExit(f"No valid rows in {input_path}")

    by_frame: Dict[str, List[Dict]] = defaultdict(list)
    for p in preds:
        by_frame[frame_id(p)].append(p)

    out_rows: List[Dict] = []
    total_before = len(preds)

    for _, frame_rows in by_frame.items():
        person_clusters = build_person_clusters(frame_rows, person_iou=person_iou, score_field=score_field)
        for cluster in person_clusters:
            person_union = union_bbox([r["person_bbox"] for r in cluster])
            # Explicit overwrite of person box for every row in the cluster.
            normalized_cluster: List[Dict] = []
            for r in cluster:
                row = dict(r)
                row["person_bbox"] = person_union
                normalized_cluster.append(row)

            if drop_duplicates:
                out_rows.extend(dedup_within_person_cluster(normalized_cluster, roi_iou=roi_iou, score_field=score_field))
            else:
                out_rows.extend(normalized_cluster)

    write_predictions(output_path, out_rows)
    print(f"Input rows: {total_before}")
    print(f"Output rows: {len(out_rows)}")
    print(f"Suppressed: {total_before - len(out_rows)}")
    print(f"Saved to: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply frame-level HOI dedup and person union-box merge.")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input CSV path.")
    ap.add_argument("-o", "--output", type=Path, help="Output CSV path.")
    ap.add_argument("--person-iou", type=float, default=0.5, help="Person IoU threshold for person clustering.")
    ap.add_argument("--roi-iou", type=float, default=0.5, help="ROI IoU threshold for duplicate HOI suppression.")
    ap.add_argument(
        "--score-field",
        type=str,
        default="verb_score",
        choices=["verb_score", "person_score", "object_score", "cdn_score"],
        help="Score used to choose representative row within duplicate HOI clusters.",
    )
    ap.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Suppress duplicate HOI rows (same interaction + overlapping ROI) within each person cluster.",
    )
    args = ap.parse_args()

    output_path = args.output or args.input.with_name(args.input.stem + "_nms.csv")
    run_nms(
        input_path=args.input,
        output_path=output_path,
        person_iou=args.person_iou,
        roi_iou=args.roi_iou,
        score_field=args.score_field,
        drop_duplicates=args.drop_duplicates,
    )


if __name__ == "__main__":
    main()
