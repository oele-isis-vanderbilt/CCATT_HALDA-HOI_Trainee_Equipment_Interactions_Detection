"""
Export HICO-format JSON from merged Phase-2 annotations.

Recommended (direct merged CSV from 5.600):
python Main_Code/5.601export_hico_annotations.py \
  --input-csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/labels_combined_three_sources.csv" \
  --output-json "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/merged_hico_annotations_merged.json" \
  --change-log-json "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/merged_hico_annotations_change_log.json"

"""
# Stats -phase2
# Input rows: 13752
# Output JSON frames: 1068

#Stats-phase3
# Input rows: 9740
# Output JSON frames: 1338

#Stats-phase4
#input rows=29572 
#Output Json frames= 1812

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

OUTPUT_JSON = "merged_hico_annotations_merged.json"
CHANGE_LOG_JSON = "merged_hico_annotations_change_log.json"
CUSTOM_OBJ_ID_MAP = {
    80: 91,
    81: 92,
    82: 93,
    83: 94,
    84: 95,
    85: 96,
    86: 97,
    87: 98,
    88: 99,
    89: 100,
    90: 101,
    91: 102,
}
ALLOWED_OBJ_IDS = set(CUSTOM_OBJ_ID_MAP.values())
ALLOWED_VERB_IDS = {58, 118}


def prefixed_frame_name(frame_file: str) -> str:
    """
    Keep frame names in standard relative path form:
    '<subdir>/frame_x.jpg'
    """
    p = Path(str(frame_file))
    # Preserve relative folder + filename when available.
    if len(p.parts) >= 2:
        return f"{p.parts[-2]}/{p.parts[-1]}"
    return p.name


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while keeping all values as strings."""
    return pd.read_csv(path, dtype=str)


def parse_bbox(raw) -> List[float] | None:
    """Parse bbox strings like '[x1, y1, x2, y2]' into a float list."""
    try:
        if isinstance(raw, str):
            raw = raw.strip().replace("(", "[").replace(")", "]")
            vals = list(map(float, json.loads(raw)))
        else:
            vals = list(map(float, raw))
        if len(vals) != 4:
            return None
        return [vals[0], vals[1], vals[2], vals[3]]
    except Exception:
        return None


def parse_bbox_union(raw) -> List[float] | None:
    """
    Parse bbox from either:
    - single bbox string/list
    - pipe-joined bbox list (e.g., "[...|[...]")
    Returns union bbox when multiple boxes are present.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str) and "|" in raw:
        boxes: List[List[float]] = []
        for chunk in raw.split("|"):
            b = parse_bbox(chunk)
            if b is not None:
                boxes.append(b)
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]
    return parse_bbox(raw)


def add_person(
    bbox: List[float], people: Dict[Tuple[float, float, float, float], int], annotations: List[dict]
) -> int:
    key = tuple(bbox)
    if key not in people:
        idx = len(annotations)
        people[key] = idx
        annotations.append({"category_id": 1, "bbox": bbox})
    return people[key]


def add_object(
    object_id: int,
    bbox: List[float],
    objects: Dict[Tuple[int, float, float, float, float], int],
    annotations: List[dict],
) -> int:
    key = (int(object_id), *bbox)
    if key not in objects:
        idx = len(annotations)
        objects[key] = idx
        annotations.append({"id": 1, "category_id": int(object_id), "bbox": bbox})
    return objects[key]


def map_object_id(object_id: int) -> int | None:
    """Convert custom index IDs (80-91) to COCO-style IDs (91-102)."""
    if object_id in CUSTOM_OBJ_ID_MAP:
        return CUSTOM_OBJ_ID_MAP[object_id]
    if object_id in ALLOWED_OBJ_IDS:
        return object_id
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HICO JSON from a merged Phase-2 CSV (output of 5.600).")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Direct merged CSV input (e.g., labels_combined_three_sources.csv).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(OUTPUT_JSON),
        help="Path to write the merged HICO-format JSON.",
    )
    parser.add_argument(
        "--change-log-json",
        type=Path,
        default=Path(CHANGE_LOG_JSON),
        help="Path to write remap/dedup change log JSON.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Keep rows without final/verb labels when exporting (default: drop).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_unlabeled = args.include_unlabeled
    input_csv = args.input_csv.expanduser().resolve()
    if not input_csv.exists():
        print(f"[ERROR] input CSV not found: {input_csv}")
        return
    combined_df: pd.DataFrame = load_csv(input_csv).copy()
    if "__camera_root" not in combined_df.columns:
        combined_df["__camera_root"] = "direct_input"
    if "__video_dir" not in combined_df.columns:
        combined_df["__video_dir"] = combined_df.get("video", "video")
    combined_df["__camera_root"] = combined_df["__camera_root"].fillna("direct_input").astype(str)
    combined_df["__video_dir"] = combined_df["__video_dir"].fillna("video").astype(str)
    if "__source_csv" not in combined_df.columns:
        combined_df["__source_csv"] = str(input_csv)
    print(f"[CSV] loaded direct input rows={len(combined_df)} from {input_csv}")

    # Prefer existing verb_id when present; otherwise fall back to final_label/label.
    if "verb_label" not in combined_df.columns:
        if "verb_id" in combined_df.columns:
            combined_df["verb_label"] = combined_df["verb_id"]
        elif "final_label" in combined_df.columns:
            combined_df["verb_label"] = combined_df["final_label"]
        elif "label" in combined_df.columns:
            combined_df["verb_label"] = combined_df["label"]

    if not include_unlabeled and "verb_label" in combined_df.columns:
        combined_df = combined_df[~pd.isna(combined_df["verb_label"])]

    if "frame_file" not in combined_df.columns:
        print("[ERROR] No frame_file column found in combined data; cannot export HICO JSON.")
        return

    combined_df = combined_df[~pd.isna(combined_df["frame_file"])]

    # Build HICO-format JSON
    results = []
    frame_change_logs: List[dict] = []
    merged_box_replacements: List[dict] = []
    skipped_frames = 0
    object_id_remaps = 0
    rows_skipped_invalid_object_id = 0
    invalid_frames = 0

    for (cam_root, video_dir, frame_file), rows in combined_df.groupby(
        ["__camera_root", "__video_dir", "frame_file"]
    ):
        frame_file_original = str(frame_file)
        frame_name = prefixed_frame_name(frame_file_original)
        annotations: List[dict] = []
        hois: List[dict] = []
        people_map: Dict[Tuple[float, float, float, float], int] = {}
        object_map: Dict[Tuple[int, float, float, float, float], int] = {}

        skipped_missing = 0
        merged_rows = 0
        premerge_hoi_refs = 0
        for _, r in rows.iterrows():
            pb = parse_bbox_union(r.get("merged_person_bboxes")) or parse_bbox_union(r.get("person_bbox"))
            rb = parse_bbox_union(r.get("merged_roi_bboxes")) or parse_bbox_union(r.get("roi_bbox"))
            if pb is None or rb is None:
                skipped_missing += 1
                continue

            subj_idx = add_person(pb, people_map, annotations)
            try:
                object_id_raw = int(float(r.get("object_id")))
            except Exception:
                skipped_missing += 1
                continue
            mapped_object_id = map_object_id(object_id_raw)
            if mapped_object_id is None:
                rows_skipped_invalid_object_id += 1
                continue
            if mapped_object_id != object_id_raw:
                object_id_remaps += 1
            obj_idx = add_object(mapped_object_id, rb, object_map, annotations)
            premerge_hoi_refs += 1

            cluster_size_raw = r.get("cluster_size")
            try:
                cluster_size = int(float(cluster_size_raw)) if not pd.isna(cluster_size_raw) else 1
            except Exception:
                cluster_size = 1
            if cluster_size > 1:
                merged_rows += 1
                merged_box_replacements.append(
                    {
                        "frame_file": frame_name,
                        "frame_file_original": frame_file_original,
                        "replacement_person_bbox": pb,
                        "replacement_object_bbox": rb,
                        "merged_candidate_ids": str(r.get("merged_candidate_ids", "")),
                        "original_person_bboxes": str(r.get("merged_person_bboxes", "")),
                        "original_object_bboxes": str(r.get("merged_roi_bboxes", "")),
                        "cluster_size": cluster_size,
                    }
                )

            # Prefer explicit binary manual/auto label mapping:
            # 0/0.0 -> verb 58, 1/1.0 -> verb 118
            lbl_raw = r.get("label")
            if lbl_raw is None or (isinstance(lbl_raw, float) and pd.isna(lbl_raw)):
                lbl_raw = r.get("verb_label")
            try:
                lbl = 0 if pd.isna(lbl_raw) else int(float(lbl_raw))
            except Exception:
                lbl = 0

            if lbl in (58, 118):
                verb_id = lbl
            else:
                verb_id = 118 if lbl == 1 else 58
            hois.append({"subject_id": subj_idx, "object_id": obj_idx, "category_id": verb_id})

        # Deduplicate HOIs after remapping to canonical annotation ids.
        hois_dedup = []
        seen = set()
        hoi_dupes_removed = 0
        for h in hois:
            key = (int(h["subject_id"]), int(h["object_id"]), int(h["category_id"]))
            if key in seen:
                hoi_dupes_removed += 1
                continue
            seen.add(key)
            hois_dedup.append(h)
        hois = hois_dedup

        # Per-frame validation:
        # - subject_id points to person annotation
        # - object_id points to non-person annotation
        # - object category IDs are in updated mapping (91-102)
        # - verb IDs are in allowed set (58/118)
        frame_valid = True
        for h in hois:
            s = int(h["subject_id"])
            o = int(h["object_id"])
            v = int(h["category_id"])
            if s < 0 or s >= len(annotations) or o < 0 or o >= len(annotations):
                frame_valid = False
                break
            if annotations[s].get("category_id") != 1:
                frame_valid = False
                break
            obj_cat = int(annotations[o].get("category_id"))
            if obj_cat == 1 or obj_cat not in ALLOWED_OBJ_IDS:
                frame_valid = False
                break
            if v not in ALLOWED_VERB_IDS:
                frame_valid = False
                break
        if not frame_valid:
            invalid_frames += 1
            skipped_frames += 1
            print(
                f"[INFO] Skipped invalid frame {frame_file} in {video_dir} ({cam_root}) "
                f"(validation failed)"
            )
            continue

        if hois and annotations:
            results.append(
                {
                    "file_name": frame_name,
                    "file_name_original": frame_file_original,
                    "annotations": annotations,
                    "hoi_annotation": hois,
                }
            )
            frame_change_logs.append(
                {
                    "file_name": frame_name,
                    "file_name_original": frame_file_original,
                    "annotations_count": len(annotations),
                    "hoi_before_dedup": premerge_hoi_refs,
                    "hoi_after_dedup": len(hois),
                    "hoi_duplicates_removed": hoi_dupes_removed,
                    "rows_with_merged_boxes": merged_rows,
                    "rows_skipped_missing_boxes": skipped_missing,
                }
            )
        else:
            skipped_frames += 1
            print(
                f"[INFO] Skipped frame {frame_file} in {video_dir} ({cam_root}) "
                f"(hois={len(hois)}, annots={len(annotations)}, missing_boxes={skipped_missing})"
            )

    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[JSON] wrote {len(results)} frames -> {output_json}")

    totals = {
        "frames_exported": len(results),
        "frames_skipped": skipped_frames,
        "frames_invalid": invalid_frames,
        "object_id_rows_remapped_80_91_to_91_102": object_id_remaps,
        "rows_skipped_invalid_object_id": rows_skipped_invalid_object_id,
        "merged_box_rows": sum(x["rows_with_merged_boxes"] for x in frame_change_logs),
        "hoi_before_dedup": sum(x["hoi_before_dedup"] for x in frame_change_logs),
        "hoi_after_dedup": sum(x["hoi_after_dedup"] for x in frame_change_logs),
        "hoi_duplicates_removed": sum(x["hoi_duplicates_removed"] for x in frame_change_logs),
    }
    change_log = {
        "totals": totals,
        "merged_box_replacements": merged_box_replacements,
        "frame_changes": frame_change_logs,
    }
    change_log_json = args.change_log_json.expanduser().resolve()
    change_log_json.parent.mkdir(parents=True, exist_ok=True)
    with change_log_json.open("w") as f:
        json.dump(change_log, f, indent=2)
    print(f"[ChangeLog] wrote -> {change_log_json}")

if __name__ == "__main__":
    main()
