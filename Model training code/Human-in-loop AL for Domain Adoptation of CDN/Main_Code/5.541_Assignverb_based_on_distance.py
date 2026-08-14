#!/usr/bin/env python3
"""
Pre-label pairs as no_interaction (verb 58) using spatial rules:
- IoU <= threshold, or
- center distance >= threshold

python Main_Code/5.541_Assignverb_based_on_distance.py \
  --csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_nms.csv" \
  --iou-thresh 0.0 \
  --dist-thresh 150

================================================================================

Outputs stats:
Result:

# Backup created:
/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2/low_conf_review_bundle/sampled_uncertain_nms_backup.csv
# Updated file:
/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2/low_conf_review_bundle/sampled_uncertain_nms.csv
# Labeling summary:


"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def parse_box(raw) -> Optional[list[float]]:
  """Parse bbox stored as stringified list into [x1, y1, x2, y2]."""
  if raw is None or (isinstance(raw, float) and pd.isna(raw)):
    return None
  if isinstance(raw, (list, tuple)):
    vals: Sequence[float] = raw
  else:
    try:
      vals = ast.literal_eval(str(raw))
    except Exception:
      return None
  if not isinstance(vals, (list, tuple)) or len(vals) < 4:
    return None
  try:
    return [float(v) for v in vals[:4]]
  except Exception:
    return None


def boxes_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
  """Return True if two xyxy boxes overlap with positive area."""
  return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
  ix1 = max(a[0], b[0])
  iy1 = max(a[1], b[1])
  ix2 = min(a[2], b[2])
  iy2 = min(a[3], b[3])
  iw = max(0.0, ix2 - ix1)
  ih = max(0.0, iy2 - iy1)
  inter = iw * ih
  if inter <= 0:
    return 0.0
  area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  den = area_a + area_b - inter
  return inter / den if den > 0 else 0.0


def center_dist(a: Sequence[float], b: Sequence[float]) -> float:
  ax = (a[0] + a[2]) / 2.0
  ay = (a[1] + a[3]) / 2.0
  bx = (b[0] + b[2]) / 2.0
  by = (b[1] + b[3]) / 2.0
  dx = ax - bx
  dy = ay - by
  return (dx * dx + dy * dy) ** 0.5


def maybe_append_source(existing, new_tag: str) -> str:
  if not existing or pd.isna(existing):
    return new_tag
  existing_str = str(existing)
  if new_tag in existing_str:
    return existing_str
  return f"{existing_str};{new_tag}"


def main() -> None:
  ap = argparse.ArgumentParser(description="Assign no_interaction (58) using IoU and/or distance thresholds.")
  ap.add_argument("--csv", type=Path, required=True, help="manual_review_nms.csv to update.")
  ap.add_argument("--iou-thresh", type=float, default=0.0, help="Set to 0 for no-overlap behavior; rows with IoU <= this become no_interaction.")
  ap.add_argument("--dist-thresh", type=float, help="If set, rows with center distance >= this become no_interaction.")
  ap.add_argument(
      "--auto-nms",
      type=Path,
      dest="auto_nms",
      help="Optional auto_labels_nms.csv to append auto_no_overlap rows into (defaults to sibling auto_labels_nms.csv if present).",
  )
  ap.add_argument("--output", type=Path, help="Optional output path; defaults to in-place update.")
  args = ap.parse_args()

  csv_path: Path = args.csv
  if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

  df = pd.read_csv(csv_path)
  original_df = df.copy()

  if "label" not in df.columns:
    df["label"] = pd.NA
  if "label_source" not in df.columns:
    df["label_source"] = ""

  updated = 0
  verb_changed = 0
  missing_boxes = 0
  moved_to_auto = 0

  for idx, row in df.iterrows():
    pb = parse_box(row.get("person_bbox"))
    rb = parse_box(row.get("roi_bbox"))
    if pb is None or rb is None:
      missing_boxes += 1
      continue
    iou = box_iou(pb, rb)
    dist = center_dist(pb, rb)

    low_iou = iou <= float(args.iou_thresh)
    far_apart = args.dist_thresh is not None and dist >= float(args.dist_thresh)
    if not (low_iou or far_apart):
      continue

    prev_verb = row.get("verb_id")
    df.at[idx, "label"] = 0
    df.at[idx, "verb_id"] = 58
    df.at[idx, "label_source"] = maybe_append_source(row.get("label_source"), "auto_no_overlap")
    updated += 1
    try:
      if float(prev_verb) != 58:
        verb_changed += 1
    except Exception:
      verb_changed += 1

  out_path = args.output or csv_path
  if out_path == csv_path:
    backup = csv_path.with_name(f"{csv_path.stem}_backup{csv_path.suffix}")
    original_df.to_csv(backup, index=False)
    print(f"[Backup] wrote original -> {backup}")

  # Move auto_no_overlap rows from manual_review_nms to auto_labels_nms
  auto_nms_path = args.auto_nms
  if auto_nms_path is None:
    default_auto = csv_path.with_name("auto_labels_nms.csv")
    if default_auto.exists():
      auto_nms_path = default_auto

  move_mask = df["label_source"].astype(str).str.contains("auto_no_overlap", na=False)
  to_move = df.loc[move_mask].copy()
  if auto_nms_path is not None and not to_move.empty:
    # Normalize values for auto file
    to_move["manual_flag"] = False
    to_move["final_label"] = 0
    to_move["label"] = 0
    to_move["label_source"] = "auto_no_overlap"

    if auto_nms_path.exists():
      auto_df = pd.read_csv(auto_nms_path)
    else:
      auto_df = pd.DataFrame()

    existing_ids = set(auto_df["candidate_id"].astype(str)) if "candidate_id" in auto_df else set()
    if "candidate_id" in to_move:
      to_move = to_move[~to_move["candidate_id"].astype(str).isin(existing_ids)]

    # Align columns
    all_cols = list(dict.fromkeys(list(auto_df.columns) + list(to_move.columns)))
    auto_df = auto_df.reindex(columns=all_cols)
    to_move = to_move.reindex(columns=all_cols)

    auto_df = pd.concat([auto_df, to_move], ignore_index=True)
    auto_df.to_csv(auto_nms_path, index=False)
    moved_to_auto = len(to_move)

    # Drop moved rows from manual df
    df = df.loc[~move_mask]
  elif to_move.empty:
    moved_to_auto = 0
  else:
    print(f"[WARN] auto_no_overlap rows detected but no auto_labels_nms target found; keeping in {out_path}")

  df.to_csv(out_path, index=False)

  total_verb58 = int((df.get("verb_id") == 58).sum())

  print(
    f"[Done] {updated} rows set to no_interaction (verb 58); "
    f"{missing_boxes} rows skipped due to missing boxes. "
    f"verb_id changed to 58 for {verb_changed} rows. "
    f"Moved {moved_to_auto} rows to {auto_nms_path if moved_to_auto else 'manual review list'}. "
    f"Total verb_id==58 in file: {total_verb58}. "
    f"Saved -> {out_path}"
  )


if __name__ == "__main__":
  main()
