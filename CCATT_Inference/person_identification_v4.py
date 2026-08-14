#!/usr/bin/env python3
"""
Step 3 of the CCATT inference pipeline (fast, CPU-only): match HOI subject
boxes to a role/identity.

What this script does:
  1. Assumes role assignment for each video has already been produced
     separately (step 2, generate_role_assignment_csvs.py), exported to a
     CSV of (frame_id, time_seconds, track_id, x1, y1, x2, y2, conf, role)
     rows -- one row per tracked person box per processed frame.
  2. For each HOI subject box in --input_root's combined_segments.csv
     files, converts its frame_id into elapsed seconds and matches it to
     the nearest role-assignment CSV row(s) within --match_time_tolerance
     seconds, by IoU (falling back to center distance).
  3. HOI boxes with no role-assignment row within tolerance are left
     unmatched and skipped -- only frames present in *both* the
     role-assignment CSV and the HOI results are ever compared.

This script does not run any detection or tracking model itself -- it only
reads two sets of CSVs and joins them, so it is safe and fast to re-run any
time (e.g. after tweaking --hoi_sampled_fps or --match_time_tolerance), and
it will not touch combined_segments.csv itself.

This script depends on --role_csv_root already being populated -- run
generate_role_assignment_csvs.py (step 2, slow/GPU) first.

Example (see README.md for the full 3-step pipeline):
  python3 person_identification_v4.py \\
    --input_root /path/to/hoi_results/FINAL_DEFAULT_RUN \\
    --role_csv_root /path/to/role_assignment_csvs \\
    --hoi_sampled_fps 4.0

Optional: pass --inplace to overwrite this script's own previously-written
outputs (combined_segments_with_person_id_v4.csv /
CCATT_Trainee_Equipment_Interactions.csv) -- it never overwrites the
original combined_segments.csv.

Use --dry_run to check that --role_csv_root / --input_root (or
--input_csv) and required CSV columns all exist, and see which files would
be processed vs. skipped, without writing any output.
"""

import argparse
import bisect
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


IDENTITY_MAP = {0: "Nurse", 1: "Doctor", 2: "RT", 3: "Additional Team"}
ADDITIONAL_TEAM_CLASS_ID = 3

# role_contrastive/role_assignment.py's CLASS_NAMES use "Physician"/"Additional
# Staff" (verified against the actual CLASS_NAMES list, not just its docstring);
# the rest of the CCAT pipeline (IDENTITY_MAP above, metrics, viz) uses
# "Doctor"/"Additional Team". Map every spelling variant actually used by either
# codebase to the same class id so the output stays consistent regardless of
# which CSV vocabulary shows up.
ROLE_NAME_TO_CLASS_ID = {
    "Nurse": 0,
    "Physician": 1,
    "Doctor": 1,
    "RT": 2,
    "Additional Staff": 3,
    "Additional Team": 3,
}


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _parse_boxes(cell):
    if cell is None:
        return []
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _row_person_boxes(row, preferred_col, fallback_col="person_bounding_boxes"):
    boxes = _parse_boxes(row.get(preferred_col)) if preferred_col in row.index else []
    if boxes:
        return boxes
    if fallback_col and fallback_col != preferred_col and fallback_col in row.index:
        return _parse_boxes(row.get(fallback_col))
    return []


def _sanitize_video_id(video_id: str) -> str:
    s = str(video_id or "").strip()
    s = s.split("|")[0].strip()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    return s


def _sanitize_stem(stem: str) -> str:
    s = str(stem or "").strip()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    return s


def _extract_view_token(s: str):
    u = str(s or "").upper()
    if "CAM16" in u:
        return "CAM16"
    if "PAN" in u:
        return "PAN"
    return None


def _extract_sim_key(s: str):
    m = re.search(r"(\d{4}[A-Za-z])\s+([A-Za-z])", str(s or ""))
    if not m:
        return None
    return f"{m.group(1).upper()}_{m.group(2).upper()}"


def _measure_hoi_fps(row, fallback_fps: float) -> float:
    """The HOI extraction pipeline's actual achieved sampling rate is not a fixed
    constant across every video (it depends on that video's native fps and how the
    extractor rounded/truncated its frame-skip interval -- see 1_Frames_Extraction.py).
    combined_segments.csv already carries duration_frames/duration_seconds for each
    interval, which is exactly that achieved rate -- measure it from there per row
    instead of trusting a single CLI-wide assumption. Only falls back to
    --hoi_sampled_fps when those columns are absent or unusable for this row."""
    duration_frames = row.get("duration_frames") if hasattr(row, "get") else None
    duration_seconds = row.get("duration_seconds") if hasattr(row, "get") else None
    try:
        duration_frames = float(duration_frames)
        duration_seconds = float(duration_seconds)
    except (TypeError, ValueError):
        return fallback_fps
    if pd.isna(duration_frames) or pd.isna(duration_seconds) or duration_seconds <= 0:
        return fallback_fps
    return duration_frames / duration_seconds


def _xywh_to_xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _iou_xyxy(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except Exception:
        return 0.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_dist_xyxy(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except Exception:
        return float("inf")
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def _build_role_csv_index(role_csv_roots):
    """Same stem/sim-view indexing scheme as the temporal-interval script's video
    index, but over role-assignment *.csv files instead of video files."""
    idx = {}
    idx_sim_view = {}
    for root in role_csv_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if not p.is_file():
                continue
            stem = p.stem
            stem_sanitized = _sanitize_stem(stem)
            idx.setdefault(stem, p)
            idx.setdefault(stem_sanitized, p)
            sim_key = _extract_sim_key(stem)
            view = _extract_view_token(stem)
            if sim_key and view:
                idx_sim_view.setdefault((sim_key, view), p)
    return idx, idx_sim_view


def _find_role_csv_path(csv_index, csv_index_sim_view, video_id, source_video_stem=""):
    raw = str(source_video_stem or "").strip()
    sanitized = _sanitize_stem(raw)
    clean_video_id = _sanitize_video_id(video_id)
    for key in [raw, sanitized, str(video_id or "").strip(), clean_video_id]:
        if key and key in csv_index:
            return csv_index[key]
    sim_key = _extract_sim_key(raw or video_id)
    view = _extract_view_token(raw or video_id)
    if sim_key and view:
        return csv_index_sim_view.get((sim_key, view))
    return None


def _load_role_csv(path: Path):
    df = pd.read_csv(path)
    required = {"time_seconds", "track_id", "x1", "y1", "x2", "y2", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Role-assignment CSV {path} is missing columns {sorted(missing)} -- "
            "regenerate it with generate_role_assignment_csvs.py."
        )

    by_time = defaultdict(list)
    for r in df.itertuples(index=False):
        by_time[float(r.time_seconds)].append({
            "track_id": int(r.track_id),
            "xyxy": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)],
            "conf": float(getattr(r, "conf", 0.0) or 0.0),
            "role": str(r.role),
        })
    times_sorted = sorted(by_time.keys())
    return {"times": times_sorted, "by_time": by_time}


def _get_or_load_role_csv(cache, csv_index, csv_index_sim_view, video_id, source_video_stem):
    csv_path = _find_role_csv_path(csv_index, csv_index_sim_view, video_id, source_video_stem)
    if csv_path is None:
        return None
    key = str(csv_path)
    if key in cache:
        return cache[key]
    cache[key] = _load_role_csv(csv_path)
    return cache[key]


def _match_hoi_to_role(hoi_box_xywh, role_data, hoi_time: float, args):
    if hoi_box_xywh is None:
        return None
    try:
        hoi_xyxy = _xywh_to_xyxy(hoi_box_xywh)
    except Exception:
        return None

    times = role_data["times"]
    if not times:
        return None
    lo = bisect.bisect_left(times, hoi_time - args.match_time_tolerance)
    hi = bisect.bisect_right(times, hoi_time + args.match_time_tolerance)
    if lo >= hi:
        return None

    best = None
    best_iou = -1.0
    best_dist = float("inf")
    for t in times[lo:hi]:
        for cand in role_data["by_time"][t]:
            iou = _iou_xyxy(hoi_xyxy, cand["xyxy"])
            dist = _center_dist_xyxy(hoi_xyxy, cand["xyxy"])
            if iou > best_iou or (iou == best_iou and dist < best_dist):
                best_iou = iou
                best_dist = dist
                best = cand

    if best is None:
        return None
    if best_iou >= args.iou_threshold:
        return best, "iou", best_iou, best_dist
    if best_dist <= args.center_dist_fallback:
        return best, "center", best_iou, best_dist
    return None


def _process_csv(args, in_csv, out_csv, role_csv_roots):
    in_csv = Path(in_csv).expanduser().resolve()
    out_csv = Path(out_csv).expanduser().resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    if args.person_boxes_col not in df.columns and "person_bounding_boxes" not in df.columns:
        raise ValueError(
            f"Column '{args.person_boxes_col}' or fallback 'person_bounding_boxes' "
            f"not found in {in_csv}. Is this a combined_segments.csv from step 1 "
            "(V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py)?"
        )
    if args.video_id_col not in df.columns:
        raise ValueError(f"Column '{args.video_id_col}' not found in {in_csv}")

    csv_index, csv_index_sim_view = _build_role_csv_index(role_csv_roots)
    role_csv_cache = {}

    diag = Counter()
    subject_class_ids = []
    trainee_identity = []
    matched_track_ids_json = []
    matched_track_keys_json = []
    matched_identity_ids_json = []
    matched_details_json = []

    for _, row in df.iterrows():
        diag["rows"] += 1
        video_id = _sanitize_video_id(row.get(args.video_id_col, ""))
        boxes_info = _row_person_boxes(row, args.person_boxes_col)
        row_hoi_fps = _measure_hoi_fps(row, args.hoi_sampled_fps)
        diag["hoi_fps_measured" if row_hoi_fps != args.hoi_sampled_fps else "hoi_fps_fallback"] += 1

        row_classes = []
        row_tracks = []
        row_track_keys = []
        row_identity_ids = []
        row_details = []

        for item in boxes_info:
            diag["hoi_boxes"] += 1
            if not isinstance(item, dict):
                diag["invalid_boxes"] += 1
                continue
            frame_id = _safe_int(item.get("frame_id"), default=None)
            box_xywh = item.get("box_xywh")
            source_video_stem = str(item.get("video_stem", "")).strip()
            if frame_id is None or box_xywh is None:
                diag["invalid_boxes"] += 1
                continue

            role_data = _get_or_load_role_csv(
                role_csv_cache, csv_index, csv_index_sim_view, video_id, source_video_stem,
            )
            if role_data is None:
                diag["role_csv_not_found"] += 1
                continue

            hoi_time = frame_id / row_hoi_fps
            match = _match_hoi_to_role(box_xywh, role_data, hoi_time, args)
            if match is None:
                diag["unmatched"] += 1
                continue
            matched, match_type, iou, dist = match
            role_name = matched["role"]
            cls_id = ROLE_NAME_TO_CLASS_ID.get(role_name, -1)
            track_id = matched["track_id"]
            track_key = f"{source_video_stem}::T{track_id}" if source_video_stem else f"T{track_id}"
            row_classes.append(cls_id)
            row_tracks.append(track_id)
            row_track_keys.append(track_key)
            row_identity_ids.append(cls_id)
            row_details.append({
                "frame_id": frame_id,
                "hoi_time_seconds": round(hoi_time, 4),
                "video_stem": source_video_stem,
                "view_id": item.get("view_id", ""),
                "track_key": track_key,
                "track_id": track_id,
                "subject_class_id": cls_id,
                "role_name": role_name,
                "match_type": match_type,
                "iou": round(float(iou), 4),
                "center_dist": round(float(dist), 2),
                "role_conf": round(float(matched["conf"]), 4),
            })
            diag[f"matched_{match_type}"] += 1

        if row_details:
            track_counts = Counter(d["track_key"] for d in row_details)
            # Rank tracks by how many matched boxes they contributed to this HOI row,
            # most first. "Additional Team" rarely performs the interaction itself, so
            # walk down the ranking and take the first track whose role isn't
            # Additional Team; only fall back to the dominant (possibly Additional
            # Team) track if every matched track in this row resolved to it.
            ranked_tracks = sorted(track_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            sid = None
            for track_key, _ in ranked_tracks:
                track_classes = Counter(
                    d["subject_class_id"] for d in row_details
                    if d["track_key"] == track_key
                )
                candidate_sid = sorted(track_classes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                if candidate_sid != ADDITIONAL_TEAM_CLASS_ID:
                    sid = candidate_sid
                    break
            if sid is None:
                dominant_track_key = ranked_tracks[0][0]
                dominant_classes = Counter(
                    d["subject_class_id"] for d in row_details
                    if d["track_key"] == dominant_track_key
                )
                sid = sorted(dominant_classes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            sid = -1
        subject_class_ids.append(sid)
        trainee_identity.append(IDENTITY_MAP.get(sid, ""))
        matched_track_ids_json.append(json.dumps(row_tracks))
        matched_track_keys_json.append(json.dumps(row_track_keys))
        matched_identity_ids_json.append(json.dumps(row_identity_ids))
        matched_details_json.append(json.dumps(row_details))

    df["Subject_Class_ID"] = subject_class_ids
    df["Trainee_Identity"] = trainee_identity
    df["v4_matched_track_ids"] = matched_track_ids_json
    df["v4_matched_track_keys"] = matched_track_keys_json
    df["v4_matched_identity_ids"] = matched_identity_ids_json
    df["v4_match_details"] = matched_details_json

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    slim_cols = [
        "simulation_id", "video_id", "object_group",
        "start_time", "end_time", "duration_seconds",
        "thresholds_used", "sigmas_used", "person_bounding_boxes",
        "Subject_Class_ID", "Trainee_Identity", "v4_matched_identity_ids",
        "v4_matched_track_ids", "v4_matched_track_keys",
    ]
    slim_out = out_csv.with_name("CCATT_Trainee_Equipment_Interactions.csv")
    df.loc[:, [c for c in slim_cols if c in df.columns]].to_csv(slim_out, index=False)
    print(f"Saved: {slim_out}")

    print(
        "Diagnostics: "
        + ", ".join(f"{k}={v}" for k, v in sorted(diag.items()))
        + f", role_csvs_loaded={len(role_csv_cache)}"
    )


def _dry_run_report(args, role_csv_roots):
    role_csv_counts = {str(r): len(list(r.rglob("*.csv"))) for r in role_csv_roots}
    for r, n in role_csv_counts.items():
        print(f"  role_csv_root: {r}  ({n} CSV file(s) found)")
        if n == 0:
            print(f"    [WARN] no role-assignment CSVs here yet -- run generate_role_assignment_csvs.py first")

    if args.input_csv:
        in_csv = Path(args.input_csv).expanduser().resolve()
        if not in_csv.exists():
            raise FileNotFoundError(f"--input_csv not found: {in_csv}")
        cols = pd.read_csv(in_csv, nrows=0).columns
        for col_flag, col_name in [("--person_boxes_col", args.person_boxes_col), ("--video_id_col", args.video_id_col)]:
            if col_name not in cols and not (col_flag == "--person_boxes_col" and "person_bounding_boxes" in cols):
                print(f"  [WARN] {col_flag}='{col_name}' not found in {in_csv.name} (columns: {list(cols)})")
        print(f"  would process: {in_csv} -> {args.output_csv}")
        print("[dry_run] Re-run without --dry_run to actually write output.")
        return

    root = Path(args.input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--input_root not found: {root}")
    csvs = sorted(root.rglob("combined_segments.csv"))
    if not csvs:
        raise FileNotFoundError(f"No combined_segments.csv found under: {root}")
    print(f"  Found {len(csvs)} combined_segments.csv file(s) under {root}")
    for csv_path in csvs:
        out_path = csv_path.with_name("combined_segments_with_person_id_v4.csv")
        slim_out = out_path.with_name("CCATT_Trainee_Equipment_Interactions.csv")
        cols = pd.read_csv(csv_path, nrows=0).columns
        col_ok = (args.person_boxes_col in cols) or ("person_bounding_boxes" in cols)
        video_id_ok = args.video_id_col in cols
        status = "skip (outputs already exist)" if (not args.inplace and out_path.exists() and slim_out.exists()) else "would process"
        problems = []
        if not col_ok:
            problems.append(f"missing person-boxes column '{args.person_boxes_col}'")
        if not video_id_ok:
            problems.append(f"missing video-id column '{args.video_id_col}'")
        problem_note = f"  [WARN] {'; '.join(problems)}" if problems else ""
        print(f"    {status:28s} {csv_path}{problem_note}")
    print("[dry_run] Re-run without --dry_run to actually write output.")


def run(args):
    in_csv = Path(args.input_csv).expanduser().resolve() if args.input_csv else None
    out_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else None
    role_csv_roots = [Path(v).expanduser().resolve() for v in (args.role_csv_root or [])]

    if not role_csv_roots:
        raise ValueError(
            "Provide --role_csv_root (one or more directories containing per-video "
            "role-assignment CSVs produced by generate_role_assignment_csvs.py)."
        )
    missing_roots = [r for r in role_csv_roots if not r.exists()]
    if missing_roots:
        raise FileNotFoundError(f"--role_csv_root folder(s) not found: {missing_roots}")

    if args.dry_run:
        _dry_run_report(args, role_csv_roots)
        return

    if in_csv is not None:
        if out_csv is None:
            raise ValueError("--output_csv is required with --input_csv")
        _process_csv(args, in_csv, out_csv, role_csv_roots)
        return

    if args.input_root:
        root = Path(args.input_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--input_root not found: {root}")
        csvs = sorted(root.rglob("combined_segments.csv"))
        if not csvs:
            raise FileNotFoundError(f"No combined_segments.csv found under: {root}")
        for csv_path in csvs:
            out_path = csv_path.with_name("combined_segments_with_person_id_v4.csv")
            slim_out = out_path.with_name("CCATT_Trainee_Equipment_Interactions.csv")
            if not args.inplace and out_path.exists() and slim_out.exists():
                print(f"Skipping existing outputs: {csv_path.parent}")
                continue
            print(f"Processing: {csv_path}")
            _process_csv(args, csv_path, out_path, role_csv_roots)
        return

    raise ValueError("Provide either (--input_csv + --output_csv) or --input_root.")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 3: match HOI subject boxes to a pre-generated role-assignment CSV "
            "(frame_id, time_seconds, track_id, x1, y1, x2, y2, conf, role) by "
            "elapsed time -- runs no detection/tracking model itself."
        )
    )
    p.add_argument("--input_csv", default=None,
                   help="Process a single combined_segments.csv (must be paired with --output_csv). "
                        "For most uses, prefer --input_root instead.")
    p.add_argument("--output_csv", default=None,
                   help="Output path when using --input_csv.")
    p.add_argument("--input_root", default=None,
                   help="Root folder containing */combined_segments.csv sub-folders "
                        "(the output of step 1, V3_Create_Temporal_Predicted_HOI_intervals_"
                        "version3_Gridsearch_Threshold.py). Processes all of them.")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite this script's own previous outputs (combined_segments_with_person_id_v4.csv / "
                        "CCATT_Trainee_Equipment_Interactions.csv) even if they already exist. Without this flag, "
                        "a combined_segments.csv whose outputs already exist is skipped. Never overwrites "
                        "combined_segments.csv itself.")
    p.add_argument("--role_csv_root", nargs="*", default=None,
                   help="One or more directories containing per-video role-assignment CSVs "
                        "produced by generate_role_assignment_csvs.py.")
    p.add_argument("--person_boxes_col", default="person_bounding_boxes_all_frames",
                   help="Column containing HOI subject boxes. Defaults to all-frame boxes, "
                        "falls back to person_bounding_boxes.")
    p.add_argument("--video_id_col", default="video_id",
                   help="Column identifying which video/simulation a row belongs to (default: video_id).")
    p.add_argument("--hoi_sampled_fps", type=float, default=4.0,
                   help="Fallback rate used only when a row's duration_frames/duration_seconds "
                        "columns are missing/unusable; normally the rate is measured per-row "
                        "from those columns instead of assumed")
    p.add_argument("--iou_threshold", type=float, default=0.3,
                   help="Minimum IoU for an HOI subject box to match a role-assignment row")
    p.add_argument("--center_dist_fallback", type=float, default=50.0,
                   help="Fallback center distance threshold in pixels")
    p.add_argument("--match_time_tolerance", type=float, default=0.5,
                   help="Max |hoi_time - role_csv_time| in seconds to consider a candidate match")
    p.add_argument("--dry_run", action="store_true",
                   help="Check that --role_csv_root and --input_root/--input_csv (and their expected "
                        "columns) exist, and list which files would be processed vs. skipped, without "
                        "writing any output.")
    return p.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)
