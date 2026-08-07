#!/usr/bin/env python3
"""
Step 2 of the CCATT inference pipeline (slow, GPU): generate/refresh per-video
role-assignment CSVs.

What this script does:
  For every video referenced in the combined_segments.csv files under
  --input_root, this script runs a separate person role-identification model
  (role_contrastive/demo_role.py -- see "External model dependency" below)
  once per video and writes a CSV of
  (frame_id, time_seconds, track_id, x1, y1, x2, y2, conf, role) rows to
  --role_csv_root. person_identification_v4.py (step 3) then reads those
  CSVs to attach a trainee role/identity to each HOI interaction.

  This script does NOT reimplement the role-detection/tracking model itself
  -- it only figures out which video file each combined_segments.csv row's
  HOI boxes refer to, and shells out to demo_role.py for that video.

External model dependency (not included in this repository):
  This step calls a separate, already-trained role-identification model
  (weights + a `demo_role.py` script from the "role_contrastive" project).
  You must have that project and its trained weights available locally (or
  point --demo_role_script / --weights / --python_bin at wherever your team
  keeps them). If you don't have this dependency yet, ask the CCATT ML team
  for the "role_contrastive" model folder before running this step.

Slow vs. fast step:
  This is the SLOW, GPU-bound step -- it only needs to be re-run when new
  videos are added or you change --conf/--weights. Videos that already have
  a role-assignment CSV are skipped automatically; pass --force to
  regenerate them anyway. Step 3 (person_identification_v4.py) is fast,
  CPU-only, and safe to re-run any time.

Example:
  python3 generate_role_assignment_csvs.py \\
    --input_root  /path/to/hoi_results/FINAL_DEFAULT_RUN \\
    --video_roots /path/to/videos \\
    --role_csv_root /path/to/role_assignment_csvs \\
    --weights /path/to/role_contrastive/weights/best.pt \\
    --demo_role_script /path/to/role_contrastive/demo_role.py \\
    --python_bin /path/to/role_yolo_env/bin/python3 \\
    --conf 0.10 --device cuda:1

Use --dry_run to check that --input_root/--video_roots/--weights/
--demo_role_script/--python_bin all exist, and see which videos would be
processed vs. skipped, without launching the (slow, GPU) model at all.

See README.md in this repository for the full 3-step pipeline.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# These defaults match the CCATT ML team's shared lab layout. They will not
# exist on a fresh machine -- pass --weights/--demo_role_script/--python_bin
# explicitly (or edit these three lines) to point at your own copy of the
# role_contrastive project described above.
DEFAULT_WEIGHTS = "~/Person_identification/Model_Development/runs/role_contrastive1/weights/best.pt"
DEFAULT_DEMO_SCRIPT = "~/Person_identification/Model_Development/role_contrastive/demo_role.py"
DEFAULT_PYTHON_BIN = "~/miniconda3/envs/role_yolo/bin/python3"

# Passed as --num-frames to demo_role.py. demo_role.py stops as soon as cap.read()
# fails (end of stream), so this is just a "process the whole video" sentinel --
# it deliberately does not rely on CAP_PROP_FRAME_COUNT, which can be wrong/absent
# for some codecs/containers.
WHOLE_VIDEO_NUM_FRAMES = 10_000_000


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


def _build_video_index(video_roots):
    idx = {}
    idx_sim_view = {}
    exts = {".mp4", ".3gp", ".mov", ".mkv", ".avi"}
    for root in video_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
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


def _find_video_path(video_index, video_index_sim_view, video_stem):
    raw = str(video_stem or "").strip()
    sanitized = _sanitize_stem(raw)
    for key in [raw, sanitized]:
        if key and key in video_index:
            return video_index[key]
    sim_key = _extract_sim_key(raw)
    view = _extract_view_token(raw)
    if sim_key and view:
        return video_index_sim_view.get((sim_key, view))
    return None


def _collect_video_stems(input_root: Path, person_boxes_col: str):
    stems = set()
    csvs = sorted(input_root.rglob("combined_segments.csv"))
    print(f"Found {len(csvs)} combined_segments.csv file(s) under {input_root}")
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        col = person_boxes_col if person_boxes_col in df.columns else "person_bounding_boxes"
        if col not in df.columns:
            continue
        for cell in df[col].dropna():
            for item in _parse_boxes(cell):
                if isinstance(item, dict):
                    stem = str(item.get("video_stem", "")).strip()
                    if stem:
                        stems.add(stem)
    return stems


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_root", required=True,
                   help="Root folder containing */combined_segments.csv sub-folders "
                        "(the output of step 1, V3_Create_Temporal_Predicted_HOI_intervals_"
                        "version3_Gridsearch_Threshold.py).")
    p.add_argument("--video_roots", nargs="*", required=True,
                   help="One or more folders to search (recursively) for the original video files "
                        "referenced in --input_root's combined_segments.csv files.")
    p.add_argument("--role_csv_root", required=True,
                   help="Output directory for per-video role-assignment CSVs. Created if missing.")
    p.add_argument("--person_boxes_col", default="person_bounding_boxes_all_frames",
                   help="Column in combined_segments.csv that holds the per-frame HOI subject boxes "
                        "(default: person_bounding_boxes_all_frames).")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="Path to the trained role_contrastive model weights (.pt). "
                        f"Default: {DEFAULT_WEIGHTS}")
    p.add_argument("--conf", type=float, default=0.10,
                   help="Detection confidence threshold passed to demo_role.py (default: 0.10).")
    p.add_argument("--device", default=None, help="Device for the role model, e.g. cuda:0, cuda:1, cpu.")
    p.add_argument("--demo_role_script", default=DEFAULT_DEMO_SCRIPT,
                   help="Path to role_contrastive/demo_role.py. "
                        f"Default: {DEFAULT_DEMO_SCRIPT}")
    p.add_argument("--python_bin", default=DEFAULT_PYTHON_BIN,
                   help="Python interpreter (with the role_contrastive environment's dependencies "
                        f"installed) used to run demo_role.py. Default: {DEFAULT_PYTHON_BIN}")
    p.add_argument("--force", action="store_true",
                   help="Regenerate a video's role-assignment CSV even if one already exists. "
                        "Without this flag, videos with an existing CSV are skipped.")
    p.add_argument("--dry_run", action="store_true",
                   help="Check that --input_root/--video_roots/--weights/--demo_role_script/"
                        "--python_bin all exist, list which videos would be processed vs. "
                        "skipped, and exit -- without running the (slow, GPU) role model.")
    return p.parse_args()


def main():
    args = parse_args()
    args.weights = str(Path(args.weights).expanduser())
    args.demo_role_script = str(Path(args.demo_role_script).expanduser())
    args.python_bin = str(Path(args.python_bin).expanduser())

    input_root = Path(args.input_root).expanduser().resolve()
    role_csv_root = Path(args.role_csv_root).expanduser().resolve()
    video_roots = [Path(v).expanduser().resolve() for v in args.video_roots]

    if not input_root.exists():
        raise FileNotFoundError(
            f"--input_root not found: {input_root}\n"
            "This should be the output folder from step 1 "
            "(V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py) "
            "(it must contain one or more combined_segments.csv files)."
        )
    missing_video_roots = [r for r in video_roots if not r.exists()]
    if missing_video_roots:
        raise FileNotFoundError(f"--video_roots folder(s) not found: {missing_video_roots}")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(
            f"--weights not found: {args.weights}\n"
            "This is the trained role_contrastive model file (best.pt). Pass --weights "
            "pointing at your local copy -- see the 'External model dependency' note "
            "in this script's --help / module docstring."
        )
    if not Path(args.demo_role_script).is_file():
        raise FileNotFoundError(
            f"--demo_role_script not found: {args.demo_role_script}\n"
            "This is demo_role.py from the separate role_contrastive project. Pass "
            "--demo_role_script pointing at your local copy."
        )
    if not Path(args.python_bin).is_file():
        raise FileNotFoundError(
            f"--python_bin not found: {args.python_bin}\n"
            "This should be a Python interpreter with the role_contrastive environment's "
            "dependencies (torch, ultralytics, etc.) installed. Pass --python_bin pointing "
            "at that interpreter, e.g. /path/to/conda/envs/<env>/bin/python3"
        )

    role_csv_root.mkdir(parents=True, exist_ok=True)

    video_stems = _collect_video_stems(input_root, args.person_boxes_col)
    print(f"Found {len(video_stems)} unique video_stem(s) referenced in HOI results")

    video_index, video_index_sim_view = _build_video_index(video_roots)

    if args.dry_run:
        will_run, will_skip, will_miss = [], [], []
        for stem in sorted(video_stems):
            out_csv = role_csv_root / f"{_sanitize_stem(stem)}.csv"
            if out_csv.exists() and not args.force:
                will_skip.append(stem)
                continue
            video_path = _find_video_path(video_index, video_index_sim_view, stem)
            (will_run if video_path is not None else will_miss).append(stem)
        print(
            f"\n[dry_run] Would generate: {len(will_run)}, "
            f"would skip (already exist): {len(will_skip)}, "
            f"missing video file: {len(will_miss)}"
        )
        if will_miss:
            print("[dry_run] Video files not found for these stems (check --video_roots):")
            for s in will_miss:
                print(f"    {s}")
        print("[dry_run] Re-run without --dry_run to actually generate role-assignment CSVs.")
        return

    done, skipped, missing_video, failed = 0, 0, 0, 0
    for stem in sorted(video_stems):
        out_csv = role_csv_root / f"{_sanitize_stem(stem)}.csv"
        if out_csv.exists() and not args.force:
            print(f"[skip] already exists: {out_csv.name}")
            skipped += 1
            continue

        video_path = _find_video_path(video_index, video_index_sim_view, stem)
        if video_path is None:
            print(f"[MISSING VIDEO] no file found for video_stem={stem!r} under --video_roots")
            missing_video += 1
            continue

        cmd = [
            args.python_bin, args.demo_role_script,
            "--weights", args.weights,
            "--source", str(video_path),
            "--start-frame", "0",
            "--num-frames", str(WHOLE_VIDEO_NUM_FRAMES),
            "--conf", str(args.conf),
            "--csv_out", str(out_csv),
            "--skip_video",
        ]
        if args.device:
            cmd += ["--device", args.device]

        print(f"[run] {stem} -> {video_path.name}")
        print("      " + " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[FAILED] {stem} (exit {result.returncode})")
            failed += 1
        else:
            print(f"[done] {stem} -> {out_csv}")
            done += 1

    print(
        f"\nSummary: generated={done}, skipped_existing={skipped}, "
        f"missing_video={missing_video}, failed={failed}, total_stems={len(video_stems)}"
    )
    if missing_video or failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)
