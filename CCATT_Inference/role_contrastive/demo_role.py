#!/usr/bin/env python3
"""
Demo: run a trained RoleDetectionModel + ByteTrack over a video segment and render each
tracked bounding box with a single, whole-video-consistent role label.

Two-pass pipeline (see role_assignment.py's module docstring for why this is necessary):
  Pass 1: track the whole segment once, applying per-frame singleton resolution
          (resolve_singleton_conflicts), accumulating each track's full vote history, and
          recording which tracks ever coexisted in the same frame (build_track_conflicts).
  Global resolution: resolve_global_track_roles() picks a final label per track, only
          forcing tracks that actually coexisted (a real conflict) to compete for a singleton
          role — tracks that never appeared together (e.g. the same real person's ByteTrack
          ID fragmenting after a brief occlusion) can validly share a role, since they were
          never shown as two Nurses at once.
  Pass 2: re-read the same frames (cheap — no re-running the model) and draw each box with
          its track's finalized label.

Reuses the model.track(source=..., tracker="bytetrack.yaml", persist=True) call pattern from
Model_Development/Model_Training.ipynb (cell 56/58).

Usage:
  python demo_role.py --weights ../runs/role_contrastive1/weights/best.pt \
      --source ../temp_trimmed.mp4 --start-frame 0 --num-frames 600 \
      --out demo_role_output.mp4
"""

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
from ultralytics import YOLO

from role_assignment import (
    CLASS_NAMES,
    TrackRoleSmoother,
    build_track_conflicts,
    resolve_global_track_roles,
    resolve_singleton_conflicts,
)

HERE = Path(__file__).resolve().parent


def _measure_fps(source: str, cap: cv2.VideoCapture) -> float:
    """cv2.CAP_PROP_FPS reads 0/None for some files in this dataset (container/codec
    quirk) -- silently guessing a constant in that case (e.g. `or 22.0`) previously
    corrupted every time_seconds value for the affected videos with no warning. Fall
    back to ffprobe, which reliably reads the real fps for every file checked so far;
    raise loudly rather than guess if even that fails."""
    cv2_fps = cap.get(cv2.CAP_PROP_FPS)
    if cv2_fps and cv2_fps > 1:
        return float(cv2_fps)

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        num, _, den = out.partition("/")
        ffprobe_fps = float(num) / float(den) if den else float(num)
        if ffprobe_fps > 1:
            return ffprobe_fps
    except Exception:
        pass

    raise RuntimeError(
        f"Could not reliably measure fps for {source} via cv2 (got {cv2_fps!r}) or ffprobe. "
        "Refusing to guess a constant -- fix the source file or extend this function."
    )


def run_demo(
    weights: str,
    source: str,
    out_path: str,
    start_frame: int,
    num_frames: int,
    conf: float = 0.25,
    device: str = None,
    csv_out: str = None,
    skip_video: bool = False,
) -> None:
    model = YOLO(weights)
    smoother = TrackRoleSmoother()

    # ---- Pass 1: track the whole segment, accumulate per-track vote history ----
    per_frame_boxes: list[tuple] = []  # (xyxy, track_ids, confs) or None per processed frame
    per_frame_track_ids: list[list[int]] = []  # for build_track_conflicts()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")
    fps = _measure_fps(source, cap)
    print(f"[demo_role] measured fps={fps:.4f} for {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed = 0
    while processed < num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            source=frame, tracker="bytetrack.yaml", persist=True, conf=conf, device=device, verbose=False
        )
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)
            labels = [CLASS_NAMES[c] for c in cls_ids]

            resolved = resolve_singleton_conflicts(labels, confs.tolist())
            for tid, lab in zip(track_ids, resolved):
                smoother.update(int(tid), lab)

            per_frame_boxes.append((xyxy, track_ids, confs))
            per_frame_track_ids.append(track_ids.tolist())
        else:
            per_frame_boxes.append(None)
        processed += 1
    cap.release()

    # ---- Global resolution: one final label per track, only enforcing exclusivity between
    # tracks that were ever visible in the same frame together ----
    track_conflicts = build_track_conflicts(per_frame_track_ids)
    final_labels = resolve_global_track_roles(smoother.all_votes(), track_conflicts)

    # ---- CSV export: one row per (frame, tracked box), using the finalized per-track label.
    # Written straight from the Pass-1 in-memory records -- no video re-read needed. ----
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["frame_id", "time_seconds", "track_id", "x1", "y1", "x2", "y2", "conf", "role"])
            for i, record in enumerate(per_frame_boxes):
                if record is None:
                    continue
                xyxy, track_ids, confs = record
                frame_id = start_frame + i
                time_seconds = frame_id / fps
                for (x1, y1, x2, y2), tid, box_conf in zip(xyxy, track_ids, confs):
                    lab = final_labels.get(int(tid), "Additional Staff")
                    writer_csv.writerow([frame_id, round(time_seconds, 4), int(tid), x1, y1, x2, y2, float(box_conf), lab])
        print(f"Wrote role assignment CSV to {csv_out}")

    if skip_video:
        print(f"Final track roles: {final_labels}")
        return

    # ---- Pass 2: re-read frames and render using the finalized labels ----
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for record in per_frame_boxes:
        ok, frame = cap.read()
        if not ok:
            break

        if record is not None:
            xyxy, track_ids, confs = record
            for (x1, y1, x2, y2), tid, box_conf in zip(xyxy, track_ids, confs):
                lab = final_labels.get(int(tid), "Additional Staff")
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, (0, 200, 0), 2)
                text = f"#{tid} {lab} {box_conf:.2f}"
                cv2.putText(frame, text, (p1[0], max(0, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Wrote {len(per_frame_boxes)} frame(s) to {out_path}")
    print(f"Final track roles: {final_labels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default=str(HERE.parent / "temp_trimmed.mp4"))
    parser.add_argument("--out", default=str(HERE / "demo_role_output.mp4"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=600)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu; default lets ultralytics pick")
    parser.add_argument("--csv_out", default=None, help="Path to write per-frame per-track role CSV")
    parser.add_argument("--skip_video", action="store_true", help="Skip Pass 2 video rendering (CSV only)")
    args = parser.parse_args()

    run_demo(
        args.weights, args.source, args.out, args.start_frame, args.num_frames, args.conf,
        device=args.device, csv_out=args.csv_out, skip_video=args.skip_video,
    )


if __name__ == "__main__":
    main()
