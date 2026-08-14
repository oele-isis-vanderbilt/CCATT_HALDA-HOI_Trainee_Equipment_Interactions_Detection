#!/usr/bin/env python3
"""
Run CDN inference on clear frames and auto-label interaction vs no_interaction (57)
according to confidence and spatial rules.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
import sys
import shutil
from pathlib import Path

import pandas as pd
import torch


def load_helpers(root: Path):
    helper = root / "5_active_learning_iteration.py"
    spec = importlib.util.spec_from_file_location("five_active_learning", helper)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise SystemExit(f"Cannot load helpers from {helper}")
    spec.loader.exec_module(module)  # type: ignore
    return module


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iter_jpg_files(clear_root: Path):
    for dirpath, dirnames, filenames in os.walk(clear_root, followlinks=True):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(".jpg"):
                img = Path(dirpath) / name
                if img.is_file():
                    yield img


def chunked(iterable, chunk_size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def materialize_chunk(
    clear_root: Path,
    state_dir: Path,
    chunk_index: int,
    images: list[Path],
) -> tuple[Path, dict[str, str]]:
    """
    CDN expects a flat directory of images. Materialize only one chunk at a time
    to keep filesystem and memory pressure bounded.
    """
    flat_dir = state_dir / "flat_frames" / f"chunk_{chunk_index:05d}"
    if flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    for idx, img in enumerate(images):
        rel = img.relative_to(clear_root)
        name = f"{chunk_index:05d}_{idx:08d}__{str(rel).replace('/', '__')}"
        target = flat_dir / name
        try:
            target.symlink_to(img)
        except OSError:
            shutil.copy2(img, target)
        mapping[name] = str(rel)
    return flat_dir, mapping


def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def chunk_paths(state_dir: Path, chunk_index: int) -> dict[str, Path]:
    flat_dir = state_dir / "flat_frames" / f"chunk_{chunk_index:05d}"
    raw_csv = state_dir / f"chunk_{chunk_index:05d}_cdn_preds_raw.csv"
    status_dir = state_dir / "chunk_status"
    chunk_dir = state_dir / "chunk_outputs"
    return {
        "flat_dir": flat_dir,
        "raw_csv": raw_csv,
        "chunk_dir": chunk_dir,
        "chunk_raw_csv": chunk_dir / f"chunk_{chunk_index:05d}_cdn_preds_raw.csv",
        "chunk_std_csv": chunk_dir / f"chunk_{chunk_index:05d}_cdn_preds_std.csv",
        "chunk_labels_csv": chunk_dir / f"chunk_{chunk_index:05d}_cdn_labels.csv",
        "chunk_high_labels_csv": chunk_dir / f"chunk_{chunk_index:05d}_high_conf_labels.csv",
        "status_dir": status_dir,
        "done": status_dir / f"chunk_{chunk_index:05d}.done",
        "started": status_dir / f"chunk_{chunk_index:05d}.started",
    }


def cleanup_unfinished_chunk(state_dir: Path, chunk_index: int) -> None:
    paths = chunk_paths(state_dir, chunk_index)
    if paths["done"].exists():
        return
    if paths["flat_dir"].exists():
        shutil.rmtree(paths["flat_dir"], ignore_errors=True)
    if paths["raw_csv"].exists():
        paths["raw_csv"].unlink()
    if paths["started"].exists():
        paths["started"].unlink()


def cleanup_stale_chunks(state_dir: Path) -> None:
    flat_root = state_dir / "flat_frames"
    if flat_root.exists():
        for child in flat_root.iterdir():
            if not child.is_dir() or not child.name.startswith("chunk_"):
                continue
            try:
                chunk_index = int(child.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            cleanup_unfinished_chunk(state_dir, chunk_index)


def load_existing_frame_scores(labels_path: Path) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    if not labels_path.exists():
        return scores
    try:
        df = pd.read_csv(labels_path, usecols=["frame_file", "verb_score"])
    except Exception:
        return scores
    for row in df.itertuples(index=False):
        try:
            frame = str(row.frame_file)
            score = float(row.verb_score)
        except Exception:
            continue
        scores.setdefault(frame, []).append(score)
    return scores

def main():
    ap = argparse.ArgumentParser(description="CDN auto-labeling for verb 117 vs no_interaction (57).")
    ap.add_argument("--clear_root", type=Path, required=True, help="Directory with clear frames.")
    ap.add_argument("--state_dir", type=Path, required=True, help="Output directory for predictions/labels.")
    ap.add_argument("--cdn_repo", type=Path, required=True, help="CDN repository path.")
    ap.add_argument("--cdn_weights", type=Path, required=True, help="CDN weights path.")
    ap.add_argument("--device", default="cuda", help="Device for CDN inference.")
    ap.add_argument("--cdn_extra_args", default="", help="Extra args for CDN (e.g., '--device cuda:1').")
    ap.add_argument("--num_obj_classes", type=int, default=92, help="Number of object classes for CDN model.")
    ap.add_argument("--num_verb_classes", type=int, default=118, help="Number of verb classes for CDN model.")
    ap.add_argument("--score_thresh", type=float, default=0.5, help="Min object confidence score.")
    ap.add_argument("--subject_score_thresh", type=float, default=0.1, help="Keep only rows with subject score > this threshold.")
    ap.add_argument("--pos_verb_thresh", type=float, default=0.90, help="Verb score >= this -> verb 117.")
    ap.add_argument("--neg_verb_thresh", type=float, default=0.10, help="Verb score <= this -> verb 57.")
    ap.add_argument("--dist_thresh", type=float, default=150.0, help="Pixels; if no overlap and distance > thr -> verb 57.")
    ap.add_argument("--out_pred", type=Path, help="Path to write raw CDN preds CSV.")
    ap.add_argument("--out_std", type=Path, help="Path to write standardized preds CSV.")
    ap.add_argument("--out_labels", type=Path, help="Path to write labeled HOIs CSV.")
    ap.add_argument("--out_low_frames", type=Path, help="Path to write low-confidence frame list.")
    ap.add_argument("--out_high_frames", type=Path, help="Path to write high-confidence frame list.")
    ap.add_argument("--out_high_labels", type=Path, help="Path to write high-confidence HOI labels CSV.")
    ap.add_argument("--chunk_size", type=int, default=50000, help="Number of images to materialize and infer per chunk.")
    ap.add_argument("--write_global_raw", action="store_true", help="Append every chunk raw CSV into out_pred.")
    ap.add_argument("--write_global_std", action="store_true", help="Append every chunk standardized CSV into out_std.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    helpers = load_helpers(root)
    CDNClient = helpers.CDNClient
    standardize_cdn_predictions = helpers.standardize_cdn_predictions
    _iou = helpers._iou

    args.state_dir.mkdir(parents=True, exist_ok=True)
    out_pred = args.out_pred or args.state_dir / "cdn_preds_raw.csv"
    out_std = args.out_std or args.state_dir / "cdn_preds_std.csv"
    out_labels = args.out_labels or args.state_dir / "cdn_labels.csv"
    out_low = args.out_low_frames or args.state_dir / "low_conf_frames.txt"
    out_high = args.out_high_frames or args.state_dir / "high_conf_frames.txt"
    out_high_labels = args.out_high_labels or args.state_dir / "high_conf_labels.csv"
    status_dir = args.state_dir / "chunk_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (args.state_dir / "chunk_outputs").mkdir(parents=True, exist_ok=True)
    cleanup_stale_chunks(args.state_dir)

    VERB_ID = 117
    NO_INTERACTION = 57

    # Device sanity
    if "cuda" in args.device and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    client = CDNClient(
        args.cdn_repo,
        args.cdn_weights,
        device=args.device,
        extra_args=args.cdn_extra_args,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
    )

    frame_scores: dict[str, list[float]] = load_existing_frame_scores(out_labels)
    total_images = 0
    total_labeled_rows = 0
    saw_any_images = False
    chunk_size = max(1, args.chunk_size)

    for chunk_index, images in enumerate(chunked(iter_jpg_files(args.clear_root), chunk_size), start=1):
        saw_any_images = True
        paths = chunk_paths(args.state_dir, chunk_index)
        if paths["done"].exists():
            print(f"[CDN] chunk {chunk_index:05d}: SKIP already done")
            continue

        cleanup_unfinished_chunk(args.state_dir, chunk_index)
        total_images += len(images)
        paths["status_dir"].mkdir(parents=True, exist_ok=True)
        paths["started"].write_text(str(int(time.time())))
        print(f"[CDN] chunk {chunk_index:05d}: START images={len(images)}")
        frames_dir, name_map = materialize_chunk(args.clear_root, args.state_dir, chunk_index, images)

        try:
            print(f"[CDN] chunk {chunk_index:05d}: INFER")
            client.run_inference(frames_dir, paths["raw_csv"])
            print(f"[CDN] chunk {chunk_index:05d}: INFER done")

            raw = pd.read_csv(paths["raw_csv"], dtype=str)
            raw.to_csv(paths["chunk_raw_csv"], index=False)
            if args.write_global_raw:
                append_csv(raw, out_pred)

            std = standardize_cdn_predictions(raw)
            if std.empty:
                print(f"[CDN] chunk {chunk_index:05d}: POST no standardized predictions")
                labeled_rows = 0
                high_rows = 0
            else:
                std["frame_file"] = std["frame_file"].map(lambda x: name_map.get(x, x))
                std["frame_stem"] = std["frame_file"].apply(lambda x: Path(x).stem)
                std.to_csv(paths["chunk_std_csv"], index=False)
                if args.write_global_std:
                    append_csv(std, out_std)

                std["score"] = std["score"].astype(float)
                std["object_score"] = std["object_score"].astype(float)
                std["verb_score"] = std["verb_score"].astype(float)

                # Use only CDN predictions for verb_id 117 so thresholding is based on
                # the verb score of verb 117 specifically.
                std = std[
                    (std["verb_id"] == VERB_ID)
                    & (std["object_score"] >= args.score_thresh)
                    & (std["score"] > args.subject_score_thresh)
                ]

                records = []
                for row in std.itertuples(index=False):
                    subj = row.subject_box
                    obj = row.object_box
                    verb_score = float(row.verb_score)
                    frame = row.frame_file

                    # Threshold policy:
                    # - score >= high threshold  => interaction (GT verb_id 117)
                    # - score <= low threshold   => no_interaction (GT verb_id 57)
                    # - otherwise                => keep current uncertain middle-band handling
                    if verb_score >= args.pos_verb_thresh:
                        decision = "positive_117"
                        verb_out = VERB_ID
                    elif verb_score <= args.neg_verb_thresh:
                        decision = "negative_low_score"
                        verb_out = NO_INTERACTION
                    else:
                        iou = _iou(subj, obj)
                        cx1, cy1 = center(subj)
                        cx2, cy2 = center(obj)
                        dist = math.hypot(cx1 - cx2, cy1 - cy2)
                        if iou == 0 and dist > args.dist_thresh:
                            decision = "negative_far"
                            verb_out = NO_INTERACTION
                        else:
                            decision = "uncertain_mid_score"
                            verb_out = VERB_ID

                    records.append(
                        {
                            "frame_file": frame,
                            "frame_stem": row.frame_stem,
                            "subject_box": json.dumps(subj),
                            "object_box": json.dumps(obj),
                            "verb_id": int(row.verb_id),
                            "object_id": int(row.object_id),
                            "verb_score": verb_score,
                            "object_score": float(row.object_score),
                            "subject_score": float(row.score),
                            "verb_out": verb_out,
                            "decision": decision,
                        }
                    )
                    frame_scores.setdefault(frame, []).append(verb_score)

                df = pd.DataFrame.from_records(records)
                if not df.empty:
                    df.to_csv(paths["chunk_labels_csv"], index=False)
                    append_csv(df, out_labels)
                    total_labeled_rows += len(df)
                    high_df = df[df["decision"] != "uncertain_mid_score"].copy()
                    high_df.to_csv(paths["chunk_high_labels_csv"], index=False)
                    append_csv(high_df, out_high_labels)
                    labeled_rows = len(df)
                    high_rows = len(high_df)
                else:
                    pd.DataFrame().to_csv(paths["chunk_labels_csv"], index=False)
                    pd.DataFrame().to_csv(paths["chunk_high_labels_csv"], index=False)
                    labeled_rows = 0
                    high_rows = 0

                print(
                    f"[CDN] chunk {chunk_index:05d}: POST labeled_rows={labeled_rows} high_conf_rows={high_rows}"
                )
        except Exception:
            print(f"[CDN] chunk {chunk_index:05d}: FAILED")
            raise
        finally:
            if paths["raw_csv"].exists():
                paths["raw_csv"].unlink()
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)

        print(f"[CDN] chunk {chunk_index:05d}: CLEAN")

        paths["done"].write_text(
            json.dumps(
                {
                    "chunk_index": chunk_index,
                    "images": len(images),
                    "labeled_rows": labeled_rows,
                    "high_conf_rows": high_rows,
                },
                indent=2,
            )
        )
        if paths["started"].exists():
            paths["started"].unlink()
        print(f"[CDN] chunk {chunk_index:05d}: DONE")

    if not saw_any_images:
        raise SystemExit(f"No .jpg frames found under {args.clear_root}")

    low_frames = []
    high_frames = []
    for frame, scores in frame_scores.items():
        if any(args.neg_verb_thresh < s < args.pos_verb_thresh for s in scores):
            low_frames.append(frame)
        else:
            high_frames.append(frame)

    out_low.parent.mkdir(parents=True, exist_ok=True)
    out_high.parent.mkdir(parents=True, exist_ok=True)
    out_low.write_text("\n".join(sorted(low_frames)))
    out_high.write_text("\n".join(sorted(high_frames)))

    print(f"[CDN] Processed images: {total_images}")
    print(f"[CDN] Labeled rows: {total_labeled_rows} -> {out_labels}")
    print(f"[CDN] Low-confidence frames: {len(low_frames)} -> {out_low}")
    print(f"[CDN] High-confidence frames: {len(high_frames)} -> {out_high}")
    print(f"[CDN] High-confidence HOI labels -> {out_high_labels}")


if __name__ == "__main__":
    sys.exit(main())
