#!/usr/bin/env python3
"""
Visualize a handful of Phase 2 annotation rows on their frames.

Usage:
  python Main_Code/5_visualize_phase2_annotations.py \
    --state_dir /path/to/Phase2_Training/Data_Creation \
    --frames_dir /path/to/phase0/2_sampled_frames/<video_dir> \
    [--output_dir /path/to/save_viz] [--samples 10] [--iteration 0] [--labels_csv path/to/custom.csv]
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_box(raw) -> Optional[list[int]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, tuple)):
        vals = raw
    else:
        try:
            vals = ast.literal_eval(str(raw))
        except Exception:
            return None
    try:
        return [int(float(v)) for v in vals]
    except Exception:
        return None


def pick_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["final_label", "label", "cdn_label"]:
        if col in df.columns:
            return col
    return None


def load_candidates(state_dir: Path, iteration: Optional[int], labels_csv: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (auto_df, manual_df). If labels_csv is provided, treat as auto."""
    auto_df = pd.DataFrame()
    manual_df = pd.DataFrame()

    if labels_csv and labels_csv.exists():
        auto_df = pd.read_csv(labels_csv)
        return auto_df, manual_df

    if iteration is not None:
        iter_dir = state_dir / f"iteration_{iteration:03d}"
        auto_path = iter_dir / "auto_labels.csv"
        manual_path = iter_dir / "manual_review.csv"
        if auto_path.exists():
            auto_df = pd.read_csv(auto_path)
        if manual_path.exists():
            manual_df = pd.read_csv(manual_path)

    if auto_df.empty:
        labeled_path = state_dir / "labeled.csv"
        if labeled_path.exists():
            auto_df = pd.read_csv(labeled_path)

    return auto_df, manual_df


def score_row(row) -> float:
    for key in ("verb_score", "object_score", "cdn_score"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                continue
    return 0.0

def _frame_name_from_row(row) -> Optional[str]:
    frame_path = row.get("frame_path")
    frame_file = row.get("frame_file")
    if isinstance(frame_file, str) and frame_file:
        return Path(str(frame_file)).name
    if isinstance(frame_path, str) and frame_path:
        return Path(str(frame_path)).name
    return None

def _row_has_positive_118(row) -> bool:
    # Prefer explicit verb columns in order; treat legacy 117 as positive interaction too.
    for key in ("verb_out", "verb_id", "cdn_label_verb"):
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            vi = int(float(v))
        except Exception:
            continue
        if vi in (117, 118):
            return True
    return False


# def nms_per_object(rows: list[pd.Series], iou_thresh: float = 0.5) -> list[pd.Series]:
#     """Deduplicate HO pairs per object_id using subject-box IoU NMS."""
#     kept = []
#     rows_sorted = sorted(rows, key=lambda r: score_row(r), reverse=True)
#     seen_for_obj: dict[int, list[pd.Series]] = {}
#     for r in rows_sorted:
#         obj_id = r.get("object_id")
#         subj_box = parse_box(r.get("person_bbox")) or parse_box(r.get("subject_box"))
#         if obj_id is None or subj_box is None:
#             kept.append(r)
#             continue
#         dup = False
#         for prev in seen_for_obj.get(obj_id, []):
#             prev_box = parse_box(prev.get("person_bbox")) or parse_box(prev.get("subject_box"))
#             if box_iou(subj_box, prev_box) >= iou_thresh:
#                 dup = True
#                 break
#         if dup:
#             continue
#         kept.append(r)
#         seen_for_obj.setdefault(obj_id, []).append(r)
#     return kept


# def merge_pairs(rows: list[pd.Series], iou_thresh: float = 0.5) -> list[dict]:
#     """
#     Ensure a single verb per subject/object pair. Prefer already-mapped verb_ids (118/58);
#     fall back to a score-based guess only if verb_id is missing. Matching is by object_id
#     and subject-box IoU.
#     """
#     pairs: dict[int, list[dict]] = {}
#     for r in rows:
#         obj_id = r.get("object_id")
#         subj_box = parse_box(r.get("person_bbox")) or parse_box(r.get("subject_box"))
#         verb_score_raw = r.get("verb_score")
#         try:
#             verb_score_val = float(verb_score_raw) if verb_score_raw is not None else 0.0
#         except Exception:
#             verb_score_val = 0.0

#         mapped_vid = None
#         for key in ("verb_id", "cdn_label_verb"):
#             if key in r and pd.notna(r.get(key)):
#                 try:
#                     mapped_vid = int(r.get(key))
#                     break
#                 except Exception:
#                     continue
#         if mapped_vid == 117:
#             mapped_vid = 118
#         if mapped_vid not in (117, 118, 58):
#             mapped_vid = 118 if verb_score_val >= 0.1 else 58

#         if obj_id is None or subj_box is None:
#             row_dict = r.to_dict()
#             row_dict["verb_id"] = mapped_vid
#             row_dict["verb_score"] = verb_score_val
#             pairs.setdefault(-1, []).append({"box": subj_box, "row": row_dict})
#             continue
#         matched = None
#         for pair in pairs.get(obj_id, []):
#             if box_iou(subj_box, pair["box"]) >= iou_thresh:
#                 matched = pair
#                 break
#         if matched:
#             # Prefer mapped positive (118) when available; otherwise keep the higher verb score.
#             current_vid = matched["row"].get("verb_id")
#             prefer_new = False
#             if mapped_vid == 118 and current_vid != 118:
#                 prefer_new = True
#             elif mapped_vid == current_vid:
#                 prefer_new = verb_score_val > matched["row"].get("verb_score", 0)
#             elif verb_score_val > matched["row"].get("verb_score", 0):
#                 prefer_new = True
#             if prefer_new:
#                 matched["row"] = {**r.to_dict(), "verb_id": mapped_vid, "verb_score": verb_score_val}
#                 matched["box"] = subj_box
#         else:
#             pairs.setdefault(obj_id, []).append({"box": subj_box, "row": {**r.to_dict(), "verb_id": mapped_vid, "verb_score": verb_score_val}})
#     merged = []
#     for entries in pairs.values():
#         for entry in entries:
#             merged.append(entry["row"])
#     return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize Phase 2 annotations on sampled frames.")
    ap.add_argument("--state_dir", required=True, type=Path, help="Phase2 state directory (with labeled.csv).")
    ap.add_argument("--frames_dir", required=True, type=Path, help="Directory containing the sampled frames used for this run.")
    ap.add_argument("--output_dir", type=Path, default=None, help="Where to save visualizations (default: state_dir/vis_samples/iter_xxx).")
    ap.add_argument("--samples", type=int, default=10, help="Number of rows to visualize.")
    ap.add_argument("--iteration", type=int, help="Iteration index (used to pick manual_review.csv).")
    ap.add_argument("--labels_csv", type=Path, help="Optional explicit path to an annotations CSV.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = ap.parse_args()

    random.seed(args.seed)

    auto_df, manual_df = load_candidates(args.state_dir, args.iteration, args.labels_csv)
    if auto_df.empty and manual_df.empty:
        raise SystemExit("Annotation CSV is empty; nothing to visualize.")

    # Visualization-only: do not filter rows by label presence.
    auto_label_col = pick_label_column(auto_df) if not auto_df.empty else None
    if not auto_df.empty:
        auto_df = auto_df.copy()
        auto_df["viz_group"] = "auto"
    if not manual_df.empty:
        manual_df = manual_df.copy()
        manual_df["viz_group"] = "manual"
    combined_df = pd.concat([df for df in [auto_df, manual_df] if not df.empty], ignore_index=True)
    if combined_df.empty:
        raise SystemExit("No rows to visualize after filtering.")

    # Frame-level sampling with priority for rows containing positive interaction verb (118/117).
    frame_rows: dict[str, list[dict]] = {}
    for _, r in combined_df.iterrows():
        rd = r.to_dict()
        fn = _frame_name_from_row(rd)
        if not fn:
            continue
        frame_rows.setdefault(fn, []).append(rd)
    if not frame_rows:
        raise SystemExit("No valid frame rows to visualize.")

    pos_frames = [fn for fn, rows in frame_rows.items() if any(_row_has_positive_118(rr) for rr in rows)]
    other_frames = [fn for fn in frame_rows.keys() if fn not in set(pos_frames)]
    random.shuffle(pos_frames)
    random.shuffle(other_frames)

    target_n = max(1, int(args.samples))
    chosen_frames = (pos_frames + other_frames)[:target_n]
    if not pos_frames:
        print("[Viz] No valid interactions (verb 118/117) found; visualizing other frames.")
    else:
        print(f"[Viz] Prioritized {min(len(pos_frames), target_n)} frame(s) with verb 118/117.")
    combined_df = combined_df[combined_df.apply(lambda r: (_frame_name_from_row(r) in set(chosen_frames)), axis=1)].copy()

    out_root = args.output_dir
    if out_root is None:
        iter_part = f"iteration_{args.iteration:03d}" if args.iteration is not None else "labeled"
        out_root = args.state_dir / "vis_samples" / iter_part
    ensure_dir(out_root)

    missing_imgs = 0
    saved = 0

    def render(
        df_use: pd.DataFrame,
        out_root: Path,
        box_color_roi=(0, 255, 0),
        arrow_color=(0, 255, 0),
        text_color=(0, 255, 255),
        ignore_verb: bool = False,
        person_color=(0, 0, 255),
        manual_box_color_roi=(0, 0, 0),
        manual_arrow_color=(0, 0, 0),
        manual_text_color=(0, 0, 0),
        manual_person_color=(0, 0, 0),
    ):
        nonlocal missing_imgs, saved
        if df_use.empty:
            return
        label_col_local = pick_label_column(df_use)
        frames = {}
        for _, row in df_use.iterrows():
            frame_path = row.get("frame_path")
            frame_file = row.get("frame_file")
            frame_name = Path(str(frame_file)).name if isinstance(frame_file, str) and frame_file else None
            if frame_name is None and isinstance(frame_path, str):
                frame_name = Path(frame_path).name
            if not frame_name:
                continue
            entry = frames.setdefault(frame_name, {"rows": [], "path": frame_path, "file": frame_file})
            entry["rows"].append(row)

        for frame_name, data in frames.items():
            frame_path = data.get("path")
            frame_file = data.get("file")
            img_path = Path(str(frame_path)) if isinstance(frame_path, str) else None
            if img_path is None or not img_path.exists():
                if frame_file and isinstance(frame_file, str):
                    img_path = args.frames_dir / frame_file
            if img_path is None or not img_path.exists():
                missing_imgs += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                missing_imgs += 1
                continue

            drawn_person_boxes: set[tuple[int, int, int, int]] = set()
            rows_to_draw = [r.to_dict() if hasattr(r, "to_dict") else r for r in data["rows"]]
            for row in rows_to_draw:
                viz_group = row.get("viz_group", "")
                is_manual = viz_group == "manual"
                row_person_color = manual_person_color if is_manual else person_color
                row_roi_color = manual_box_color_roi if is_manual else box_color_roi
                row_arrow_color = manual_arrow_color if is_manual else arrow_color
                row_text_color = manual_text_color if is_manual else text_color
                ignore_verb_row = ignore_verb or is_manual

                person_box = parse_box(row.get("person_bbox")) or parse_box(row.get("subject_box"))
                roi_box = parse_box(row.get("roi_bbox")) or parse_box(row.get("object_box"))
                verb_id = row.get("verb_out")
                if verb_id is None or (isinstance(verb_id, float) and pd.isna(verb_id)):
                    verb_id = row.get("verb_id")
                if verb_id is None or (isinstance(verb_id, float) and pd.isna(verb_id)):
                    verb_id = row.get("cdn_label_verb")
                label_val = row.get(label_col_local) if label_col_local else None
                src = row.get("label_source", "")
                cid = row.get("candidate_id", "")
                object_id = row.get("object_id")
                obj_score = row.get("object_score")
                person_score = row.get("person_score")
                if person_score is None or (isinstance(person_score, float) and pd.isna(person_score)):
                    person_score = row.get("subject_score")
                verb_score_val = row.get("verb_score") if "verb_score" in row else None

                if person_box:
                    person_key = (person_box[0], person_box[1], person_box[2], person_box[3])
                    if person_key not in drawn_person_boxes:
                        cv2.rectangle(img, (person_box[0], person_box[1]), (person_box[2], person_box[3]), row_person_color, 2)
                        ptxt = None
                        try:
                            ptxt = f"p={float(person_score):.3f}"
                        except Exception:
                            if person_score is not None:
                                ptxt = f"p={person_score}"
                        if ptxt:
                            cv2.putText(
                                img,
                                ptxt,
                                (person_box[0], max(15, person_box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                row_person_color,
                                2,
                            )
                        drawn_person_boxes.add(person_key)
                if roi_box:
                    cv2.rectangle(img, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), row_roi_color, 2)
                    otxt = None
                    try:
                        otxt = f"o={float(obj_score):.3f}"
                    except Exception:
                        if obj_score is not None:
                            otxt = f"o={obj_score}"
                    if object_id is not None and not (isinstance(object_id, float) and pd.isna(object_id)):
                        otxt = f"Oid:{object_id}" if not otxt else f"Oid:{object_id} {otxt}"
                    if otxt:
                        cv2.putText(
                            img,
                            otxt,
                            (roi_box[0], max(15, roi_box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            row_roi_color,
                            2,
                        )

                draw_arrow = False
                arrow_color_use = row_arrow_color
                if not ignore_verb_row and person_box and roi_box:
                    decision = str(row.get("decision", "")).strip().lower()
                    if decision == "positive_117":
                        arrow_color_use = (0, 255, 0)  # interaction: green
                        draw_arrow = True
                    elif decision == "negative_low_score":
                        arrow_color_use = (255, 0, 0)  # no interaction by low score: blue
                        draw_arrow = True
                    else:
                        # Missing/other decision values are treated as uncertain for visualization.
                        arrow_color_use = (128, 128, 128)  # all other/uncertain cases: gray
                        draw_arrow = True
                if draw_arrow:
                    px = (person_box[0] + person_box[2]) // 2
                    py = (person_box[1] + person_box[3]) // 2
                    rx = (roi_box[0] + roi_box[2]) // 2
                    ry = (roi_box[1] + roi_box[3]) // 2
                    cv2.arrowedLine(img, (px, py), (rx, ry), arrow_color_use, 2, tipLength=0.08)
                    mid_x = (px + rx) // 2
                    mid_y = (py + ry) // 2
                    score_text_parts = []
                    try:
                        score_text_parts.append(f"obj={float(obj_score):.3f}")
                    except Exception:
                        if obj_score is not None:
                            score_text_parts.append(f"obj={obj_score}")
                    try:
                        score_text_parts.append(f"verb={float(verb_score_val):.3f}")
                    except Exception:
                        if verb_score_val is not None:
                            score_text_parts.append(f"verb={verb_score_val}")
                    if score_text_parts:
                        cv2.putText(
                            img,
                            " ".join(score_text_parts),
                            (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            arrow_color_use,
                            2,
                        )

                obj_score_str = ""
                if obj_score is not None:
                    try:
                        obj_score_str = f"obj_score={float(obj_score):.3f}"
                    except Exception:
                        obj_score_str = f"obj_score={obj_score}"

                verb_score_str = ""
                if verb_score_val is not None:
                    try:
                        verb_score_str = f"verb_score={float(verb_score_val):.3f}"
                    except Exception:
                        verb_score_str = f"verb_score={verb_score_val}"

                lines = [
                    f"{frame_name}",
                    f"cid={cid}" if cid else "",
                    f"Oid:{object_id}" if object_id is not None and not (isinstance(object_id, float) and pd.isna(object_id)) else "",
                    f"verb={verb_id}" if pd.notna(verb_id) else "",
                    f"label={label_val}" if label_val is not None and not pd.isna(label_val) else "label=?",
                    f"src={src}" if isinstance(src, str) and src else "",
                    obj_score_str,
                    verb_score_str,
                ]
                text = "\n".join([ln for ln in lines if ln])
                y = 25
                for ln in text.split("\n"):
                    cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, row_text_color, 2)
                    y += 24

            out_path = out_root / frame_name
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), img)
            saved += 1

    if auto_df.empty and manual_df.empty:
        raise SystemExit("No labeled rows to visualize (after filtering).")
    render(
        combined_df,
        out_root,
        box_color_roi=(0, 255, 0),
        arrow_color=(0, 255, 0),
        text_color=(0, 255, 255),
        ignore_verb=False,
        person_color=(0, 0, 255),
        manual_box_color_roi=(0, 0, 0),
        manual_arrow_color=(0, 0, 0),
        manual_text_color=(0, 0, 0),
        manual_person_color=(0, 0, 0),
    )

    print(f"[Viz] saved={saved} missing_imgs={missing_imgs} -> {out_root}")


if __name__ == "__main__":
    main()
