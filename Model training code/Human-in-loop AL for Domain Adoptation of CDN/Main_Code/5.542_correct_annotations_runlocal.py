#!/usr/bin/env python3
"""
Interactive UI to label/clean Phase 2 manual_review.csv rows (or any similar CSV).
Shows person (red) + equipment ROI (green) boxes per frame and records labels.

python Main_Code/5.542_correct_annotations_runlocal.py \
  --csv "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_nms.csv" \
  --frames_dir "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/images" \
  --only-unlabeled \
  --output "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/low_conf_review_bundle/sampled_uncertain_nms_labeled.csv"
  """

import argparse
import ast
from pathlib import Path
from typing import Optional, Sequence

import cv2
import pandas as pd


def parse_box(raw) -> Optional[list[int]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, tuple)):
        vals: Sequence[float] = raw
    else:
        try:
            vals = ast.literal_eval(str(raw))
        except Exception:
            return None
    try:
        return [int(float(v)) for v in vals]
    except Exception:
        return None


def draw_boxes(img, person_box, roi_box, text: str):
    if person_box:
        cv2.rectangle(img, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0, 0, 255), 2)
    if roi_box:
        cv2.rectangle(img, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 0), 2)
    y = 25
    for line in [text, "Enter=accept  0=flip verb(58<->118)  u=undo  d=delete  s=save  q=quit"]:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25
    return img


def review_csv(csv_path: Path, frames_dir: Path, output_override: Optional[Path], only_unlabeled: bool) -> None:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        df["label"] = pd.NA
    if "label_source" not in df.columns:
        df["label_source"] = ""

    subset = df.copy()
    if only_unlabeled and "label" in subset.columns:
        subset = subset[subset["label"].isna()].copy()
    subset["__deleted__"] = False
    subset = subset.reset_index().rename(columns={"index": "__orig_index__"})
    history = []
    out_path = output_override or csv_path.with_name(f"{csv_path.stem}_labeled.csv")

    print(
        f"[{csv_path.name}] Reviewing {len(subset)} rows. Keys: 1/y=interaction, 0/n=no_interaction, u=undo, d=delete, s=save, q/Esc=quit."
    )

    # Build a recursive basename index once so we can resolve frame_file entries
    # that include nested relative paths not rooted exactly at frames_dir.
    name_index: dict[str, list[Path]] = {}
    for p in frames_dir.rglob("*.jpg"):
        name_index.setdefault(p.name, []).append(p)
    for p in frames_dir.rglob("*.jpeg"):
        name_index.setdefault(p.name, []).append(p)
    for p in frames_dir.rglob("*.png"):
        name_index.setdefault(p.name, []).append(p)

    def resolve_image_path(frame_file_val, frame_path_val) -> Optional[Path]:
        candidates: list[Path] = []
        if isinstance(frame_file_val, str) and frame_file_val:
            ff = frame_file_val.replace("\\", "/")
            # 1) exact relative path under frames_dir
            candidates.append(frames_dir / ff)
            # 2) basename under frames_dir
            candidates.append(frames_dir / Path(ff).name)
        if isinstance(frame_path_val, str) and frame_path_val:
            candidates.append(Path(frame_path_val))

        for c in candidates:
            if c.exists():
                return c

        # 3) recursive basename lookup with optional suffix preference
        if isinstance(frame_file_val, str) and frame_file_val:
            ff = frame_file_val.replace("\\", "/")
            base = Path(ff).name
            matches = name_index.get(base, [])
            if not matches:
                return None
            # Prefer path that ends with the same relative suffix when possible.
            suffix = ff.lstrip("/")
            for m in matches:
                if str(m).replace("\\", "/").endswith(suffix):
                    return m
            return matches[0]
        return None

    for i, row in subset.iterrows():
        if subset.at[i, "__deleted__"]:
            continue

        frame_file = row.get("frame_file")
        frame_path = row.get("frame_path")
        img_path = resolve_image_path(frame_file, frame_path)
        if img_path is None:
            print(f"[WARN] missing image for row {i+1}: {frame_file or frame_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] could not read image: {img_path}")
            continue

        pb = parse_box(row.get("person_bbox"))
        rb = parse_box(row.get("roi_bbox"))
        cur_verb = row.get("verb_id")
        try:
            cur_verb = int(cur_verb)
        except Exception:
            cur_verb = None
        if cur_verb == 117:
            cur_verb = 118
        if cur_verb == 57:
            cur_verb = 58
        info = (
            f"{i+1}/{len(subset)}  cid={row.get('candidate_id','')}  eq={row.get('equipment_type','')}  "
            f"obj={row.get('object_id','')}  verb={cur_verb}  src={row.get('label_source','')}"
        )
        vis = draw_boxes(img.copy(), pb, rb, info)
        if pb and rb:
            ps = ((pb[0] + pb[2]) // 2, (pb[1] + pb[3]) // 2)
            rs = ((rb[0] + rb[2]) // 2, (rb[1] + rb[3]) // 2)
            arrow_color = (0, 255, 0) if cur_verb == 118 else (200, 200, 200) if cur_verb == 58 else (128, 128, 128)
            cv2.arrowedLine(vis, ps, rs, arrow_color, 2, tipLength=0.08)

        cv2.imshow("HOI labeler", vis)
        k = cv2.waitKey(0)
        # Enter accepts current predicted verb as GT and moves to next row.
        if k in [13, 10]:
            history.append(("label", i, subset.at[i, "label"], subset.at[i, "__deleted__"], subset.at[i, "verb_id"]))
            v = subset.at[i, "verb_id"]
            try:
                v = int(v)
            except Exception:
                v = 58
            if v == 117:
                v = 118
            if v == 57:
                v = 58
            subset.at[i, "verb_id"] = 118 if v != 58 else 58
            subset.at[i, "label"] = 1 if subset.at[i, "verb_id"] == 118 else 0
            subset.at[i, "__deleted__"] = False
            subset.at[i, "label_source"] = "manual_accept"
        elif k in [ord("1"), ord("y")]:
            history.append(("label", i, subset.at[i, "label"], subset.at[i, "__deleted__"]))
            subset.at[i, "label"] = 1
            subset.at[i, "__deleted__"] = False
            subset.at[i, "verb_id"] = 118
            subset.at[i, "label_source"] = "manual_set_118"
        elif k in [ord("0"), ord("n")]:
            # 0 means prediction is wrong: flip verb 58<->118, then move next.
            history.append(("label", i, subset.at[i, "label"], subset.at[i, "__deleted__"], subset.at[i, "verb_id"]))
            v = subset.at[i, "verb_id"]
            try:
                v = int(v)
            except Exception:
                v = 58
            if v == 117:
                v = 118
            if v == 57:
                v = 58
            subset.at[i, "verb_id"] = 58 if v == 118 else 118
            subset.at[i, "label"] = 1 if subset.at[i, "verb_id"] == 118 else 0
            subset.at[i, "__deleted__"] = False
            subset.at[i, "label_source"] = "manual_flip_58_118"
        elif k in [ord("u")]:
            if history:
                rec = history.pop()
                if len(rec) == 5:
                    _, last_idx, prev_label, prev_deleted, prev_verb = rec
                    subset.at[last_idx, "verb_id"] = prev_verb
                else:
                    _, last_idx, prev_label, prev_deleted = rec
                subset.at[last_idx, "label"] = prev_label
                subset.at[last_idx, "__deleted__"] = prev_deleted
                print(f"[Undo] row {last_idx+1} restored.")
            else:
                print("[Undo] nothing to undo")
            continue
        elif k in [ord("d"), ord("D")]:
            history.append(("delete", i, subset.at[i, "label"], subset.at[i, "__deleted__"]))
            subset.at[i, "__deleted__"] = True
            subset.at[i, "label"] = pd.NA
            print(f"[Delete] marked row {i+1} for removal from saved annotations.")
            continue
        elif k in [ord("s")]:
            print("[Save] saving current progress...")
            drop_cols = [c for c in ["__orig_index__", "__deleted__"] if c in subset.columns]
            reviewed = subset.drop(columns=drop_cols)
            if only_unlabeled:
                to_save = df.copy()
                # Apply reviewed rows back to full dataframe; delete rows marked deleted.
                for _, rr in subset.iterrows():
                    orig_idx = int(rr["__orig_index__"])
                    if rr["__deleted__"]:
                        if orig_idx in to_save.index:
                            to_save = to_save.drop(index=orig_idx)
                        continue
                    if orig_idx in to_save.index:
                        to_save.loc[orig_idx, reviewed.columns] = reviewed.loc[rr.name].values
            else:
                to_save = subset.loc[~subset["__deleted__"]].drop(columns=drop_cols)
            to_save.to_csv(out_path, index=False)
            print(f"Saved -> {out_path}")
        elif k in [ord("q"), 27]:
            print("Exiting by user request.")
            break

    cv2.destroyAllWindows()

    # Final save
    drop_cols = [c for c in ["__orig_index__", "__deleted__"] if c in subset.columns]
    if only_unlabeled:
        final_df = df.copy()
        reviewed = subset.drop(columns=drop_cols)
        for _, rr in subset.iterrows():
            orig_idx = int(rr["__orig_index__"])
            if rr["__deleted__"]:
                if orig_idx in final_df.index:
                    final_df = final_df.drop(index=orig_idx)
                continue
            if orig_idx in final_df.index:
                final_df.loc[orig_idx, reviewed.columns] = reviewed.loc[rr.name].values
    else:
        final_df = subset.loc[~subset["__deleted__"]].drop(columns=drop_cols)
    final_df.to_csv(out_path, index=False)
    alt_path = out_path.with_name(f"{out_path.stem}_copy{out_path.suffix}")
    final_df.to_csv(alt_path, index=False)
    print(f"Wrote labels -> {out_path}")
    print(f"Wrote labels (alt copy) -> {alt_path}")


def main():
    ap = argparse.ArgumentParser(description="Visual UI for correcting Phase 2 manual_review.csv labels.")
    ap.add_argument("--csv", type=Path, help="manual_review.csv (single-video mode).")
    ap.add_argument("--frames_dir", type=Path, help="Frames/visualization directory matching the CSV (single-video mode).")
    ap.add_argument("--phase1_root", type=Path, help="(Optional legacy) root containing per-video 4_* folders with review CSVs.")
    ap.add_argument("--frames_root", type=Path, help="(Optional legacy) root containing per-video frames for multi-video mode.")
    ap.add_argument("--output", type=Path, help="Output CSV (single-video mode only; default: <csv>_labeled.csv next to input).")
    ap.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Review only rows where label is missing (recommended after 5.541 auto-label pass).",
    )
    args = ap.parse_args()

    tasks: list[tuple[Path, Path]] = []

    if args.csv:
        if not args.frames_dir:
            raise SystemExit("--frames_dir is required when --csv is provided")
        tasks.append((args.csv, args.frames_dir))
    elif args.phase1_root:
        if not args.frames_root:
            raise SystemExit("--frames_root is required when using --phase1_root")
        phase1_root = args.phase1_root
        for video_dir in sorted(p for p in phase1_root.glob("4_*") if p.is_dir()):
            # Prefer manual-review CSV; fallback to combined if manual missing.
            manual_csv = video_dir / "manual_review.csv"
            combined_csv = video_dir / "bootstrap_predictions_combined.csv"
            csv_path = manual_csv if manual_csv.exists() else combined_csv if combined_csv.exists() else None
            if csv_path is None:
                continue
            video_name = video_dir.name  # e.g., 4_<video_name>
            candidates = []
            base = video_name[2:] if video_name.startswith("4_") else video_name
            candidates.append(args.frames_root / base)
            candidates.append(args.frames_root / video_name)
            frames_dir = next((p for p in candidates if p.exists()), None)
            if frames_dir is None:
                # fallback to visualization frames if available
                viz = video_dir / "4_viz_labels"
                if viz.exists():
                    frames_dir = viz
                    print(f"[WARN] frames dir not found for {video_name}; falling back to {viz}")
                else:
                    print(f"[WARN] skipping {csv_path}: no matching frames directory under {args.frames_root}")
                    continue
            tasks.append((csv_path, frames_dir))
    else:
        raise SystemExit("Provide either --csv (with --frames_dir) or --phase1_root (with --frames_root).")

    for idx, (csv_path, frames_dir) in enumerate(tasks, start=1):
        print(f"\n=== Task {idx}/{len(tasks)}: {csv_path} ===")
        review_csv(
            csv_path=csv_path,
            frames_dir=frames_dir,
            output_override=args.output if len(tasks) == 1 else None,
            only_unlabeled=args.only_unlabeled,
        )


if __name__ == "__main__":
    main()
