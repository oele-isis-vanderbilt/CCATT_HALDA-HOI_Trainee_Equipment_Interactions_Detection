#!/usr/bin/env python3
"""
Phase 2+ active learning iteration (CDN-only, no detections/IoU).
Build candidates directly from CDN HOI outputs (object_id 81/82/83) and update state_dir.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# -----------------------------
# Utility helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def bbox_to_xyxy(arr) -> List[float]:
    vals = [float(v) for v in arr]
    if len(vals) != 4:
        raise ValueError(f"Expected 4 values for bbox, got {vals}")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:  # maybe xywh
        x2 = x1 + vals[2]
        y2 = y1 + vals[3]
    return [x1, y1, x2, y2]


def parse_bbox_field(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return bbox_to_xyxy(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parts = [float(x) for x in raw.replace("[", "").replace("]", "").split(",") if x.strip()]
                parsed = parts
        return bbox_to_xyxy(parsed)
    return raw


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in {"", "nan", "none"}:
        return True
    return False


OBJECT_ID_MAP = {
    80: "IV Pump",
    81: "propaq",
    82: "MV",
}


def normalize_path(p: Path) -> Path:
    """Strip accidental surrounding quotes and return a clean Path."""
    return Path(str(p).strip("'\""))


# -----------------------------
# CDN helpers
# -----------------------------

class CDNClient:
    def __init__(
        self,
        repo_path: Path,
        pretrained: Path,
        device: str = "cuda",
        extra_args: Optional[str] = None,
        output_name: str = "df_preds.csv",
        num_obj_classes: int = 92,
        num_verb_classes: int = 118,
    ) -> None:
        self.repo_path = repo_path
        self.pretrained = pretrained
        self.device = device
        self.extra_args = extra_args or ""
        self.output_name = output_name
        self.num_obj_classes = num_obj_classes
        self.num_verb_classes = num_verb_classes

    def run_inference(self, image_dir: Path, output_csv: Path) -> Path:
        image_dir = normalize_path(image_dir)
        if not self.repo_path.exists():
            raise SystemExit(f"CDN repo not found at {self.repo_path}")
        nproc = max(1, int(os.environ.get("CDN_NPROC", "1")))
        launcher = ["python"]
        if nproc > 1:
            launcher = [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(nproc),
            ]
        cmd = launcher + [
            "main.py",
            "--pretrained",
            str(self.pretrained),
            "--dataset_file",
            "hico",
            "--hoi_path",
            str(image_dir),
            "--num_obj_classes",
            str(self.num_obj_classes),
            "--num_verb_classes",
            str(self.num_verb_classes),
            "--backbone",
            "resnet50",
            "--num_queries",
            "64",
            "--dec_layers_hopd",
            "3",
            "--dec_layers_interaction",
            "3",
            "--eval",
            "--use_nms_filter",
        ]
        if "--device" not in (self.extra_args or ""):
            cmd.extend(["--device", self.device])
        if self.extra_args:
            cmd.extend(shlex.split(self.extra_args))
        print(f"[CDN] running inference: {' '.join(cmd)} (cwd={self.repo_path})")
        cdn_silent = os.environ.get("CDN_SILENT", "1").strip().lower() in {"1", "true", "yes", "on"}
        if cdn_silent:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_path,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                err_tail = (proc.stderr or "").splitlines()[-40:]
                raise subprocess.CalledProcessError(
                    proc.returncode,
                    cmd,
                    output=None,
                    stderr="\n".join(err_tail),
                )
        else:
            subprocess.run(cmd, cwd=self.repo_path, check=True)
        produced = self.repo_path / self.output_name
        if not produced.exists():
            raise SystemExit(f"Expected CDN output missing: {produced}")
        ensure_dir(output_csv.parent)
        shutil.copy2(produced, output_csv)
        print(f"[CDN] copied predictions -> {output_csv}")
        return output_csv


def standardize_cdn_predictions(df: pd.DataFrame) -> pd.DataFrame:
    def pick(columns: List[str], heuristics: List[str] | None = None) -> str:
        for c in columns:
            if c in df.columns:
                return c
        if heuristics:
            cols_lower = {c.lower(): c for c in df.columns}
            for h in heuristics:
                for lower, original in cols_lower.items():
                    if h in lower:
                        return original
        raise ValueError(f"Missing required columns; checked {columns} with heuristics {heuristics}")

    img_col = pick(["file_name", "image_id", "image", "img", "file"], heuristics=["file", "image", "img"])
    subj_col = pick(["person_box", "subject_box", "sub_box", "bbox_subject", "human_box"], heuristics=["subject", "person", "human"])
    obj_col = pick(["object_box", "obj_box", "bbox_object"], heuristics=["object", "roi", "box2"])
    verb_col = pick(["verb_id", "verb", "category_id", "verb_class"], heuristics=["verb", "class", "label"])
    score_col = pick(["score", "hoi_score", "prob"], heuristics=["score", "prob"])
    obj_id_col = pick(["object_id", "obj_id", "obj", "object_class", "obj_class"], heuristics=["object_id", "obj_id", "object", "class"])
    obj_score_col = pick(["obj_scores", "object_score", "object_prob"], heuristics=["obj", "object", "score"])
    verb_score_col = pick(["verb_scores_index_decoder", "verb_score", "verb_prob"], heuristics=["verb", "score", "prob"])

    img_idx = df.columns.get_loc(img_col)
    subj_idx = df.columns.get_loc(subj_col)
    obj_idx = df.columns.get_loc(obj_col)
    verb_idx = df.columns.get_loc(verb_col)
    score_idx = df.columns.get_loc(score_col)
    obj_id_idx = df.columns.get_loc(obj_id_col)
    obj_score_idx = df.columns.get_loc(obj_score_col)
    verb_score_idx = df.columns.get_loc(verb_score_col)

    records = []
    for row in df.itertuples(index=False, name=None):
        img_val = row[img_idx]
        score_val = row[score_idx]
        verb_val = row[verb_idx]
        obj_id_val = row[obj_id_idx]
        obj_score_val = row[obj_score_idx]
        verb_score_val = row[verb_score_idx]
        if _is_missing(img_val) or _is_missing(score_val) or _is_missing(obj_id_val) or _is_missing(verb_val):
            continue
        try:
            subj_box = parse_bbox_field(row[subj_idx])
            obj_box = parse_bbox_field(row[obj_idx])
            if subj_box is None or obj_box is None:
                continue
            obj_id = int(obj_id_val)
            # obj_score may come as a scalar or a JSON/array-like string; strip brackets and take the first value.
            if isinstance(obj_score_val, str):
                obj_score_val = obj_score_val.replace("[", "").replace("]", "").split(",")[0]
            obj_score = float(obj_score_val)
            if obj_score < 0.5:
                continue
            if isinstance(verb_score_val, str):
                verb_score_val = verb_score_val.replace("[", "").replace("]", "").split(",")[0]
            verb_score = float(verb_score_val)
            # Keep the verb id from the model output; down-stream thresholds handle auto/manual/ignore.
            verb_id = int(verb_val)
            score = float(score_val)
        except Exception:
            continue
        records.append(
            {
                "frame_file": Path(str(img_val)).name,
                "frame_stem": Path(str(img_val)).stem,
                "subject_box": subj_box,
                "object_box": obj_box,
                "verb_id": verb_id,
                "object_id": obj_id,
                "score": score,
                "object_score": obj_score,
                "verb_score": verb_score,
            }
        )
    return pd.DataFrame.from_records(records)


def build_candidates_from_cdn(cdn_preds: pd.DataFrame, frames_dir: Path) -> pd.DataFrame:
    rows = []
    counter: dict[tuple[str, str], int] = {}
    for _, pred in cdn_preds.iterrows():
        obj_id = int(pred["object_id"])
        if obj_id not in OBJECT_ID_MAP:
            continue
        equipment = OBJECT_ID_MAP[obj_id]
        key = (pred["frame_stem"], equipment)
        idx = counter.get(key, 0)
        counter[key] = idx + 1
        cand_id = f"{pred['frame_stem']}_{equipment.replace(' ', '_')}_{idx}"
        frame_file = Path(pred["frame_file"]).name
        rows.append(
            {
                "candidate_id": cand_id,
                "video": frames_dir.name,
                "frame_file": frame_file,
                "frame_path": str(frames_dir / frame_file),
                "equipment_type": equipment,
                "person_bbox": pred["subject_box"],
                "roi_bbox": pred["object_box"],
                "person_score": pred["score"],
                "object_score": pred.get("object_score"),
                "verb_score": pred.get("verb_score"),
                "verb_id": pred["verb_id"],
                "object_id": obj_id,
                "cdn_score": pred["score"],
            }
        )
    return pd.DataFrame(rows)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def hoi_nms_pairs(cands: pd.DataFrame, person_iou: float = 0.4, roi_iou: float = 0.5) -> pd.DataFrame:
    """
    Deduplicate HOI candidates per frame/equipment/object. object_id is already unique per
    HOI, so we suppress overlaps on the person box only (person-only NMS). Keeps the
    highest scoring verb per pair (prefers higher verb_score, then cdn_score).
    """
    if cands.empty:
        return cands
    keep_rows = []
    grouped = cands.groupby(["frame_file", "equipment_type", "object_id"], dropna=False)
    for _, group in grouped:
        rows = group.to_dict(orient="records")
        rows_sorted = sorted(
            rows,
            key=lambda r: float(r.get("verb_score") or r.get("cdn_score") or r.get("object_score") or 0.0),
            reverse=True,
        )
        kept = []
        for r in rows_sorted:
            pb = r.get("person_bbox")
            if pb is None:
                kept.append(r)
                continue
            dup = False
            for k in kept:
                kb = k.get("person_bbox")
                if kb is None:
                    continue
                if _iou(pb, kb) >= person_iou:
                    dup = True
                    break
            if not dup:
                kept.append(r)
        keep_rows.extend(kept)
    return pd.DataFrame(keep_rows)


def apply_external_nms(cdn_csv: Path, roi_iou: float = 0.0, output_path: Path | None = None) -> Path:
    """
    Run person-only NMS via 5.5_remove_hoi_duplicates.py and return the output CSV path.
    """
    script_path = Path(__file__).resolve().parent / "5.5_remove_hoi_duplicates.py"
    output_csv = output_path or cdn_csv.with_name(f"{cdn_csv.stem}_nms.csv")
    cmd = [
        sys.executable,
        str(script_path),
        "-i",
        str(cdn_csv),
        "-o",
        str(output_csv),
        "--roi-iou",
        str(roi_iou),
    ]
    print(f"[Active] Running external NMS: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_csv


def attach_cdn_labels(
    candidates: pd.DataFrame,
    positive_verb_ids: list[int],
    negative_verb_ids: list[int],
) -> pd.DataFrame:
    AUTO_VERB = 0.75
    MANUAL_MIN = 0.6  # (MANUAL_MIN, AUTO_VERB) -> manual review bucket for verb_id 117

    labeled_rows = []
    for _, row in candidates.iterrows():
        raw_vid = int(row["verb_id"])
        score = float(row.get("verb_score") or 0.0) #or row.get("cdn_score") 

        label = None
        pos_score = None
        neg_score = None
        label_score = None
        label_verb = None
        manual_flag = False
        ignored_low_score = False
        mapped_vid = raw_vid

        if raw_vid == 117 and score >= AUTO_VERB:
            label = 1
            pos_score = score
            label_score = score
            label_verb = 118
            mapped_vid = 118
        elif raw_vid == 57 and score >= AUTO_VERB:
            label = 0
            neg_score = score
            label_score = score
            mapped_vid = 58
            label_verb = 58
        
        elif MANUAL_MIN < score < AUTO_VERB and raw_vid == 117:
            # Manual review bucket: keep as manual but remap to 118 for consistency
            manual_flag = True
            mapped_vid = 118

        elif score < MANUAL_MIN:
            # Very low confidence -> treat as negative (58)
            label = 0
            neg_score = score
            label_score = score
            label_verb = 58
            mapped_vid = 58
        else:
            ignored_low_score = True

        row_dict = row.to_dict()
        row_dict["verb_id"] = mapped_vid
        row_dict.update(
            {
                "cdn_label": label,
                "cdn_pos_score": pos_score,
                "cdn_neg_score": neg_score,
                "cdn_label_score": label_score,
                "cdn_label_verb": label_verb,
                "cdn_matched": 1,
                "manual_flag": manual_flag,
                "ignored_low_score": ignored_low_score,
            }
        )
        labeled_rows.append(row_dict)

    return pd.DataFrame(labeled_rows)


# -----------------------------
# Active learning core
# -----------------------------

def merge_manual_labels(candidates_df: pd.DataFrame, manual_labels_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    manual_df = pd.read_csv(manual_labels_path)
    if "candidate_id" not in manual_df.columns or "label" not in manual_df.columns:
        raise SystemExit("Manual labels file must contain candidate_id and label columns.")
    merged = candidates_df.merge(manual_df[["candidate_id", "label"]], on="candidate_id", how="left", suffixes=("", "_manual"))
    newly_labeled = merged[merged["label_manual"].notna()].copy()
    if newly_labeled.empty:
        return pd.DataFrame(), candidates_df
    newly_labeled["final_label"] = newly_labeled["label_manual"].astype(int)
    remaining = merged[merged["label_manual"].isna()].drop(columns=["label_manual"])
    newly_labeled = newly_labeled.drop(columns=["label_manual"])
    return newly_labeled, remaining


def compute_uncertainty(row: pd.Series) -> float:
    pos = row.get("cdn_pos_score") or 0.0
    neg = row.get("cdn_neg_score") or 0.0
    best = max(pos, neg)
    return 1.0 - best


def run_active_iteration(
    state_dir: Path,
    frames_dir: Path,
    cdn_repo: Path,
    cdn_weights: Path,
    cdn_extra_args: str,
    positive_verb_ids: List[int],
    negative_verb_ids: List[int],
    budget_ratio: float,
    iteration: int,
    manual_labels: Optional[Path],
) -> dict:
    state_dir = normalize_path(state_dir)
    frames_dir = normalize_path(frames_dir)
    ensure_dir(state_dir)
    video_out_dir = state_dir  # outputs stay within the provided state_dir (one per video)
    iter_dir = video_out_dir / f"iteration_{iteration:03d}"
    ensure_dir(iter_dir)

    labeled_path = state_dir / "labeled.csv"
    labeled_df = pd.read_csv(labeled_path) if labeled_path.exists() else pd.DataFrame()

    client = CDNClient(cdn_repo, cdn_weights, extra_args=cdn_extra_args)
    preds_csv = iter_dir / "cdn_preds_raw.csv"
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")
    client.run_inference(frames_dir, preds_csv)
    cdn_preds_raw = pd.read_csv(preds_csv, dtype=str)
    cdn_preds = standardize_cdn_predictions(cdn_preds_raw)
    if cdn_preds.empty:
        raise SystemExit(f"No valid CDN predictions after standardization; raw columns={list(cdn_preds_raw.columns)}")
    # Allow auto labels (117/58) plus manual-review verb (117); ignore the rest.
    allowed_verbs = set(positive_verb_ids + negative_verb_ids)
    cdn_preds = cdn_preds[cdn_preds["verb_id"].isin(allowed_verbs)]
    cdn_preds = cdn_preds[cdn_preds["object_id"].isin(OBJECT_ID_MAP.keys())].reset_index(drop=True)
    cdn_preds.to_csv(iter_dir / "cdn_preds_standardized.csv", index=False)

    cand_df = build_candidates_from_cdn(cdn_preds, frames_dir)
    cand_df = hoi_nms_pairs(cand_df, person_iou=0.4, roi_iou=0.5)
    cdn_preds_path = iter_dir / "cdn_preds.csv"
    save_dataframe(cand_df, cdn_preds_path)

    # Attach labels/mapping before external NMS so dedup runs on filtered/mapped verbs.
    cand_df = attach_cdn_labels(cand_df, positive_verb_ids, negative_verb_ids)
    labeled_preds_path = iter_dir / "cdn_preds_labeled.csv"
    save_dataframe(cand_df, labeled_preds_path)

    # Drop ignored rows before NMS.
    if "ignored_low_score" in cand_df.columns:
        cand_df_for_nms = cand_df[~cand_df["ignored_low_score"]].copy()
    else:
        cand_df_for_nms = cand_df.copy()
    nms_input = iter_dir / "cdn_preds_for_nms.csv"
    save_dataframe(cand_df_for_nms, nms_input)

    # Person-only NMS using 5.5_remove_hoi_duplicates.py so downstream auto/manual labels are deduped.
    nms_csv = apply_external_nms(nms_input, roi_iou=0.0, output_path=iter_dir / "cdn_preds_nms.csv")
    cand_df = pd.read_csv(nms_csv)
    for bool_col in ("manual_flag", "ignored_low_score"):
        if bool_col in cand_df.columns:
            cand_df[bool_col] = (
                cand_df[bool_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
            )
    if cand_df.empty:
        raise SystemExit("No CDN candidates found for object ids 81/82/83.")

    # Drop already labeled candidates.
    if not labeled_df.empty and "candidate_id" in labeled_df.columns:
        cand_df = cand_df[~cand_df["candidate_id"].isin(labeled_df["candidate_id"])]

    # Remove ignored low-score rows from the pool.
    remaining_df = cand_df[~cand_df.get("ignored_low_score", False)].copy()

    # Auto-label from CDN predictions (verb_score >= 0.1 already mapped in standardization).
    auto_mask = remaining_df["cdn_label"].notna()
    auto_df = remaining_df[auto_mask].copy()
    remaining_df = remaining_df[~auto_mask].copy()
    auto_df["final_label"] = auto_df["cdn_label"]
    auto_df["label_source"] = "cdn_confident"

    # Manual review for mid-confidence (<0.1 verb_score).
    manual_df = remaining_df[remaining_df.get("manual_flag", False)].copy()
    remaining_df = remaining_df[~remaining_df.get("manual_flag", False)].copy()

    # Update state.
    labeled_out = pd.concat([labeled_df, auto_df], ignore_index=True)
    save_dataframe(labeled_out, labeled_path)
    save_dataframe(auto_df, iter_dir / "auto_labels.csv")
    save_dataframe(manual_df, iter_dir / "manual_review.csv")

    summary = {
        "iteration": iteration,
        "auto_labeled": len(auto_df),
        "manual_requested": len(manual_df),
        "unlabeled_remaining": 0,
    }
    save_json(summary, iter_dir / "summary.json")
    print(f"[Active] iter={iteration} auto={summary['auto_labeled']} manual={summary['manual_requested']} remaining={summary['unlabeled_remaining']}")
    return summary

def main() -> None:
    ap = argparse.ArgumentParser(description="Run one active-learning iteration with CDN-only (no detections/IoU).")
    ap.add_argument("--state_dir", required=True, type=Path, help="State dir with labeled.csv and unlabeled.csv.")
    ap.add_argument("--frames_dir", required=True, type=Path, help="Frames directory for this video.")
    ap.add_argument("--cdn_repo", required=True, type=Path, help="Path to CDN repository.")
    ap.add_argument(
        "--cdn_weights",
        type=Path,
        default=Path("/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase1_Training/AU_run1/logs/checkpoint_best.pth"),
        help="Path to CDN weights (fine-tuned per round).",
    )
    ap.add_argument("--cdn_extra_args", default="", help="Extra args passed to CDN main.py.")
    ap.add_argument(
        "--positive_verb_ids",
        type=int,
        nargs="+",
        required=True,
        help="Verb IDs corresponding to interaction_yes (e.g., hold, carry)",
    )
    ap.add_argument(
        "--negative_verb_ids",
        type=int,
        nargs="+",
        required=True,
        help="Verb IDs corresponding to interaction_no (e.g., watch, no_interaction)",
    )
    ap.add_argument("--budget_ratio", type=float, default=0.2, help="Fraction of uncertain samples to send to manual labels.")
    ap.add_argument("--iteration", type=int, required=True, help="Iteration index (increment each round).")
    ap.add_argument("--manual_labels", type=Path, help="CSV with manual labels from prior iteration (candidate_id,label).")
    args = ap.parse_args()

    run_active_iteration(
        state_dir=args.state_dir,
        frames_dir=args.frames_dir,
        cdn_repo=args.cdn_repo,
        cdn_weights=args.cdn_weights,
        cdn_extra_args=args.cdn_extra_args,
        positive_verb_ids=args.positive_verb_ids,
        negative_verb_ids=args.negative_verb_ids,
        budget_ratio=args.budget_ratio,
        iteration=args.iteration,
        manual_labels=args.manual_labels,
    )

if __name__ == "__main__":
    main()
