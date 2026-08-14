#!/usr/bin/env python3
"""
Step 0 of the CCATT inference pipeline: turn frame-level HOI confidence scores
into temporal trainee-equipment interaction intervals (start/end times).

What this script does:
  Reads a per-video *_df_preds.csv of frame-level object/verb confidence
  scores (produced upstream by the HOI detection model), smooths each
  object's confidence curve with a Gaussian filter, thresholds it, and
  groups consecutive high-confidence frames into interaction segments with
  a start time and end time. Per-object thresholds and smoothing sigmas
  default to the calibrated values in `thresholds` / `sigmas` below (these
  match the paper's post-processing hyperparameter table) -- override them
  only if you are recalibrating for a new object/camera setup.

  This is the first stage in the 3-step pipeline described in this repo's
  README.md. Its output (combined_segments.csv-style files, one row per
  interaction interval) is the --input_root consumed by
  generate_role_assignment_csvs.py and person_identification_v4.py.

Two ways to run it:
  1. Single-view:  --pred_csv <one *_df_preds.csv>
  2. Multi-view / per-simulation (recommended): --pred_csvs <multiple
     *_df_preds.csv, one per camera view> together with matching
     --video_paths (same order). Optionally add --gt_csv/--gt_sheet_tokens
     to also score predictions against expert ground-truth annotations.

Advanced / research mode:
  --grid_search sweeps threshold/sigma combinations against --gt_csv to
  find better per-object hyperparameters. Most users adapting this to a new
  camera/object set up will not need this -- the shipped defaults already
  match the calibration described in the paper. See README.md for details.

Use --dry_run to check that all input files exist and are readable without
running the (potentially slow, GPU/CPU-heavy) interval-generation itself.

See README.md in this repository for full example commands.
"""

import os, re, json, argparse, itertools, shutil, subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt
import io
import cv2
import concurrent.futures as cf
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Map GT Interaction_Type labels to compact object_class IDs
MAP_INTERACTION_TO_OBJ = {
    # Dave
    "DAVE_IV1": 80,
    "DAVE_IV2": 81,
    "DAVE_IV1_ALT": 82,   # map to “Model2” if GT names differ
    "DAVE_IV2_ALT": 83,
    "DAVE_MV": 84,
    "DAVE_PROPAQ": 85,
    # Chuck
    "CHUCK_IV1": 86,
    "CHUCK_IV2": 87,
    "CHUCK_IV1_ALT": 88,
    "CHUCK_IV2_ALT": 89,
    "CHUCK_MV": 90,
    "CHUCK_PROPAQ": 91,
}

# Invert to recover instance-specific names (e.g., DAVE_IV1) from object IDs
OBJID_TO_INSTANCE_NAME = {int(v): str(k) for k, v in MAP_INTERACTION_TO_OBJ.items()}

#
# Default per-object HOI thresholds (keyed by object_class IDs 80–91).
# These are applied inside predicted_intervals_only via get_obj_thresh():
#   thresh_obj = get_obj_thresh(oid_int, obj_thresh_map, DEFAULT_OBJ_THRESH, global_default_thresh)
# If an object_class is not in this dict, the global default (or CLI override) is used instead.
# DEFAULT_OBJ_THRESH = {
#     80: 0.5,   # Dave version1 IV1
#     81: 0.5, #35,   # Dave version1 IV2
#     82: 0.6,  # Dave version3 IV1
#     83: 0.5, #0.35,   # Dave version3 IV2
#     84: 0.1,  # Dave MV
#     85: 0.2,  # Dave Propaq
#     86: 0.5,   # Chuck version1 IV1
#     87: 0.6,  # Chuck version1 IV2
#     88: 0.2,  # Chuck version3 IV1
#     89: 0.5,   # Chuck version3 IV2
#     90: 0.25,  # Chuck MV
#     91: 0.2,  # Chuck Propaq
# } 

# # Default per-object Gaussian smoothing sigmas (in frames).
# DEFAULT_OBJ_SIGMA = {
#     80: 30, # Dave version1 IV1
#     81: 30, # Dave version1 IV2
#     82: 30, # Dave version3 IV1
#     83: 30, # Dave version3 IV2
#     84: 30, # Dave MV
#     85: 30, # Dave Propaq
#     86: 30, # Chuck version1 IV1
#     87: 30, # Chuck version1 IV2
#     88: 20, # Chuck version3 IV1
#     89: 20, # Chuck version3 IV2
#     90: 30, # Chuck MV
#     91: 30, # Chuck Propaq
# }

thresholds = {
    80: 0.50,
    81: 0.80,
    82: 0.60,
    83: 0.99,
    84: 0.20,
    85: 0.30,
    86: 0.50,
    87: 0.50,
    88: 0.40,
    89: 0.50,
    90: 0.40,
    91: 0.30,
}

MAX_PERSON_BOXES_PER_INTERVAL = 5

sigmas = {
    80: 30,
    81: 30,
    82: 30,
    83: 30,
    84: 10,
    85: 5,
    86: 30,
    87: 30,
    88: 10,
    89: 20,
    90: 10,
    91: 5,
}

# Backward-compatible aliases used throughout the script.
# Canonical maps are `thresholds` and `sigmas`.
DEFAULT_OBJ_THRESH = thresholds
DEFAULT_OBJ_SIGMA = sigmas

# Define static objects in xyxy → will be converted to xywh
camera_view_boxes = {
    "view1_v1": {
        80: [828, 183, 868, 278],   # dave_v1_Iv1 (was 91)
        81: [753, 288, 799, 453],   # dave_v1_Iv2 (was 92)
        84: [792, 328, 875, 426],   # dave_MV (was 95)
        86: [1210, 211, 1261, 298], # chuck_IV1 (was 97)
        87: [1235, 283, 1274, 393], # chuck_IV2 (was 98)
        91: [1170, 284, 1243, 377], # propaq (was 102)
    },
    "view1_v2": {
        82: [785, 359, 873, 450],   # dave v2 Iv1 (was 93)
        83: [723, 299, 789, 454],   # dave v2 Iv2 (was 94)
        84: [880, 335, 963, 441],   # dave_MV (was 95)
        89: [1231, 274, 1289, 372], # chuck_IV2 v2 (was 100)
        91: [1173, 266, 1240, 383], # propaq (was 102)
    },
    "view2_v1": {
        80: [929, 253, 980, 341],   # dave_v1_Iv1 (was 91)
        81: [1006, 317, 1064, 403], # dave_v1_Iv2 (was 92)
        85: [932, 331, 1020, 426],  # Dave propaq (was 96)
        86: [550, 355, 613, 454],   # chuck_IV1 (was 97)
        87: [505, 482, 548, 581],   # chuck_IV2 (was 98)
        90: [514, 519, 627, 621],   # Ventilator (was 101)
    },
    "view2_v2": {
        83: [972, 261, 1038, 351],  # dave v2 Iv2 (was 94)
        85: [930, 280, 1013, 395],  # Dave propaq (was 96)
        88: [516, 511, 620, 593],   # chuck_IV1 v2 (was 99)
        90: [622, 453, 735, 558],   # Ventilator (was 101)
    },
}

def get_camera_view(fname: str):
    fname = (fname or "").lower()
    if "old_cam" in fname:
        return "view2_v1"
    elif "new_cam" in fname:
        return "view2_v2"
    elif "old_pan" in fname:
        return "view1_v1"
    elif "new_pan" in fname:
        return "view1_v2"
    # Additional heuristics for names like "cam16_v2", "pan_v1", "pan2", etc.
    if "cam" in fname:
        if "v2" in fname or "_2" in fname:
            return "view2_v2"
        if "v1" in fname or "_1" in fname:
            return "view2_v1"
    if "pan" in fname:
        if "v2" in fname or "_2" in fname:
            return "view1_v2"
        if "v1" in fname or "_1" in fname:
            return "view1_v1"
    return None


# Invert map to get names back from object ids (for nicer filenames).
# We intentionally collapse all IV-related object ids on each side into a single
# semantic label (e.g., CHUCK_IV, DAVE_IV) because annotations do not reliably
# distinguish between multiple IV pumps (IV1 vs IV2, ALT variants).
OBJ_TO_INTERACTION = {
    80: "DAVE_IV",
    81: "DAVE_IV",
    82: "DAVE_IV",
    83: "DAVE_IV",
    84: "DAVE_MV",
    85: "DAVE_PROPAQ",
    86: "CHUCK_IV",
    87: "CHUCK_IV",
    88: "CHUCK_IV",
    89: "CHUCK_IV",
    90: "CHUCK_MV",
    91: "CHUCK_PROPAQ",
}
# Group related objects for F1 aggregation (IV family per side)
OBJ_GROUP = {
    80: "DAVE_IV", 81: "DAVE_IV", 82: "DAVE_IV", 83: "DAVE_IV",
    84: "DAVE_MV", 85: "DAVE_PROPAQ",
    86: "CHUCK_IV", 87: "CHUCK_IV", 88: "CHUCK_IV", 89: "CHUCK_IV",
    90: "CHUCK_MV", 91: "CHUCK_PROPAQ",
}

def oid_to_label(oid):
    """Return a human-readable label for an object or group key."""
    try:
        oid_int = int(oid)
    except Exception:
        oid_int = None
    if oid_int is not None:
        if oid_int in OBJ_TO_INTERACTION:
            return OBJ_TO_INTERACTION[oid_int]
        if oid_int in OBJ_GROUP:
            return OBJ_GROUP[oid_int]
    # Fall back to string form
    return str(oid)

# ---------- helpers ----------
def parse_frames(series):
    """
    Extract frame index from filenames like '000123.jpg' or paths ending in that.
    We intentionally take the *last* integer in the string (closest to the extension)
    to avoid picking dates like 2024, 09, etc. from folder names.
    """
    s = series.astype(str)
    # Extract the last run of digits in each string
    f = s.str.extract(r'(\d+)(?!.*\d)')[0]
    return f.dropna().astype(int)

def infer_fps(df, fallback=4.0):
    # 1) explicit columns
    for c in ["fps", "video_fps", "extraction_fps"]:
        if c in df.columns:
            try:
                v = float(df[c].iloc[0])
            except Exception:
                continue
            if v > 0:
                return v
    # 2) parse from filename/path tokens like "...fps4..." or "..._FPS_5..."
    s = " ".join(map(str, df.get("filename", pd.Series([], dtype=str)).head(50).tolist()))
    m = re.search(r'fps[_\-]?(\d+(?:\.\d+)?)', s, flags=re.I)
    if m: 
        v = float(m.group(1))
        if v > 0: return v
    # 3) if duration info present
    if "video_duration_sec" in df.columns:
        try:
            dur = float(df["video_duration_sec"].iloc[0])
            if dur > 0:
                frames = parse_frames(df["filename"])
                if len(frames): return max(frames) / dur
        except: pass
    return float(fallback)

def frame_to_mmss(fid, fps):
    t = int(round(fid / fps))
    return f"{t//60:02}:{t%60:02}"

def annotate_interval_time_labels(ax, intervals, fps, *, y_frac=0.9, color="black", fontsize=7, rotation=90):
    """
    Annotate interval start/end timestamps directly on an axis.
    x is in data coordinates (frame ids), y is in axis-fraction coordinates.
    """
    for start_f, end_f in (intervals or []):
        mid = 0.5 * (float(start_f) + float(end_f))
        txt = f"{frame_to_mmss(int(start_f), fps)}-{frame_to_mmss(int(end_f), fps)}"
        ax.text(
            mid,
            float(y_frac),
            txt,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            alpha=0.9,
            rotation=rotation,
            clip_on=True,
        )

def annotate_gt_status(ax, gt_intervals, *, y_frac=0.96):
    """
    Ensure every plot explicitly communicates GT presence.
    """
    if gt_intervals:
        return
    ax.text(
        0.01, float(y_frac), "GT unavailable for this plot",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="red", alpha=0.85
    )

def hms_to_seconds(hms_str):
    """Convert time-like strings ('HH:MM:SS', 'MM:SS', 'HH:MM.SS', etc.) to seconds. Returns None if invalid."""
    try:
        if pd.isna(hms_str):
            return None
    except Exception:
        pass

    # Numeric values are treated as seconds already
    if isinstance(hms_str, (int, float, np.integer, np.floating)):
        if np.isnan(hms_str):
            return None
        return int(round(float(hms_str)))

    s = str(hms_str).strip()
    if not s or s in {".", "..", "-", "nan", "NaN", "None"}:
        return None

    # Normalize common Excel-style "HH:MM.SS" or "MM.SS" by replacing the last '.' with ':'
    if "." in s and s.count(":") <= 1:
        # replace only the last '.' to preserve decimals if present
        parts = s.rsplit(".", 1)
        if len(parts) == 2 and parts[1].strip().isdigit():
            s = ":".join(parts)

    # Try pandas timedelta parser for flexible formats (HH:MM:SS, MM:SS, HH:MM:SS.mmm, HH:MM.SS, etc.)
    try:
        td = pd.to_timedelta(s)
        return int(round(td.total_seconds()))
    except Exception:
        return None

def extract_obj_score(x):
    if isinstance(x, (list, tuple)): return float(x[0])
    if isinstance(x, str):
        x = x.strip().strip('[]')
        try: return float(x.split(',')[0])
        except: return np.nan
    try: return float(x)
    except: return np.nan

def contiguous_runs(idxs):
    if len(idxs) == 0: return []
    splits = np.where(np.diff(idxs) != 1)[0] + 1
    chunks = np.split(idxs, splits)
    return [(c[0], c[-1]) for c in chunks]

def parse_box_xywh(val, fmt="auto"):
    """
    Parse a box from string/list into [x,y,w,h] (floats).
    `fmt` can be:
      - "xyxy": treat input as [x1, y1, x2, y2]
      - "xywh": treat input as [x, y, w, h]
      - "auto": heuristic fallback (existing behavior)
    """
    # normalize to list of 4 floats
    if isinstance(val, (list, tuple, np.ndarray)):
        b = list(val)
    else:
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        if len(parts) < 4:
            return None
        try:
            b = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
        except Exception:
            return None
    if len(b) != 4:
        return None
    x0, y0, x2_or_w, y2_or_h = b
    fmt = (fmt or "auto").lower()

    if fmt == "xyxy":
        w = max(0.0, x2_or_w - x0)
        h = max(0.0, y2_or_h - y0)
        return [x0, y0, w, h]
    if fmt == "xywh":
        return [x0, y0, max(0.0, x2_or_w), max(0.0, y2_or_h)]

    # auto: Heuristic – if the 3rd/4th numbers are larger than the first two, treat as x2,y2
    if x2_or_w > x0 and y2_or_h > y0:
        w = x2_or_w - x0
        h = y2_or_h - y0
        return [x0, y0, w, h]
    # otherwise assume already xywh
    return [x0, y0, x2_or_w, y2_or_h]


# Helper: build per-frame box map for given column or fallback columns
def build_frame_to_box_map(df_like, primary_col, fallback_cols=(), box_format="auto", scale=None):
    """Return dict: frame_id -> box(x,y,w,h) from `primary_col` or first available in `fallback_cols`.
    If `scale` is (sx, sy), multiply coordinates and sizes by those factors."""
    cols = [c for c in (primary_col, *fallback_cols) if c in df_like.columns]
    if not cols:
        return {}
    col = cols[0]
    out = {}
    for fid, boxval in df_like.reset_index()[["frame_id", col]].itertuples(index=False):
        box = parse_box_xywh(boxval, fmt=box_format)
        if box is not None and scale:
            sx, sy = scale
            box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
        out[int(fid)] = box
    return out

def draw_box(frame, box_xywh, color, thickness=2, label=None):
    """
    Draw XYWH box on BGR frame. Clips to frame size.
    """
    if box_xywh is None: 
        return
    h, w = frame.shape[:2]
    x, y, bw, bh = box_xywh
    x1, y1 = int(max(0, round(x))), int(max(0, round(y)))
    x2, y2 = int(min(w-1, round(x + bw))), int(min(h-1, round(y + bh)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, str(label), (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ---------- Helper: Render HOI plot as image for video overlay ----------
def render_hoi_plot_image(processed_entry, fps, thresh, width, current_frame=None, gt_intervals=None):
    """
    Render the smoothed HOI plot (with threshold and intervals) as an image for overlay in video.
    Returns: BGR uint8 image (numpy array), resized to width and aspect ratio preserved.
    If current_frame is not None, plot the full HOI curve in very light gray (#DDDDDD) and
    overlay the curve up to current_frame in blue ("HOI (smoothed)").
    Optionally overlays GT intervals as red semi-transparent spans if gt_intervals is provided.
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import io

    xs = processed_entry["frames"]
    hoi = processed_entry["hoi"]
    merged = processed_entry["merged"]
    # Compose the plot
    fig, ax = plt.subplots(figsize=(12, 2.5))  # Wide and short
    xs_arr = np.array(xs)
    hoi_arr = np.array(hoi)
    # Determine what to plot based on current_frame
    if current_frame is None or (isinstance(current_frame, int) and current_frame < 0):
        # Do not plot any HOI curve
        pass
    else:
        # Only plot up to current_frame in blue with label
        mask = xs_arr <= current_frame
        if np.any(mask):
            ax.plot(xs_arr[mask], hoi_arr[mask], color="blue", label="HOI (smoothed)")
        # If no frames <= current_frame, plot nothing
    ax.axhline(thresh, linestyle="--", linewidth=1, label=f"Th={thresh}")
    # Draw GT intervals if provided
    if gt_intervals is not None:
        for i, (gs, ge) in enumerate(gt_intervals):
            ax.axvspan(gs, ge, color="red", alpha=0.15, label="GT" if i == 0 else None)
        annotate_interval_time_labels(ax, gt_intervals, fps, y_frac=0.88, color="red")
        annotate_gt_status(ax, gt_intervals)
    for i, (s, e) in enumerate(merged):
        ax.axvspan(s, e, color="green", alpha=0.12, label="Predicted interval" if i == 0 else None)
    annotate_interval_time_labels(ax, merged, fps, y_frac=0.72, color="green")
    ax.set_title("Smoothed HOI Confidence with Predicted Intervals", fontsize=12)
    ax.set_xlabel("Time (mm:ss)")
    ax.set_yticks([])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    # x-axis ticks in mm:ss
    if len(xs) > 0:
        ticks = np.linspace(0, len(xs)-1, min(10, len(xs)), dtype=int)
        ax.set_xticks(xs_arr[ticks])
        ax.set_xticklabels([frame_to_mmss(f, fps) for f in xs_arr[ticks]], rotation=45)
    ax.legend(fontsize=9)
    plt.tight_layout()
    # Save to in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    # Read image from buffer as numpy array (OpenCV)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    # Resize to width, preserve aspect ratio
    h0, w0 = img.shape[:2]
    if w0 != width:
        scale = width / float(w0)
        new_h = int(round(h0 * scale))
        img = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    # Ensure BGR for OpenCV
    return img

def compute_frame_f1(processed, gt_by_label, obj_to_interaction, cam_prefix=None, available_ids=None, start_frame=None):
    """
    Compute per-object frame-level F1 scores, comparing prediction intervals to GT intervals.
    Returns:
      - dict: oid -> {"pos": F1_pos or None, "weighted": weighted_F1 or None}
        where weighted includes negative class (no_interaction).
      - macro_F1_pos: average over objects where F1_pos is defined (0 if no GT/pred).
      - macro_F1_weighted: average of weighted F1 over objects where defined.
    Optionally filters GT intervals by camera prefix.
    """
    # 1. Determine global max frame id
    max_frame = 0
    # From preds
    for data in processed.values():
        if "frames" in data and len(data["frames"]) > 0:
            maxf = int(np.max(data["frames"]))
            if maxf > max_frame:
                max_frame = maxf
        # Also check merged intervals
        for (s, e) in data.get("merged", []):
            if e > max_frame:
                max_frame = e
    # From GT
    for intervals in gt_by_label.values():
        for (gs, ge) in intervals:
            if ge > max_frame:
                max_frame = ge
    if start_frame is None or start_frame < 0:
        start_frame = 0
    if start_frame > max_frame:
        start_frame = max_frame
    f1_per_obj = {}
    f1_values_pos = []
    f1_values_weighted = []
    f1_values_macro_balanced = []
    for oid, data in processed.items():
        # Handle both numeric object ids and string group names
        oid_int = None
        try:
            oid_int = int(oid)
        except Exception:
            oid_int = None

        if available_ids is not None and oid_int is not None and oid_int not in available_ids:
            f1_per_obj[oid] = None
            continue
        pred_mask = np.zeros(max_frame + 1, dtype=np.uint8)
        for (s, e) in data.get("merged", []):
            pred_mask[s:e+1] = 1
        label = obj_to_interaction.get(oid if oid in obj_to_interaction else oid_int)
        # Optionally skip this object if label does not match cam_prefix
        if cam_prefix is not None and label is not None and isinstance(label, str) and not label.upper().startswith(cam_prefix.upper()):
            f1_per_obj[oid] = None
            continue
        gt_intervals = gt_by_label.get(label, [])
        gt_mask = np.zeros(max_frame + 1, dtype=np.uint8)
        for (gs, ge) in gt_intervals:
            gt_mask[gs:ge+1] = 1
        # Restrict evaluation window to frames >= start_frame
        pred_eval = pred_mask[start_frame:] if start_frame else pred_mask
        gt_eval = gt_mask[start_frame:] if start_frame else gt_mask
        TP = int(np.sum((pred_eval == 1) & (gt_eval == 1)))
        FP = int(np.sum((pred_eval == 1) & (gt_eval == 0)))
        FN = int(np.sum((pred_eval == 0) & (gt_eval == 1)))
        TN = int(np.sum((pred_eval == 0) & (gt_eval == 0)))

        # Positive-class F1 (interaction)
        if TP == 0 and FP == 0 and FN == 0:
            f1_pos = 0.0
        else:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Negative-class F1 (no_interaction)
        # Treat TN as TP_neg, FN as FP_neg, FP as FN_neg
        precision_neg = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        recall_neg    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0.0

        # Weighted by support of each class
        pos_support = TP + FN
        neg_support = TN + FP
        total_support = pos_support + neg_support
        if total_support > 0:
            f1_weighted = (pos_support * f1_pos + neg_support * f1_neg) / total_support
        else:
            f1_weighted = None

        # Equal-weight macro over interaction and no_interaction
        f1_macro_balanced = 0.5 * (f1_pos + f1_neg)

        f1_per_obj[oid] = {"pos": f1_pos, "weighted": f1_weighted, "balanced": f1_macro_balanced}
        f1_values_pos.append(f1_pos)
        if f1_weighted is not None:
            f1_values_weighted.append(f1_weighted)
        f1_values_macro_balanced.append(f1_macro_balanced)

    f1_macro_pos = float(np.mean(f1_values_pos)) if f1_values_pos else None
    f1_macro_weighted = float(np.mean(f1_values_weighted)) if f1_values_weighted else None
    f1_macro_balanced = float(np.mean(f1_values_macro_balanced)) if f1_values_macro_balanced else None
    return f1_per_obj, f1_macro_pos, f1_macro_weighted, f1_macro_balanced

def interval_overlap_frames(a, b):
    s = max(int(a[0]), int(b[0]))
    e = min(int(a[1]), int(b[1]))
    return max(0, e - s + 1)

def union_interval_frames(intervals):
    if not intervals:
        return 0
    runs = sorted((int(s), int(e)) for (s, e) in intervals)
    total = 0
    ps, pe = runs[0]
    for cs, ce in runs[1:]:
        if cs <= pe + 1:
            pe = max(pe, ce)
        else:
            total += pe - ps + 1
            ps, pe = cs, ce
    total += pe - ps + 1
    return total

def total_gt_pred_overlap_frames(gt_intervals, pred_intervals):
    """
    Total overlapped duration between two interval sets (symmetric).
    Computes union over all pairwise intersections once to avoid double-counting.
    """
    overlaps = []
    for as_, ae in gt_intervals:
        for bs, be in pred_intervals:
            s = max(int(as_), int(bs))
            e = min(int(ae), int(be))
            if e >= s:
                overlaps.append((s, e))
    return union_interval_frames(overlaps)

def merge_sorted_intervals(intervals, merge_gap_frames=0, min_dur_frames=1):
    if not intervals:
        return []
    runs = sorted((int(s), int(e)) for (s, e) in intervals)
    merged = []
    ps, pe = runs[0]
    for cs, ce in runs[1:]:
        if cs - pe <= int(merge_gap_frames):
            pe = max(pe, ce)
        else:
            merged.append((ps, pe))
            ps, pe = cs, ce
    merged.append((ps, pe))
    return [(s, e) for (s, e) in merged if (e - s + 1) >= int(min_dur_frames)]


def _subtract_intervals(spans, subtract):
    """Remove all time covered by `subtract` from each span in `spans`."""
    result = []
    for (s, e) in spans:
        pieces = [(s, e)]
        for (rs, re) in sorted(subtract):
            new_pieces = []
            for (ps, pe) in pieces:
                if re < ps or rs > pe:
                    new_pieces.append((ps, pe))
                else:
                    if ps < rs:
                        new_pieces.append((ps, rs - 1))
                    if pe > re:
                        new_pieces.append((re + 1, pe))
            pieces = new_pieces
        result.extend(pieces)
    return sorted(result)


def _resolve_iv_propaq_conflicts_predicted(processed):
    """Trim PROPAQ predicted segments that overlap IV segments (same patient)."""
    for iv_oids, propaq_oid in (({80, 81, 82, 83}, 85), ({86, 87, 88, 89}, 91)):
        if propaq_oid not in processed:
            continue
        iv_spans = []
        for iv_oid in iv_oids:
            if iv_oid in processed:
                iv_spans.extend(processed[iv_oid].get("merged", []))
        if not iv_spans:
            continue
        trimmed = _subtract_intervals(processed[propaq_oid]["merged"], iv_spans)
        processed[propaq_oid]["merged"] = trimmed


def _resolve_iv_propaq_conflicts_gt(gt_by_label):
    """Trim PROPAQ GT segments that overlap IV GT segments (same patient)."""
    for side in ("DAVE", "CHUCK"):
        iv_lbl, propaq_lbl = f"{side}_IV", f"{side}_PROPAQ"
        if propaq_lbl not in gt_by_label or iv_lbl not in gt_by_label:
            continue
        trimmed = _subtract_intervals(gt_by_label[propaq_lbl], gt_by_label[iv_lbl])
        if trimmed:
            gt_by_label[propaq_lbl] = trimmed
        else:
            del gt_by_label[propaq_lbl]


def merge_gt_dicts(gt_dicts, merge_gap_frames=0, min_dur_frames=1):
    merged = {}
    for gt_by_label in gt_dicts:
        for label, spans in (gt_by_label or {}).items():
            merged.setdefault(label, []).extend((int(s), int(e)) for (s, e) in spans)
    for label in list(merged.keys()):
        merged[label] = merge_sorted_intervals(
            merged[label],
            merge_gap_frames=merge_gap_frames,
            min_dur_frames=min_dur_frames,
        )
        if not merged[label]:
            del merged[label]
    return merged

def compute_interval_metrics(processed, gt_by_label, obj_to_interaction, cam_prefix=None, available_ids=None, start_frame=None, fps=4.0):
    """
    Interval metrics:
      - overlap_ratio = overlapped GT duration / GT duration
      - false_interactions_count = predicted intervals with zero overlap against GT
      - false_interaction_pred_duration_pct = duration of predicted interaction outside GT / total predicted interaction duration
      - false_interaction_pred_duration_pct_per_video_duration = duration of predicted interaction outside GT / total evaluated video duration
      - model_start_latency_s = average delay between GT start and earliest overlapping predicted start (over GT events that overlap)
    """
    metrics_per_obj = {}
    overlap_vals = []
    false_count_vals = []
    false_count_pct_vals = []
    false_pred_dur_pct_vals = []
    false_pred_dur_pct_video_vals = []
    latency_vals = []
    total_false_count = 0

    for oid, data in processed.items():
        oid_int = None
        try:
            oid_int = int(oid)
        except Exception:
            oid_int = None

        if available_ids is not None and oid_int is not None and oid_int not in available_ids:
            metrics_per_obj[oid] = None
            continue

        label = obj_to_interaction.get(oid if oid in obj_to_interaction else oid_int)
        if cam_prefix is not None and label is not None and isinstance(label, str) and not label.upper().startswith(cam_prefix.upper()):
            metrics_per_obj[oid] = None
            continue

        pred_intervals = [(int(s), int(e)) for (s, e) in data.get("merged", [])]
        gt_intervals = [(int(s), int(e)) for (s, e) in gt_by_label.get(label, [])]

        if start_frame is not None and start_frame > 0:
            pred_intervals = [(max(s, start_frame), e) for (s, e) in pred_intervals if e >= start_frame]
            gt_intervals = [(max(s, start_frame), e) for (s, e) in gt_intervals if e >= start_frame]

        gt_total = union_interval_frames(gt_intervals)
        overlap_total = total_gt_pred_overlap_frames(gt_intervals, pred_intervals)
        overlap_ratio = (overlap_total / gt_total) if gt_total > 0 else None

        false_count = 0
        for ps, pe in pred_intervals:
            if not any(interval_overlap_frames((ps, pe), (gs, ge)) > 0 for (gs, ge) in gt_intervals):
                false_count += 1
        pred_count = len(pred_intervals)
        false_count_pct = (false_count / pred_count) if pred_count > 0 else None

        pred_total = union_interval_frames(pred_intervals)
        overlap_pred_total = total_gt_pred_overlap_frames(gt_intervals, pred_intervals)
        false_pred_dur = max(0, pred_total - overlap_pred_total)
        false_pred_dur_pct = (false_pred_dur / pred_total) if pred_total > 0 else None
        # Evaluated video duration (frame count) for this object/group after start-frame clipping.
        all_ends = [e for _, e in pred_intervals] + [e for _, e in gt_intervals]
        if all_ends:
            eval_start = int(start_frame) if (start_frame is not None and start_frame > 0) else 0
            eval_end = int(max(all_ends))
            eval_video_dur_frames = max(0, eval_end - eval_start + 1)
        else:
            eval_video_dur_frames = 0
        false_pred_dur_pct_video = (false_pred_dur / eval_video_dur_frames) if eval_video_dur_frames > 0 else None

        latency_s = None
        if gt_intervals and pred_intervals:
            # Metric 4: per GT event, compare GT start vs earliest overlapping prediction start.
            per_gt_lat = []
            for gs, ge in gt_intervals:
                starts = [
                    ps for (ps, pe) in pred_intervals
                    if interval_overlap_frames((ps, pe), (gs, ge)) > 0
                ]
                if starts:
                    per_gt_lat.append((min(starts) - gs) / float(fps))
            if per_gt_lat:
                latency_s = float(np.mean(per_gt_lat))

        metrics_per_obj[oid] = {
            "gt_pred_overlap_ratio": overlap_ratio,
            "false_interactions_count": false_count,
            "false_interactions_count_pct": false_count_pct,
            "false_interaction_pred_duration_pct": false_pred_dur_pct,
            "false_interaction_pred_duration_pct_per_video_duration": false_pred_dur_pct_video,
            "false_interaction_pred_duration_frames": false_pred_dur,
            "pred_total_duration_frames": pred_total,
            "eval_video_duration_frames": eval_video_dur_frames,
            "model_start_latency_s": latency_s,
        }

        if overlap_ratio is not None:
            overlap_vals.append(overlap_ratio)
        if pred_total > 0:
            false_count_vals.append(false_count)
        if false_count_pct is not None:
            false_count_pct_vals.append(false_count_pct)
        total_false_count += int(false_count)
        if false_pred_dur_pct is not None:
            false_pred_dur_pct_vals.append(false_pred_dur_pct)
        if false_pred_dur_pct_video is not None:
            false_pred_dur_pct_video_vals.append(false_pred_dur_pct_video)
        if latency_s is not None:
            latency_vals.append(latency_s)

    summary = {
        "macro_overlap_ratio": float(np.mean(overlap_vals)) if overlap_vals else None,
        "macro_false_interactions_count": float(np.mean(false_count_vals)) if false_count_vals else None,
        "macro_false_interactions_count_pct": float(np.mean(false_count_pct_vals)) if false_count_pct_vals else None,
        "total_false_interactions_count": int(total_false_count),
        "macro_false_interaction_pred_duration_pct": float(np.mean(false_pred_dur_pct_vals)) if false_pred_dur_pct_vals else None,
        "macro_false_interaction_pred_duration_pct_per_video_duration": float(np.mean(false_pred_dur_pct_video_vals)) if false_pred_dur_pct_video_vals else None,
        "macro_model_start_latency_s": float(np.mean(latency_vals)) if latency_vals else None,
    }
    return metrics_per_obj, summary

def fuse_processed_runs(processed_runs, fps, merge_gap_s=4.0, min_dur_s=4.0):
    mgap = int(round(float(merge_gap_s) * float(fps)))
    mind = int(round(float(min_dur_s) * float(fps)))
    by_oid = {}

    for processed in processed_runs:
        for oid, data in (processed or {}).items():
            oid_int = int(oid)
            entry = by_oid.setdefault(oid_int, {
                "score_by_frame": {},
                "verb_by_frame": {},
                "obj_by_frame": {},
                "subj_box_by_frame": {},
                "object_box_by_frame": {},
                "thresholds": [],
                "sigmas": [],
            })
            frames = np.asarray(data.get("frames", []), dtype=int)
            hoi = np.asarray(data.get("hoi", []), dtype=float)
            verb = np.asarray(data.get("verb", []), dtype=float)
            obj = np.asarray(data.get("obj", []), dtype=float)

            for idx, fid in enumerate(frames):
                fid = int(fid)
                entry["score_by_frame"][fid] = entry["score_by_frame"].get(fid, 0.0) + float(hoi[idx])
                if idx < len(verb):
                    entry["verb_by_frame"][fid] = entry["verb_by_frame"].get(fid, 0.0) + float(verb[idx])
                if idx < len(obj):
                    entry["obj_by_frame"][fid] = entry["obj_by_frame"].get(fid, 0.0) + float(obj[idx])
            # Preserve subject/person boxes per frame for interval-level CSV export.
            for fid, box in (data.get("frame_to_subj", {}) or {}).items():
                fi = int(fid)
                if fi not in entry["subj_box_by_frame"] and box is not None:
                    entry["subj_box_by_frame"][fi] = box
            for fid, box in (data.get("frame_to_box", {}) or {}).items():
                fi = int(fid)
                if fi not in entry["object_box_by_frame"] and box is not None:
                    entry["object_box_by_frame"][fi] = box

            if data.get("thresh") is not None:
                entry["thresholds"].append(float(data["thresh"]))
            if data.get("sigma") is not None:
                entry["sigmas"].append(float(data["sigma"]))

    fused = {}
    for oid_int, entry in by_oid.items():
        frames = np.array(sorted(entry["score_by_frame"].keys()), dtype=int)
        hoi = np.array([entry["score_by_frame"][int(fid)] for fid in frames], dtype=float)
        verb = np.array([entry["verb_by_frame"].get(int(fid), 0.0) for fid in frames], dtype=float)
        obj = np.array([entry["obj_by_frame"].get(int(fid), 0.0) for fid in frames], dtype=float)
        thresh = max(entry["thresholds"]) if entry["thresholds"] else DEFAULT_OBJ_THRESH.get(oid_int, 0.25)
        sigma_val = max(entry["sigmas"]) if entry["sigmas"] else DEFAULT_OBJ_SIGMA.get(oid_int, 3)

        keep_idx = np.where(hoi > thresh)[0]
        raw_runs = [(frames[s], frames[e]) for (s, e) in contiguous_runs(keep_idx)]
        merged = merge_sorted_intervals(raw_runs, merge_gap_frames=mgap, min_dur_frames=mind)

        fused[oid_int] = {
            "frames": frames,
            "hoi": hoi,
            "verb": verb,
            "obj": obj,
            "merged": merged,
            "thresh": thresh,
            "sigma": sigma_val,
            "sel": None,
            "frame_to_box": {},
            "frame_to_subj": dict(entry.get("subj_box_by_frame", {})),
            "frame_to_obj": dict(entry.get("object_box_by_frame", {})),
        }
    return fused

def emit_metric_summary(header_prefix, processed, gt_by_label, obj_to_interaction, *,
                        cam_prefix=None, available_ids=None, start_frame=None, fps=4.0):
    f1_per_obj, f1_macro_pos, f1_macro_weighted, _f1_macro_balanced = compute_frame_f1(
        processed, gt_by_label, obj_to_interaction,
        cam_prefix=cam_prefix, available_ids=available_ids, start_frame=start_frame
    )
    interval_per_obj, interval_summary_obj = compute_interval_metrics(
        processed, gt_by_label, obj_to_interaction,
        cam_prefix=cam_prefix, available_ids=available_ids, start_frame=start_frame, fps=fps
    )
    grouped_pred, grouped_gt = aggregate_by_group(processed, gt_by_label, start_frame=start_frame)
    f1_group, f1_group_pos, f1_group_weighted, f1_group_macro_all = compute_frame_f1(
        grouped_pred, grouped_gt, {k: k for k in grouped_pred.keys()},
        cam_prefix=None, available_ids=None, start_frame=None
    )
    interval_group, interval_summary_group = compute_interval_metrics(
        grouped_pred, grouped_gt, {k: k for k in grouped_pred.keys()},
        cam_prefix=None, available_ids=None, start_frame=None, fps=fps
    )

    print(f"{header_prefix}Frame-level F1 per object:")
    for oid, f1v in f1_per_obj.items():
        lbl = oid_to_label(oid)
        print(f"  obj={lbl} ({oid}): pos={f1v['pos'] if isinstance(f1v, dict) else f1v}  weighted={f1v['weighted'] if isinstance(f1v, dict) else None}")
    print(f"{header_prefix}Frame-level macro F1 (pos): {f1_macro_pos}")
    print(f"{header_prefix}Frame-level macro F1 (weighted incl. no_interaction): {f1_macro_weighted}")

    print(f"{header_prefix}Interval metrics per object:")
    for oid, mv in interval_per_obj.items():
        lbl = oid_to_label(oid)
        if isinstance(mv, dict):
            print(f"  obj={lbl} ({oid}): overlap={mv['gt_pred_overlap_ratio']}  false_count={mv['false_interactions_count']}  false_count_pct={mv['false_interactions_count_pct']}  false_pred_dur_pct={mv['false_interaction_pred_duration_pct']}  start_latency_s={mv['model_start_latency_s']}")
        else:
            print(f"  obj={lbl} ({oid}): overlap=None  false_count=None  false_count_pct=None  false_pred_dur_pct=None  start_latency_s=None")
    print(f"{header_prefix}Interval macro overlap ratio (object): {interval_summary_obj['macro_overlap_ratio']}")
    print(f"{header_prefix}Interval macro falsely predicted interactions count (object): {interval_summary_obj['macro_false_interactions_count']}")
    print(f"{header_prefix}Interval macro falsely predicted interactions count pct (object): {interval_summary_obj['macro_false_interactions_count_pct']}")
    print(f"{header_prefix}Interval total falsely predicted interactions count (object): {interval_summary_obj['total_false_interactions_count']}")
    print(f"{header_prefix}Interval macro false interaction prediction duration pct (object): {interval_summary_obj['macro_false_interaction_pred_duration_pct']}")
    print(f"{header_prefix}Interval macro false interaction prediction duration pct per total video duration (object): {interval_summary_obj['macro_false_interaction_pred_duration_pct_per_video_duration']}")
    print(f"{header_prefix}Interval macro model start latency s (object): {interval_summary_obj['macro_model_start_latency_s']}")

    print(f"{header_prefix}Frame-level F1 per group (IV/MV/Propaq per side):")
    for grp, f1v in f1_group.items():
        print(f"  {grp}: pos={f1v['pos'] if isinstance(f1v, dict) else f1v}  weighted={f1v['weighted'] if isinstance(f1v, dict) else None}")
    print(f"{header_prefix}Frame-level macro F1 (group pos): {f1_group_pos}")
    print(f"{header_prefix}Frame-level macro F1 (group weighted incl. no_interaction): {f1_group_weighted}")
    print(f"{header_prefix}Frame-level macro F1 (group equal pos/neg): {f1_group_macro_all}")

    print(f"{header_prefix}Interval metrics per group (IV/MV/Propaq per side):")
    for grp, mv in interval_group.items():
        if isinstance(mv, dict):
            print(f"  {grp}: overlap={mv['gt_pred_overlap_ratio']}  false_count={mv['false_interactions_count']}  false_count_pct={mv['false_interactions_count_pct']}  false_pred_dur_pct={mv['false_interaction_pred_duration_pct']}  start_latency_s={mv['model_start_latency_s']}")
        else:
            print(f"  {grp}: overlap=None  false_count=None  false_count_pct=None  false_pred_dur_pct=None  start_latency_s=None")
    print(f"{header_prefix}Interval macro overlap ratio (group): {interval_summary_group['macro_overlap_ratio']}")
    print(f"{header_prefix}Interval macro falsely predicted interactions count (group): {interval_summary_group['macro_false_interactions_count']}")
    print(f"{header_prefix}Interval macro falsely predicted interactions count pct (group): {interval_summary_group['macro_false_interactions_count_pct']}")
    print(f"{header_prefix}Interval total falsely predicted interactions count (group): {interval_summary_group['total_false_interactions_count']}")
    print(f"{header_prefix}Interval macro false interaction prediction duration pct (group): {interval_summary_group['macro_false_interaction_pred_duration_pct']}")
    print(f"{header_prefix}Interval macro false interaction prediction duration pct per total video duration (group): {interval_summary_group['macro_false_interaction_pred_duration_pct_per_video_duration']}")
    print(f"{header_prefix}Interval macro model start latency s (group): {interval_summary_group['macro_model_start_latency_s']}")

    return {
        "f1_per_obj": f1_per_obj,
        "f1_macro_pos": f1_macro_pos,
        "f1_macro_weighted": f1_macro_weighted,
        "interval_per_obj": interval_per_obj,
        "interval_summary_obj": interval_summary_obj,
        "grouped_pred": grouped_pred,
        "grouped_gt": grouped_gt,
        "f1_group": f1_group,
        "f1_group_pos": f1_group_pos,
        "f1_group_weighted": f1_group_weighted,
        "f1_group_macro_all": f1_group_macro_all,
        "interval_group": interval_group,
        "interval_summary_group": interval_summary_group,
    }

def save_processed_plots(processed, gt_by_label, fps, demo_outdir, *,
                         first_interaction_frame=None, title="", plot_prefix=""):
    if not demo_outdir:
        return
    base_plot_path = Path(demo_outdir) / "plots"
    base_plot_path.mkdir(parents=True, exist_ok=True)

    safe_prefix = str(plot_prefix or "")
    # Simulation-level aggregated GT by semantic object group
    _grouped_gt_raw = {}
    for _lbl, _spans in (gt_by_label or {}).items():
        _grp = label_to_group(_lbl)
        if not _grp:
            continue
        _grouped_gt_raw.setdefault(_grp, []).extend((int(s), int(e)) for (s, e) in _spans)
    grouped_gt_sim = {}
    for _grp, _spans in _grouped_gt_raw.items():
        # merge overlaps/touching intervals for clean simulation-level GT overlay
        grouped_gt_sim[_grp] = merge_sorted_intervals(_spans, merge_gap_frames=0, min_dur_frames=1)

    for idx, (oid, data) in enumerate(processed.items()):
        xs = data["frames"]
        hoi = data["hoi"]
        verb = data["verb"]
        obj = data["obj"]
        merged = data["merged"]
        thresh_obj = data.get("thresh")
        sigma_obj = data.get("sigma", DEFAULT_OBJ_SIGMA.get(int(oid), 3))

        fig, (ax_raw, ax_sm) = plt.subplots(
            nrows=2, ncols=1, figsize=(16, 8), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]}
        )

        sel_raw = data.get("sel")
        if sel_raw is not None and not sel_raw.empty:
            xs_raw = sel_raw.index.values
            raw_hoi = sel_raw["score"].values
            raw_verb = sel_raw["verb_scores_index_decoder"].values if "verb_scores_index_decoder" in sel_raw.columns else None
            raw_obj = sel_raw["obj_scores"].apply(extract_obj_score).values if "obj_scores" in sel_raw.columns else None
            ax_raw.plot(xs_raw, raw_hoi, label="Raw HOI Score")
            if raw_verb is not None:
                ax_raw.plot(xs_raw, raw_verb, label="Raw Verb Score", linestyle="--")
            if raw_obj is not None:
                ax_raw.plot(xs_raw, raw_obj, label="Raw Object Score", linestyle=":")
        else:
            ax_raw.plot(xs, hoi, label="HOI Score (Gaussian)")
            if len(verb):
                ax_raw.plot(xs, verb, label="Verb Score (Gaussian)", linestyle="--")
            if len(obj):
                ax_raw.plot(xs, obj, label="Object Score (Gaussian)", linestyle=":")

        ax_raw.set_ylabel("Score")
        ax_raw.grid(True, axis="y", linestyle="--", alpha=0.3)
        gt_name = OBJ_TO_INTERACTION.get(int(oid), oid_to_label(oid))
        group_name = OBJ_GROUP.get(int(oid))
        gt_group_intervals = grouped_gt_sim.get(group_name, [])
        for i, (gs, ge) in enumerate(gt_group_intervals):
            lbl = f"GT-group ({group_name})" if i == 0 and group_name else ("GT-group" if i == 0 else None)
            ax_raw.axvspan(gs, ge, color="magenta", alpha=0.08, label=lbl)
        for i, (s, e) in enumerate(merged):
            ax_raw.axvspan(s, e, color="green", alpha=0.12, label="Predicted interval" if i == 0 else None)
        annotate_interval_time_labels(ax_raw, gt_group_intervals, fps, y_frac=0.96, color="magenta")
        annotate_interval_time_labels(ax_raw, merged, fps, y_frac=0.72, color="green")
        annotate_gt_status(ax_raw, gt_group_intervals)
        if first_interaction_frame is not None and first_interaction_frame >= 0:
            start_lbl = f"Start of simulation ({frame_to_mmss(first_interaction_frame, fps)})"
            ax_raw.axvline(first_interaction_frame, color="black", linestyle=":", linewidth=1.2, label=start_lbl)
        ax_raw.legend(fontsize=9, loc="upper left")
        ax_raw.set_title(f"Raw Frame-level Scores  |  obj={oid}, verb=117  |  {title}", fontsize=11)

        ax_sm.plot(xs, hoi, label="HOI Score (Gaussian)")
        ax_sm.plot(xs, verb, label="Verb Score (Gaussian)", linestyle="--")
        ax_sm.plot(xs, obj, label="Object Score (Gaussian)", linestyle=":")
        ax_sm.axhline(thresh_obj, linestyle="--", linewidth=1, label=f"Threshold = {thresh_obj}")
        for i, (gs, ge) in enumerate(gt_group_intervals):
            lbl = f"GT-group ({group_name})" if i == 0 and group_name else ("GT-group" if i == 0 else None)
            ax_sm.axvspan(gs, ge, color="magenta", alpha=0.08, label=lbl)
        for i, (s, e) in enumerate(merged):
            ax_sm.axvspan(s, e, color="green", alpha=0.12, label="Predicted interval" if i == 0 else None)
        annotate_interval_time_labels(ax_sm, gt_group_intervals, fps, y_frac=0.96, color="magenta")
        annotate_interval_time_labels(ax_sm, merged, fps, y_frac=0.72, color="green")
        annotate_gt_status(ax_sm, gt_group_intervals)
        if first_interaction_frame is not None and first_interaction_frame >= 0:
            start_lbl = f"Start of simulation ({frame_to_mmss(first_interaction_frame, fps)})"
            ax_sm.axvline(first_interaction_frame, color="black", linestyle=":", linewidth=1.2, label=start_lbl)
        gt_flag = f" + GT-group[{group_name}]" if group_name and gt_group_intervals else ""
        ax_sm.set_title(
            f"Smoothed Scores with GT and Predictions{gt_flag}  |  obj={oid}  |  FPS={fps:.2f}  |  Th={thresh_obj}, σ={sigma_obj}",
            fontsize=11
        )
        ax_sm.set_xlabel("Time (mm:ss)")
        ax_sm.set_ylabel("Score")
        ax_sm.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax_sm.legend(fontsize=9, loc="upper left")

        if len(xs) > 0:
            tick_ct = min(10, len(xs))
            ticks = np.linspace(0, len(xs)-1, tick_ct, dtype=int)
            ax_sm.set_xticks(xs[ticks])
            ax_sm.set_xticklabels([frame_to_mmss(f, fps) for f in xs[ticks]], rotation=45)

        plt.tight_layout()
        out_plot = base_plot_path / f"plot_{safe_prefix}obj{oid}.png"
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

    grouped_pred, grouped_gt = aggregate_by_group(processed, gt_by_label, start_frame=first_interaction_frame)
    for grp_name, pdata in grouped_pred.items():
        pred_spans = pdata.get("merged", [])
        gt_spans = grouped_gt.get(grp_name, [])
        if not pred_spans and not gt_spans:
            continue

        max_frame = 0
        for s, e in pred_spans + gt_spans:
            max_frame = max(max_frame, e)

        member_oids = []
        for oid, od in processed.items():
            try:
                oid_int = int(oid)
            except Exception:
                continue
            if OBJ_GROUP.get(oid_int) == grp_name:
                member_oids.append(oid_int)
        member_oids = sorted(member_oids)

        def overlay_spans(ax_):
            ax_.set_ylim(0.0, 1.0)
            for i, (gs, ge) in enumerate(gt_spans):
                ax_.axvspan(gs, ge, color="red", alpha=0.15, label="GT" if i == 0 else None)
            for i, (s, e) in enumerate(pred_spans):
                ax_.axvspan(s, e, color="green", alpha=0.12, label="Predicted" if i == 0 else None)
            annotate_interval_time_labels(ax_, gt_spans, fps, y_frac=0.88, color="red")
            annotate_interval_time_labels(ax_, pred_spans, fps, y_frac=0.72, color="green")
            annotate_gt_status(ax_, gt_spans)
            if first_interaction_frame is not None and first_interaction_frame >= 0:
                start_lbl = f"Start ({frame_to_mmss(first_interaction_frame, fps)})"
                ax_.axvline(first_interaction_frame, color="black", linestyle=":", linewidth=1.2, label=start_lbl)

        is_iv_group = str(grp_name).upper().endswith("_IV")
        if is_iv_group and member_oids:
            iv1_oids, iv2_oids = [], []
            for oid_int in member_oids:
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}").upper()
                if "IV1" in inst_name:
                    iv1_oids.append(oid_int)
                elif "IV2" in inst_name:
                    iv2_oids.append(oid_int)

            fig, (ax_raw1, ax_raw2, ax_sm, ax_int) = plt.subplots(
                nrows=4, ncols=1, figsize=(16, 11), sharex=True,
                gridspec_kw={"height_ratios": [1, 1, 1, 0.8]}
            )
            for oid_int in (iv1_oids or []):
                od = processed.get(oid_int) or processed.get(str(oid_int))
                if not od:
                    continue
                sel = od.get("sel")
                xs_plot = od.get("frames")
                hoi_plot = od.get("hoi")
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}")
                if sel is not None and not sel.empty:
                    ax_raw1.plot(sel.index.values, sel["score"].values, linestyle=":", linewidth=1.0, alpha=0.75, label=f"{inst_name} raw")
                if xs_plot is not None and hoi_plot is not None:
                    ax_raw1.plot(xs_plot, hoi_plot, linestyle="-", linewidth=2.0, alpha=0.95, label=f"{inst_name} smoothed")
                th = od.get("thresh")
                if th is not None:
                    ax_raw1.axhline(float(th), linestyle="--", linewidth=1, alpha=0.8, label=f"Th {inst_name}={float(th):.2f}")
            overlay_spans(ax_raw1)
            ax_raw1.set_ylabel("Score")
            ax_raw1.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_raw1.legend(fontsize=9, loc="upper right")
            ax_raw1.set_title(f"IV1 raw+smoothed HOI scores: {grp_name} | FPS={fps:.2f}")

            for oid_int in (iv2_oids or []):
                od = processed.get(oid_int) or processed.get(str(oid_int))
                if not od:
                    continue
                sel = od.get("sel")
                xs_plot = od.get("frames")
                hoi_plot = od.get("hoi")
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}")
                if sel is not None and not sel.empty:
                    ax_raw2.plot(sel.index.values, sel["score"].values, linestyle=":", linewidth=1.0, alpha=0.75, label=f"{inst_name} raw")
                if xs_plot is not None and hoi_plot is not None:
                    ax_raw2.plot(xs_plot, hoi_plot, linestyle="-", linewidth=2.0, alpha=0.95, label=f"{inst_name} smoothed")
                th = od.get("thresh")
                if th is not None:
                    ax_raw2.axhline(float(th), linestyle="--", linewidth=1, alpha=0.8, label=f"Th {inst_name}={float(th):.2f}")
            overlay_spans(ax_raw2)
            ax_raw2.set_ylabel("Score")
            ax_raw2.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_raw2.legend(fontsize=9, loc="upper right")
            ax_raw2.set_title(f"IV2 raw+smoothed HOI scores: {grp_name} | FPS={fps:.2f}")

            for oid_int in member_oids:
                od = processed.get(oid_int) or processed.get(str(oid_int))
                if not od:
                    continue
                xs_plot = od.get("frames")
                hoi_plot = od.get("hoi")
                if xs_plot is None or hoi_plot is None:
                    continue
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}")
                ax_sm.plot(xs_plot, hoi_plot, label=f"{inst_name} smoothed")
                th = od.get("thresh")
                if th is not None:
                    ax_sm.axhline(float(th), linestyle="--", linewidth=1, alpha=0.5, label=f"Th {inst_name}={float(th):.2f}")
            overlay_spans(ax_sm)
            ax_sm.set_ylabel("Score")
            ax_sm.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_sm.legend(fontsize=9, loc="upper right")
            ax_sm.set_title(f"IV instance smoothed HOI scores: {grp_name}")

            overlay_spans(ax_int)
            ax_int.set_yticks([])
            ax_int.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_int.set_title(f"Grouped intervals: {grp_name}")
            ax_int.set_xlabel("Time (mm:ss)")
            if max_frame > 0:
                ticks = np.linspace(0, max_frame, 10, dtype=int)
                ax_int.set_xticks(ticks)
                ax_int.set_xticklabels([frame_to_mmss(f, fps) for f in ticks], rotation=45)
            plt.tight_layout()
            plt.savefig(base_plot_path / f"plot_{safe_prefix}grp_{grp_name}_with_scores.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            fig, (ax_raw, ax_sm, ax_int) = plt.subplots(
                nrows=3, ncols=1, figsize=(16, 9), sharex=True,
                gridspec_kw={"height_ratios": [1, 1, 0.8]}
            )
            for oid_int in member_oids:
                od = processed.get(oid_int) or processed.get(str(oid_int))
                if not od:
                    continue
                sel = od.get("sel")
                xs_plot = od.get("frames")
                hoi_plot = od.get("hoi")
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}")
                if sel is not None and not sel.empty:
                    ax_raw.plot(sel.index.values, sel["score"].values, linestyle=":", linewidth=1.0, alpha=0.75, label=f"{inst_name} raw")
                if xs_plot is not None and hoi_plot is not None:
                    ax_raw.plot(xs_plot, hoi_plot, linestyle="-", linewidth=2.0, alpha=0.95, label=f"{inst_name} smoothed")
                th = od.get("thresh")
                if th is not None:
                    ax_raw.axhline(float(th), linestyle="--", linewidth=1, alpha=0.8, label=f"Th {inst_name}={float(th):.2f}")
            overlay_spans(ax_raw)
            ax_raw.set_ylabel("Score")
            ax_raw.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_raw.legend(fontsize=9, loc="upper right")
            ax_raw.set_title(f"{grp_name} raw+smoothed HOI scores | FPS={fps:.2f}")

            for oid_int in member_oids:
                od = processed.get(oid_int) or processed.get(str(oid_int))
                if not od:
                    continue
                xs_plot = od.get("frames")
                hoi_plot = od.get("hoi")
                if xs_plot is None or hoi_plot is None:
                    continue
                inst_name = OBJID_TO_INSTANCE_NAME.get(oid_int, f"obj{oid_int}")
                ax_sm.plot(xs_plot, hoi_plot, label=f"{inst_name} smoothed")
                th = od.get("thresh")
                if th is not None:
                    ax_sm.axhline(float(th), linestyle="--", linewidth=1, alpha=0.5, label=f"Th {inst_name}={float(th):.2f}")
            overlay_spans(ax_sm)
            ax_sm.set_ylabel("Score")
            ax_sm.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_sm.legend(fontsize=9, loc="upper right")
            ax_sm.set_title(f"{grp_name} smoothed HOI scores")

            overlay_spans(ax_int)
            ax_int.set_yticks([])
            ax_int.grid(True, axis="y", linestyle="--", alpha=0.3)
            ax_int.set_title(f"Grouped intervals: {grp_name}")
            ax_int.set_xlabel("Time (mm:ss)")
            if max_frame > 0:
                ticks = np.linspace(0, max_frame, 10, dtype=int)
                ax_int.set_xticks(ticks)
                ax_int.set_xticklabels([frame_to_mmss(f, fps) for f in ticks], rotation=45)
            plt.tight_layout()
            plt.savefig(base_plot_path / f"plot_{safe_prefix}grp_{grp_name}_with_scores.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(14, 3))
        overlay_spans(ax2)
        ax2.set_title(f"Grouped intervals: {grp_name} | FPS={fps:.2f}")
        ax2.set_xlabel("Time (mm:ss)")
        ax2.set_yticks([])
        ax2.grid(True, axis="y", linestyle="--", alpha=0.3)
        if max_frame > 0:
            ticks = np.linspace(0, max_frame, 10, dtype=int)
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([frame_to_mmss(f, fps) for f in ticks], rotation=45)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(base_plot_path / f"plot_{safe_prefix}grp_{grp_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

def label_to_group(label: str):
    """Map GT interaction label to grouped family (IV/MV/Propaq) by patient side."""
    if not label:
        return None
    s = str(label).upper()
    if "DAVE" in s:
        if "IV" in s:
            return "DAVE_IV"
        if "MV" in s or "VENT" in s:
            return "DAVE_MV"
        if "PROP" in s or "MON" in s:
            return "DAVE_PROPAQ"
    if "CHUCK" in s:
        if "IV" in s:
            return "CHUCK_IV"
        if "MV" in s or "VENT" in s:
            return "CHUCK_MV"
        if "PROP" in s or "MON" in s:
            return "CHUCK_PROPAQ"
    return None

def aggregate_by_group(processed, gt_by_label, start_frame=None):
    """Aggregate predictions and GT intervals into device families for F1."""
    grouped = {}
    # aggregate predictions
    for oid, data in processed.items():
        grp = OBJ_GROUP.get(int(oid))
        if not grp:
            continue
        merged = data.get("merged", [])
        if start_frame is not None and start_frame > 0:
            merged = [(max(s, start_frame), e) for (s, e) in merged if e >= start_frame]
        g = grouped.setdefault(grp, {"merged": []})
        g["merged"].extend(merged)
    # merge overlapping intervals within each group
    for grp, data in grouped.items():
        runs = sorted(data["merged"])
        merged = []
        if runs:
            ps, pe = runs[0]
            for cs, ce in runs[1:]:
                if cs <= pe + 1:
                    pe = max(pe, ce)
                else:
                    merged.append((ps, pe)); ps, pe = cs, ce
            merged.append((ps, pe))
        data["merged"] = merged
    # aggregate GT
    gt_grouped = {}
    for label, intervals in gt_by_label.items():
        grp = label_to_group(label)
        if not grp:
            continue
        gt_grouped.setdefault(grp, []).extend(intervals)
    return grouped, gt_grouped

# ---------- GT loader helpers ----------

def load_gr_annotations(gt_file, gt_sheet_token, debug: bool = False):
    """
    Load ground-truth annotations from an Excel file using (Group, Patient_Name, Interaction_Type).

    Behavior
    --------
    1) Read *all* sheets from the Excel file at `gt_file`.
    2) Within each sheet, keep only rows whose 'Group' column value exactly matches
       the provided `gt_sheet_token` (after stripping whitespace).
    3) Construct a neutral lookup key of the form '{Patient_Name}{Interaction_Type}',
       stored in a column named 'GTLookupKey'.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary mapping GTLookupKey -> filtered DataFrame containing only rows
        for that (Patient_Name, Interaction_Type) under the specified Group.
        If nothing matches, returns an empty dict.
    """
    gt_file = str(gt_file)
    if gt_sheet_token is None:
        if debug:
            print("[load_gr_annotations] gt_sheet_token is None; returning empty dict.")
        return {}

    token_str = str(gt_sheet_token).strip()
    try:
        all_sheets = pd.read_excel(gt_file, sheet_name=None)
    except Exception as e:
        print(f"[load_gr_annotations] Failed to read GT Excel file {gt_file}: {e}")
        return {}

    def norm_col(name: str):
        """Normalize column names to improve matching when there are stray spaces/periods."""
        s = re.sub(r"[_\\s]+", " ", str(name or "")).strip().lower()
        s = s.replace(".", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s

    filtered_frames = []

    for sheet_name, df in all_sheets.items():
        if df is None or df.empty:
            continue

        # Build a map from normalized -> original column name
        norm_map = {norm_col(c): c for c in df.columns}

        def find_col(candidates):
            for cand in candidates:
                key = norm_col(cand)
                if key in norm_map:
                    return norm_map[key]
            # fallback: startswith match (handles Excel auto-suffix like '.1')
            for cand in candidates:
                key = norm_col(cand)
                for col in df.columns:
                    if norm_col(col).startswith(key):
                        return col
            return None

        group_col = find_col(["group"])
        patient_col = find_col(["patient_name", "patient"])
        interaction_col = find_col(["interaction_type", "interaction", "type"])

        if not group_col or not patient_col or not interaction_col:
            if debug:
                print(f"[load_gr_annotations] Skipping sheet '{sheet_name}' due to missing required columns. "
                      f"Found columns: {list(df.columns)}")
            continue

        # Filter by Group == gt_sheet_token
        mask = df[group_col].astype(str).str.strip() == token_str
        df_group = df.loc[mask].copy()
        if df_group.empty:
            if debug:
                print(f"[load_gr_annotations] Sheet '{sheet_name}': no rows matched Group='{token_str}'.")
            continue

        # Build neutral lookup key {Patient_Name}{Interaction_Type}
        df_group["GTLookupKey"] = (
            df_group[patient_col].astype(str).str.strip() +
            df_group[interaction_col].astype(str).str.strip()
        )

        filtered_frames.append(df_group)

    if not filtered_frames:
        if debug:
            print(f"[load_gr_annotations] No rows matched Group='{token_str}' across any sheets in {gt_file}.")
        return {}

    gt_df = pd.concat(filtered_frames, ignore_index=True)

    gt_by_key = {}
    for key, grp in gt_df.groupby("GTLookupKey"):
        gt_by_key[key] = grp.reset_index(drop=True)

    if debug:
        print(f"[load_gr_annotations] Loaded GT for {len(gt_by_key)} (Patient_Name, Interaction_Type) keys "
              f"from file: {gt_file} using Group='{token_str}'.")
    return gt_by_key
def load_gt_annotations(gt_path, fps, video_path=None, pred_path=None, debug=False, sheet_token_override=None):
    """
    Load GT intervals from either CSV or Excel (multiple sheets).
    Filters rows to those whose 'Patient' token is found in the video/pred filename if available.
    Returns:
      intervals: dict label -> list[(start_frame, end_frame)]
      first_interaction_frame: int or None (frame index of "Time of 1st interaction with equipment")
      patient_hint: "CHUCK"/"DAVE"/None (derived from Patient column if present)
    """
    if gt_path is None:
        return {}, None, None
    gt_path = str(gt_path)
    if Path(gt_path).suffix.lower() in (".xlsx", ".xls") and not sheet_token_override:
        print(f"Warning: GT file is a multi-sheet Excel but no --gt_sheet_token was provided. "
              f"Returning empty GT to prevent cross-simulation leakage. "
              f"Pass --gt_sheet_token <simulation_id> to load GT for the correct session.")
        return {}, None, None
    video_key = ""
    if debug:
        print("Debugging check video_path,pred_path,",video_path,pred_path)
    for cand in [video_path, pred_path]:
        if cand:
            video_key = Path(str(cand)).stem.lower()
            if debug:
                print("Debugging check video_key",video_key)
            break
    if sheet_token_override and debug:
        print("Debugging group token (sheet_token_override) ->", sheet_token_override)
    view_key = get_camera_view(video_path) or get_camera_view(pred_path)
    if debug:
        print("Debugging check view key",view_key)
    first_interaction_frame = None
    patient_hint = None

    def parse_rows(df):
        nonlocal patient_hint
        df = df.copy()

        def norm_col(name: str):
            """Normalize column names to improve matching when there are stray spaces/periods."""
            s = re.sub(r"[_\\s]+", " ", str(name or "")).strip().lower()
            s = s.replace(".", "_").replace(" ", "_")
            while "__" in s:
                s = s.replace("__", "_")
            return s

        def find_col(candidates):
            cand_norm = [norm_col(c) for c in candidates]
            norm_map = {norm_col(c): c for c in df.columns}
            for cn in cand_norm:
                if cn in norm_map:
                    return norm_map[cn]
            # fallback: startswith match (helps when Excel adds ".1" for duplicate headers)
            for cn in cand_norm:
                for col in df.columns:
                    if norm_col(col).startswith(cn):
                        return col
            return None

        # ------------------------------------------------------------
        # 1) GROUP FILTER: authoritative selection of GT rows
        # ------------------------------------------------------------
        if sheet_token_override:
            group_col = find_col(["group"])
            if group_col:
                token = sheet_token_override.strip().lower()
                mask_group = df[group_col].astype(str).str.strip().str.lower() == token

                if debug:
                    print(f"[GT debug] Group filter: gt_sheet_token={sheet_token_override}, "
                          f"matches={mask_group.sum()} / {len(df)}")

                df = df[mask_group]
                if df.empty:
                    if debug:
                        print("[GT debug] Group filter produced 0 rows; skipping this sheet.")
                    return {}, None
            else:
                # No Group column — cannot verify this sheet belongs to the requested group.
                # Skipping to prevent data from unrelated sessions leaking into GT.
                if debug:
                    print(f"[GT debug] Skipping sheet: no 'Group' column but "
                          f"sheet_token_override='{sheet_token_override}' requires group matching.")
                return {}, None

        # Filter by Patient token if present (supports Patient / Patient_name)
        # Use robust column detection instead of hard-coding 'patient_name' to avoid KeyError
        patient_col = find_col(["patient", "patient_name"])
        extra_patient_col = None
        # If both Patient and Patient_Name exist, treat the second as an auxiliary patient column
        patient_name_col = find_col(["patient_name"])
        if patient_col and patient_name_col and patient_name_col != patient_col:
            extra_patient_col = patient_name_col

        if debug:
            print("Debugging check,patient_col", patient_col, "extra_patient_col", extra_patient_col)
        # Candidate time columns (used later and for debug samples)
        start_cols = [
            "Hands-On Interaction Start Time (Hr:Min.Sec)",
            "Hands-On Interaction Start Time (Hr:Min:Sec)",
            "Hands-On Interaction Start Time",
            "Updated_Interaction_Start_Time","Start","start","start_time"
        ]
        end_cols = [
            "Hands-Off Interaction Stop Time (Hr:Min.Sec)",
            "Hands-Off Interaction Stop Time (Hr:Min:Sec)",
            "Hands-Off Interaction Stop Time",
            "Updated_Interaction_End_Time","End","end","end_time"
        ]
        if patient_col:
            # capture patient hint (e.g., Chuck/Dave)
            for v in df[patient_col].dropna().astype(str).str.lower():
                if "chuck" in v:
                    patient_hint = patient_hint or "CHUCK"
                    break
                if "dave" in v:
                    patient_hint = patient_hint or "DAVE"
                    break
            # Require at least one valid patient name in the column.
            # Sheets where patient values are missing (all NaN) or are non-patient
            # identifiers (e.g., simulation IDs) are skipped to prevent data leakage.
            if patient_hint is None:
                if debug:
                    uniq_vals = df[patient_col].dropna().astype(str).unique().tolist()[:5]
                    print(f"[GT debug] Skipping sheet: patient column '{patient_col}' contains no "
                          f"valid patient names (DAVE/CHUCK). Unique values seen: {uniq_vals}")
                return {}, None
        # No video-filename-based patient filter here.
        # All rows from the group-filtered sheet are kept; classify_interaction()
        # assigns CHUCK vs DAVE labels per row using Patient_name column.
        # GT intervals from multiple sheets are merged downstream by merge_gt_dicts(),
        # so overlapping segments across sheets are collapsed into a single GT interval.
        if debug and patient_col:
            uniq_pat = set(df[patient_col].dropna().astype(str).unique().tolist())
            print(f"[GT debug] Keeping all {len(df)} group-filtered rows. "
                  f"Patient values seen: {list(uniq_pat)[:10]}")
        if df.empty:
            return {}, None
        type_cols = [
            "Interaction_Type",
            "interaction_type",
        ]
        first_cols = [
            "Time of 1st interaction with equipment (Hr:Min.Sec)",
            "Time of 1st interaction with equipment (Hr:Min:Sec)",
            "Time of 1st interaction with equipment",
            "Time of 1st interaction with equipment ",
            "start_of_simulation","simulation_start","start_of_sim"
        ]
        start_col = find_col(start_cols)
        end_col   = find_col(end_cols)
        type_col  = find_col(type_cols)
        first_col = find_col(first_cols)
        occlusion_col = find_col(["Occlusion","occlusion", "occluded"])
        out = {}
        first_frame_local = None

        if debug:
            print("\n[GT debug] Columns found:")
            print(f"  start_col={start_col}, end_col={end_col}, type_col={type_col}, patient_col={patient_col}, first_col={first_col}, occlusion_col={occlusion_col}")
            print("[GT debug] Raw columns:", list(df.columns))
            if patient_col:
                print(f"[GT debug] Patient column='{patient_col}'; CHUCK/DAVE assigned per-row by classify_interaction (no filename filter).")

        def classify_interaction(raw_label, patient_side=None):
            """
            Map Interaction_Type text to canonical device labels.

            Semantics
            ---------
            - Any token containing 'iv' represents interaction with the IV pump subsystem
              (including IV lines and IV meds).
            - Any token containing 'vent'/'ventilator' represents interaction with the mechanical ventilator.
            - Any token containing 'propaq'/'prop'/'monitor' represents interaction with the Propaq monitor.
            - Composite entries (e.g., 'IV Pump + Propaq + Ventilator') indicate concurrent interactions
              with multiple devices, so the same time interval is assigned to all corresponding devices.
            - The patient side (CHUCK vs DAVE) is determined per-row from the Patient/Patient_Name column
              when available; if it cannot be inferred from the row, we fall back to the sheet-level
              `patient_hint`. If neither is available, we assign the interval to both sides and rely on
              later grouping.
            """
            lbl = str(raw_label or "").strip()
            s = lbl.lower()
            if not s:
                return []

            # Split on '+', ',', '/', ';' to capture composite interactions
            tokens = [t.strip() for t in re.split(r"[+,/;]+", s) if t.strip()]
            if not tokens:
                tokens = [s]

            devices = set()
            for t in tokens:
                if "vent" in t:  # ventilator / MV
                    devices.add("MV")
                if "propaq" in t or "prop" in t or "monitor" in t:
                    devices.add("PROPAQ")
                if "iv" in t:    # any 'iv' token → IV subsystem (pump + lines + meds)
                    devices.add("IV")

            if not devices:
                return []

            # Determine side for this specific row, if possible.
            side = None
            ps = str(patient_side or "").lower().strip()
            if "chuck" in ps:
                side = "CHUCK"
            elif "dave" in ps:
                side = "DAVE"
            else:
                # Fall back to sheet-level hint if row-level name is not informative.
                if patient_hint == "CHUCK":
                    side = "CHUCK"
                elif patient_hint == "DAVE":
                    side = "DAVE"

            labels = set()

            def add_side(side_prefix: str):
                """Assign this interval to canonical device labels on a given side.

                We intentionally collapse all IV-related interactions on a side into a
                single label (e.g., CHUCK_IV, DAVE_IV) because the annotations do not
                reliably distinguish between multiple IV pumps (IV1 vs IV2).
                """
                # IV subsystem → single grouped label per side
                if "IV" in devices:
                    labels.add(f"{side_prefix}_IV")
                # Mechanical ventilator
                if "MV" in devices:
                    labels.add(f"{side_prefix}_MV")
                # Propaq monitor
                if "PROPAQ" in devices:
                    labels.add(f"{side_prefix}_PROPAQ")

            if side is not None:
                # Row-level side (or sheet-level hint) is known.
                add_side(side)
            else:
                # No side information → assign to both; downstream grouping (OBJ_GROUP)
                # will aggregate by device family.
                add_side("DAVE")
                add_side("CHUCK")

            return sorted(labels)

        def adjust_for_view(name: str):
            """View-specific remapping is not needed for grouped GT labels.

            GT labels are defined at the side+device-family level
            (e.g., CHUCK_IV, DAVE_MV), so they stay the same across views.
            """
            if not name:
                return None
            return name

        if start_col and end_col:
            def clean_time(v):
                s = str(v).strip()
                if s in {"", ".", "..", "-", "nan", "NaN", "None"}:
                    return np.nan
                return v

            df[start_col] = df[start_col].apply(clean_time)
            df[end_col]   = df[end_col].apply(clean_time)
            if first_col:
                df[first_col] = df[first_col].apply(clean_time)
                # capture first interaction BEFORE dropping NaNs on start/end
                for v in df[first_col].dropna():
                    sec = hms_to_seconds(v)
                    if sec is None:
                        continue
                    fr = int(round(sec * fps))
                    if first_frame_local is None or fr < first_frame_local:
                        first_frame_local = fr
            if occlusion_col:
                occ_mask = ~df[occlusion_col].astype(str).str.strip().str.upper().eq("Y")
                if debug:
                    removed = int((~occ_mask).sum())
                    print(f"[GT debug] Excluding occluded GT rows: removed={removed} kept={int(occ_mask.sum())}")
                df = df[occ_mask].copy()
            df = df.dropna(subset=[start_col, end_col]).copy()
            if debug:
                print(f"[GT debug] Sample times (first 5 rows after cleaning):")
                for _, r in df.head(5).iterrows():
                    print(f"  start={r[start_col]!r}, end={r[end_col]!r}, first={r[first_col] if first_col else None}, interaction={r[type_col] if type_col else None}")
                print(f"[GT debug] Rows after time dropna: {len(df)}")
            for _, r in df.iterrows():
                raw_label = str(r[type_col]).strip() if type_col else ""
                row_patient = r[patient_col] if patient_col and patient_col in r.index else None
                canonical_list = classify_interaction(raw_label, patient_side=row_patient)
                adjusted_labels = []
                for c in canonical_list:
                    adj = adjust_for_view(c)
                    adjusted_labels.append(adj or c)
                labels_for_row = adjusted_labels or ([raw_label] if raw_label else [])
                s_sec = hms_to_seconds(r[start_col]); e_sec = hms_to_seconds(r[end_col])
                if s_sec is None or e_sec is None:
                    continue  # skip rows with missing/invalid times instead of crashing
                s_fr  = int(round(s_sec * fps)); e_fr = int(round(e_sec * fps))
                if e_fr >= s_fr:
                    for label in labels_for_row:
                        out.setdefault(label, []).append((s_fr, e_fr))
        # Deduplicate identical (start_frame, end_frame) intervals per label.
        # This guards against duplicate annotation rows for the same
        # (Group, Patient, Interaction_Type, Start, End) combination.
        for k, intervals in list(out.items()):
            if not intervals:
                continue
            # Remove exact duplicates and sort by (start, end)
            uniq = sorted(set((int(s), int(e)) for (s, e) in intervals))
            out[k] = [(s, e) for (s, e) in uniq]

        if debug:
            print(f"[GT debug] Parsed {sum(len(v) for v in out.values())} intervals from this sheet.")
        return out, first_frame_local

    intervals = {}
    first_interaction_frame = None
    total_rows_considered = 0
    suffix = Path(gt_path).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        sheets = pd.read_excel(gt_path, sheet_name=None)
        for sheet_nm, df in sheets.items():
            df = df.dropna(how="all")  # strip phantom rows from Excel formatting
            if debug:
                print(f"\n[GT debug] Processing sheet: {sheet_nm}")
            total_rows_considered += len(df)
            part, first_frame_local = parse_rows(df)
            for k, v in part.items():
                intervals.setdefault(k, []).extend(v)
            if first_frame_local is not None:
                if first_interaction_frame is None or first_frame_local < first_interaction_frame:
                    first_interaction_frame = first_frame_local
    else:
        try:
            df = pd.read_csv(gt_path)
            total_rows_considered += len(df)
            intervals, first_frame_local = parse_rows(df)
            if first_frame_local is not None:
                first_interaction_frame = first_frame_local
        except Exception:
            intervals = {}
    if not intervals and not debug:
        try:
            uniq_patients = set()
            if suffix in [".xlsx", ".xls"]:
                for df in pd.read_excel(gt_path, sheet_name=None).values():
                    if df.empty:
                        continue
                    uniq_patients.update(df.get("Patient", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                    uniq_patients.update(df.get("Patient_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
            else:
                df = pd.read_csv(gt_path)
                uniq_patients.update(df.get("Patient", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
                uniq_patients.update(df.get("Patient_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
            print(f"Warning: GT intervals parsed = 0 (rows considered={total_rows_considered}). Patient tokens seen: {sorted(uniq_patients)}. Use --debug_gt for more details.")
        except Exception:
            print(f"Warning: GT intervals parsed = 0 (rows considered={total_rows_considered}). Use --debug_gt for more details.")
    return intervals, first_interaction_frame, patient_hint

 # ---------- threshold helpers (deprecated: thresholds now come solely from DEFAULT_OBJ_THRESH) ----------

# ---------- core ----------
def predicted_intervals_only(
    pred_csv_path,
    obj_id=101, verb_id=117,
    sigma=None,
    merge_gap_s=6, min_dur_s=1,
    fps_override=None,
    show=True, title="",
    gt_csv_path=None, gt_interaction=None,
    sheet_token_override=None,
    video_path=None, demo_outdir=None, demo_fps=None,
    all_objs=False,
    enable_video_demos=True,
    box_format="xyxy",
    box_base_w=None, box_base_h=None,
    debug_gt=False,
    emit_summary=True,
    return_details=False,
    apply_iv_propaq_resolution=True,
):
    # If video demos are disabled, ignore any provided video_path to avoid
    # accidentally writing video outputs or touching the file.
    if not enable_video_demos:
        print("Video demos disabled: will only compute plots/F1 (no video output).")
        video_path = None

    with open(pred_csv_path, 'r', errors='replace') as _f:
        _total_lines = sum(1 for _ in _f) - 1  # subtract header
    df = pd.read_csv(pred_csv_path, on_bad_lines='skip', engine='python')
    _skipped = _total_lines - len(df)
    if _skipped > 0:
        print(f"[WARN] {pred_csv_path}: skipped {_skipped} of {_total_lines} rows due to parse errors ({100*_skipped/_total_lines:.2f}% lost)")
    # Determine obj_ids based on all_objs flag
    if all_objs:
        obj_ids = sorted(df["object_class"].dropna().unique().astype(int))
    else:
        # support multiple object ids
        if not isinstance(obj_id, (list, tuple, np.ndarray)):
            obj_ids = [obj_id]
        else:
            obj_ids = obj_id
    # Per-object HOI thresholds and sigmas come from hard-coded defaults; sigma can be overridden globally via CLI.
    print("Per-object HOI thresholds and sigmas in use (hard-coded defaults; optional global sigma override via --sigma):")
    for oid in obj_ids:
        oid_int = int(oid)
        thresh_obj = DEFAULT_OBJ_THRESH.get(oid_int, 0.25)
        sigma_obj = int(sigma) if sigma is not None else DEFAULT_OBJ_SIGMA.get(oid_int, 3)
        print(f"  object_class={oid_int} ({oid_to_label(oid_int)}): thresh={thresh_obj}, sigma={sigma_obj}")
    fps = float(fps_override) if fps_override else infer_fps(df, fallback=4.0)
    print(f"FPS={fps:.3f}")

    # FPS for plotting (model) and for demo writing (video)
    eff_fps = float(demo_fps) if demo_fps else fps

    # Optional box scaling (e.g., if predictions were made on resized frames)
    scale_xy = None
    video_dims = None
    if video_path:
        src_probe = cv2.VideoCapture(video_path)
        if src_probe.isOpened():
            vw = int(src_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(src_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_dims = (vw, vh)
        src_probe.release()

    # Compute a quick max box extent (on a subset for speed) for diagnostics
    max_x2 = max_y2 = 0.0
    for boxval in df["object_box"].dropna().head(5000):
        b = parse_box_xywh(boxval, fmt=box_format)
        if b:
            max_x2 = max(max_x2, b[0] + b[2])
            max_y2 = max(max_y2, b[1] + b[3])
    if video_dims:
        vw, vh = video_dims
        if box_base_w and box_base_h:
            scale_xy = (vw / float(box_base_w), vh / float(box_base_h))
            print(f"Scaling boxes by ({scale_xy[0]:.3f}, {scale_xy[1]:.3f}) to match video size {vw}x{vh} from base {box_base_w}x{box_base_h}.")
        else:
            # warn if boxes seem much smaller than the video frame
            if (max_x2 > 0 and vw > 0 and max_x2 < 0.7 * vw) or (max_y2 > 0 and vh > 0 and max_y2 < 0.7 * vh):
                print(f"Warning: box extents (max x2={max_x2:.1f}, y2={max_y2:.1f}) are notably smaller than video size ({vw}x{vh}). If preds were made on resized frames, set --box_base_w/--box_base_h to rescale.")

    # Infer GT interaction names from requested object ids if not explicitly provided
    inferred_interactions = None
    if gt_interaction is None:
        try:
            req_ids = set(int(x) for x in obj_ids)
            inferred_interactions = [name for name, oid in MAP_INTERACTION_TO_OBJ.items() if int(oid) in req_ids]
            # normalize to None if empty
            if not inferred_interactions:
                inferred_interactions = None
        except Exception:
            inferred_interactions = None
    else:
        inferred_interactions = [str(gt_interaction)]

    processed = {}
    first_interaction_frame = None  # will be populated from GT if available
    short_pred_segments = []
    for oid in obj_ids:
        oid_int = int(oid)
        # Per-object threshold from DEFAULT_OBJ_THRESH only; sigma uses per-object defaults with optional global override
        thresh_obj = DEFAULT_OBJ_THRESH.get(oid_int, 0.25)
        sigma_obj  = int(sigma) if sigma is not None else DEFAULT_OBJ_SIGMA.get(oid_int, 3)

        sel = df[(df["object_class"] == oid) & (df["verb_class"] == verb_id)].copy()
        if sel.empty:
            print(f"Warning: no rows found for (object={oid}, verb={verb_id}) in {pred_csv_path}; skipping.")
            continue

        sel["frame_id"] = parse_frames(sel["filename"])
        sel = sel.dropna(subset=["frame_id"]).astype({"frame_id":"int"})
        # if multiple rows per frame → take max HOI score per frame
        sel = sel.loc[sel.groupby("frame_id")["score"].idxmax()].sort_values("frame_id").set_index("frame_id")

        # Gaussian smoothing (use a non-reflective boundary mode to avoid edge artifacts)
        hoi = gaussian_filter1d(sel["score"].values, sigma_obj, mode="nearest")
        verb_sm = gaussian_filter1d(sel["verb_scores_index_decoder"].values, sigma_obj, mode="nearest")
        obj_sm  = gaussian_filter1d(sel["obj_scores"].apply(extract_obj_score).values, sigma_obj, mode="nearest")

        keep_idx = np.where(hoi > thresh_obj)[0]
        frames = sel.index.values
        raw_runs = [(frames[s], frames[e]) for (s, e) in contiguous_runs(keep_idx)]

        mgap = int(round(merge_gap_s * fps))
        mind = int(round(min_dur_s * fps))
        # Step 1: remove short raw segments first
        short_runs = [(s, e) for (s, e) in raw_runs if (e - s) < mind]
        for (s, e) in short_runs:
            short_pred_segments.append({
                "object_class": oid_int,
                "object_label": oid_to_label(oid_int),
                "object_group": OBJ_GROUP.get(oid_int, oid_to_label(oid_int)),
                "start_frame": int(s),
                "end_frame": int(e),
                "start_time": frame_to_mmss(int(s), fps),
                "end_time": frame_to_mmss(int(e), fps),
                "duration_frames": int(e - s + 1),
                "duration_seconds": _interval_duration_seconds(int(s), int(e), fps),
                "threshold_used": float(thresh_obj),
                "sigma_used": int(sigma_obj),
            })
        long_runs = [(s, e) for (s, e) in raw_runs if (e - s) >= mind]
        # Step 2: merge remaining long segments whose gap is <= mgap
        merged = []
        if long_runs:
            ps, pe = long_runs[0]
            for cs, ce in long_runs[1:]:
                if cs - pe <= mgap: pe = ce
                else: merged.append((ps, pe)); ps, pe = cs, ce
            merged.append((ps, pe))

        # per-frame box maps
        frame_to_box  = build_frame_to_box_map(sel, "object_box", box_format=box_format, scale=scale_xy)
        frame_to_subj = build_frame_to_box_map(sel, "subject_box", fallback_cols=("human_box",), box_format=box_format, scale=scale_xy)
        processed[oid] = {
            "sel": sel, "frames": frames,
            "hoi": hoi, "verb": verb_sm, "obj": obj_sm,
            "merged": merged,
            "frame_to_box": frame_to_box,
            "frame_to_subj": frame_to_subj,
            "thresh": thresh_obj,
            "sigma": sigma_obj,
        }

    # ---- load GT intervals (per Interaction_Type) if provided ----
    gt_by_label, first_interaction_frame, patient_hint = load_gt_annotations(
        gt_csv_path, fps, video_path=video_path, pred_path=pred_csv_path,
        debug=debug_gt, sheet_token_override=sheet_token_override
    )
    gt_by_label_all = {lbl: [(int(gs), int(ge)) for (gs, ge) in spans] for lbl, spans in (gt_by_label or {}).items()}

    # Step 1: move short GT intervals to uncertain CSV; step 2: merge remaining neighbours.
    short_gt_segments = []
    if gt_by_label:
        merge_gap_frames_gt = int(round(merge_gap_s * fps))
        min_dur_frames_gt = int(round(min_dur_s * fps))
        for lbl, spans in list(gt_by_label.items()):
            all_spans = [(int(gs), int(ge)) for (gs, ge) in spans]
            # Step 1: remove short spans first → uncertain CSV
            removed = [(gs, ge) for (gs, ge) in all_spans if (ge - gs) < min_dur_frames_gt]
            for (gs, ge) in removed:
                short_gt_segments.append({
                    "object_label": str(lbl),
                    "start_frame": int(gs),
                    "end_frame": int(ge),
                    "start_time": frame_to_mmss(int(gs), fps),
                    "end_time": frame_to_mmss(int(ge), fps),
                    "duration_frames": int(ge - gs + 1),
                    "duration_seconds": _interval_duration_seconds(int(gs), int(ge), fps),
                })
            long_spans = [(gs, ge) for (gs, ge) in all_spans if (ge - gs) >= min_dur_frames_gt]
            # Step 2: merge remaining long spans whose gap is <= merge_gap_frames_gt
            filtered = merge_sorted_intervals(
                long_spans,
                merge_gap_frames=merge_gap_frames_gt,
                min_dur_frames=1,
            )
            if filtered:
                gt_by_label[lbl] = filtered
            else:
                del gt_by_label[lbl]
    # Print GT intervals for transparency
    if gt_by_label:
        print("GT intervals (frames and mm:ss):")
        for lbl, spans in gt_by_label.items():
            for (gs, ge) in spans:
                print(f"  {lbl}: {gs}-{ge}  ({frame_to_mmss(gs, fps)}–{frame_to_mmss(ge, fps)})")
    else:
        print("GT intervals: none parsed.")

    # Clip predicted intervals to start at first_interaction_frame if provided
    if first_interaction_frame is not None and first_interaction_frame > 0:
        for oid, data in processed.items():
            merged = data.get("merged", [])
            merged = [(max(s, first_interaction_frame), e) for (s, e) in merged if e >= first_interaction_frame]
            data["merged"] = merged

    if apply_iv_propaq_resolution:
        _resolve_iv_propaq_conflicts_predicted(processed)

    save_processed_plots(
        processed,
        gt_by_label,
        fps,
        demo_outdir,
        first_interaction_frame=first_interaction_frame,
        title=title,
        plot_prefix="",
    )

    if demo_outdir:
        view_id = get_camera_view(video_path) or get_camera_view(pred_csv_path) or Path(str(video_path or pred_csv_path)).stem
        video_id = Path(str(video_path or pred_csv_path)).stem
        _write_short_interval_csv(
            Path(demo_outdir) / "lessthan5secon_uncertrain_segments_GT.csv",
            [
                {
                    "simulation_id": _simulation_key_from_token(sheet_token_override) if sheet_token_override else "single_view",
                    "view_id": view_id,
                    "video_id": video_id,
                    "gt_sheet_token": sheet_token_override,
                    "object_label": row.get("object_label"),
                    "start_frame": row.get("start_frame"),
                    "end_frame": row.get("end_frame"),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "duration_frames": row.get("duration_frames"),
                    "duration_seconds": row.get("duration_seconds"),
                }
                for row in short_gt_segments
            ],
            [
                "simulation_id",
                "view_id",
                "video_id",
                "gt_sheet_token",
                "object_label",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "duration_frames",
                "duration_seconds",
            ],
        )
        _write_short_interval_csv(
            Path(demo_outdir) / "lessthan5secon_uncertrain_segments_predicted.csv",
            [
                {
                    "simulation_id": _simulation_key_from_token(sheet_token_override) if sheet_token_override else "single_view",
                    "view_id": view_id,
                    "video_id": video_id,
                    "source_pred_csv": str(pred_csv_path),
                    "object_class": row.get("object_class"),
                    "object_label": row.get("object_label"),
                    "object_group": row.get("object_group"),
                    "start_frame": row.get("start_frame"),
                    "end_frame": row.get("end_frame"),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                    "duration_frames": row.get("duration_frames"),
                    "duration_seconds": row.get("duration_seconds"),
                    "threshold_used": row.get("threshold_used"),
                    "sigma_used": row.get("sigma_used"),
                }
                for row in short_pred_segments
            ],
            [
                "simulation_id",
                "view_id",
                "video_id",
                "source_pred_csv",
                "object_class",
                "object_label",
                "object_group",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "duration_frames",
                "duration_seconds",
                "threshold_used",
                "sigma_used",
            ],
        )

    grouped_pred, grouped_gt = aggregate_by_group(processed, gt_by_label, start_frame=first_interaction_frame)
    if grouped_gt:
        print("GT intervals (grouped):")
        for lbl, spans in grouped_gt.items():
            for (gs, ge) in spans:
                print(f"  {lbl}: {gs}-{ge}  ({frame_to_mmss(gs, fps)}–{frame_to_mmss(ge, fps)})")

    # ---- optional: create per-object demo videos with GT + predicted overlays ----
    if not enable_video_demos and video_path:
        print("Skipping demo video rendering (--no_video_demos enabled).")
    if enable_video_demos and video_path:
        src = cv2.VideoCapture(video_path)
        if not src.isOpened():
            print(f"Warning: cannot open video: {video_path}")
        else:
            # Prepare output directory
            out_dir = Path(demo_outdir) if demo_outdir else Path(pred_csv_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            width  = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fps = src.get(cv2.CAP_PROP_FPS)
            vid_fps = float(raw_fps if raw_fps and raw_fps > 0 else eff_fps)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required to create H.264 .mp4 demo videos")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            def in_any_range(fid, ranges):
                for s, e in ranges:
                    if s <= fid <= e:
                        return True
                return False

            # Determine the minimum prediction frame id across all requested objects
            try:
                min_pred_frame = min(int(d["sel"].index.min()) for d in processed.values() if len(d["sel"].index) > 0)
            except Exception:
                min_pred_frame = 1
            pred_fps = fps  # prediction/extracted-frames fps (already inferred above)

            # For each requested object, write a separate demo
            base = Path(video_path).stem
            for oid, data in processed.items():
                thresh_obj = data.get("thresh")
                name = OBJ_TO_INTERACTION.get(int(oid), f"obj{oid}")
                gt_frame_ranges = gt_by_label.get(name, [])
                # Generate initial plot image (static, for shape)
                plot_img0 = render_hoi_plot_image(data, fps, thresh_obj, width, current_frame=-1, gt_intervals=gt_frame_ranges)
                plot_h0 = plot_img0.shape[0]
                stacked_height = plot_h0 + height
                out_path = out_dir / f"{base}_{name}_demo.mp4"
                tmp_path = out_dir / f"{base}_{name}_demo.tmp_mp4v.mp4"
                writer = cv2.VideoWriter(str(tmp_path), fourcc, vid_fps, (width, stacked_height))
                if not writer.isOpened():
                    print(f"Warning: cannot open writer for {tmp_path}")
                    continue

                pred_ranges = data["merged"]
                frame_to_box = data["frame_to_box"]
                frame_to_subj = data.get("frame_to_subj", {})

                v_idx = 0
                src.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ok, frame = src.read()
                    if not ok: 
                        break
                    # time (sec) at this video frame
                    t_sec = v_idx / max(vid_fps, 1e-6)
                    # corresponding predicted frame id (nearest)
                    fid_pred = int(round(min_pred_frame + t_sec * pred_fps))
                    v_idx += 1

                    # fetch boxes for this predicted frame id; try ±1 as fallback
                    obj_box  = frame_to_box.get(fid_pred)
                    subj_box = frame_to_subj.get(fid_pred)
                    if obj_box is None and (fid_pred-1) in frame_to_box:
                        obj_box = frame_to_box[fid_pred-1]
                    if obj_box is None and (fid_pred+1) in frame_to_box:
                        obj_box = frame_to_box[fid_pred+1]
                    if subj_box is None and (fid_pred-1) in frame_to_subj:
                        subj_box = frame_to_subj[fid_pred-1]
                    if subj_box is None and (fid_pred+1) in frame_to_subj:
                        subj_box = frame_to_subj[fid_pred+1]

                    # --- Overlay mode: show Pred/GT banner and draw subject/object bounding boxes when available ---
                    pred_active = in_any_range(fid_pred, pred_ranges)
                    gt_active   = in_any_range(fid_pred, gt_frame_ranges)

                    # Draw subject (human) box only when HOI is active (pred > threshold); subject box in cyan
                    if pred_active and subj_box is not None:
                        draw_box(frame, subj_box, (255, 255, 0), thickness=2, label="H")
                    # Draw object box whenever available (object is static equipment)
                    if obj_box is not None:
                        # Use a more specific label if available
                        obj_label = OBJ_TO_INTERACTION.get(int(oid), f"obj{oid}")
                        draw_box(frame, obj_box, (0, 255, 255), thickness=2, label=obj_label)

                    # Top banner
                    banner_h = 40
                    cv2.rectangle(frame, (0, 0), (width, banner_h), (0, 0, 0), -1)
                    name_disp = OBJ_TO_INTERACTION.get(int(oid), f"obj{oid}")
                    pred_txt = f"Pred: {name_disp} {'ON' if pred_active else 'OFF'}"
                    gt_txt   = f"GT: {name_disp} {'ON' if gt_active else 'OFF'}"
                    cv2.putText(frame, pred_txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if pred_active else (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, gt_txt, (int(width * 0.45), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if gt_active else (0, 0, 255), 2, cv2.LINE_AA)

                    # Bottom HUD (diagnostics)
                    cv2.putText(
                        frame,
                        f"pred_frame={fid_pred}  srcFPS={vid_fps:.2f}  predFPS={pred_fps:.2f}",
                        (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
                    )

                    # --- Generate plot image for this frame ---
                    plot_img = render_hoi_plot_image(data, fps, thresh_obj, width, current_frame=fid_pred, gt_intervals=gt_frame_ranges)
                    if plot_img.shape[0] != plot_h0:
                        plot_img = cv2.resize(plot_img, (width, plot_h0), interpolation=cv2.INTER_AREA)

                    # --- Stack plot image above the video frame ---
                    stacked = np.vstack((plot_img, frame))
                    if stacked.shape[1] != width or stacked.shape[0] != stacked_height:
                        print("Size mismatch when writing video: "
                              f"expected (h, w)=({stacked_height}, {width}), "
                              f"got {stacked.shape}. Skipping remaining frames for this object.")
                        break
                    writer.write(stacked)

                writer.release()
                subprocess.run(
                    [
                        ffmpeg, "-y",
                        "-i", str(tmp_path),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        str(out_path),
                    ],
                    check=True,
                )
                tmp_path.unlink(missing_ok=True)
                print(f"✅ Demo written: {out_path}")

            src.release()

    # ---- Compute frame-level F1 for each object ----
    # Infer camera prefix from file names if possible
    cam_prefix = None
    if patient_hint:
        cam_prefix = patient_hint
    elif "chuck" in pred_csv_path.lower() or (video_path and "chuck" in video_path.lower()):
        cam_prefix = "CHUCK"
    elif "dave" in pred_csv_path.lower() or (video_path and "dave" in video_path.lower()):
        cam_prefix = "DAVE"
    view_key = get_camera_view(pred_csv_path) or get_camera_view(video_path)
    available_ids = set(camera_view_boxes[view_key].keys()) if view_key and view_key in camera_view_boxes else None
    summary = emit_metric_summary(
        "",
        processed,
        gt_by_label,
        OBJ_TO_INTERACTION,
        cam_prefix=cam_prefix,
        available_ids=available_ids,
        start_frame=first_interaction_frame,
        fps=fps,
    ) if emit_summary else None

    segs = []
    for oid, data in processed.items():
        for (s, e) in data["merged"]:
            segs.append({"object_class": oid,
                         "object_label": oid_to_label(oid),
                         "start_frame": s, "end_frame": e,
                         "start_time": frame_to_mmss(s, fps),
                         "end_time":   frame_to_mmss(e, fps)})
    if return_details:
        return {
            "segs": segs,
            "fps": fps,
            "processed": processed,
            "gt_by_label": gt_by_label,
            "gt_by_label_all": gt_by_label_all,
            "first_interaction_frame": first_interaction_frame,
            "patient_hint": patient_hint,
            "view_key": view_key,
            "available_ids": available_ids,
            "cam_prefix": cam_prefix,
            "summary": summary,
            "short_gt_segments": short_gt_segments,
            "short_pred_segments": short_pred_segments,
        }
    return segs, fps

def predicted_intervals_multi(
    pred_csv_paths,
    video_paths,
    gt_csv_path=None,
    gt_sheet_tokens=None,
    *,
    obj_id=101,
    verb_id=117,
    sigma=None,
    merge_gap_s=4,
    min_dur_s=4,
    fps_override=None,
    gt_interaction=None,
    demo_outdir=None,
    all_objs=False,
    debug_gt=False,
    return_summary=False,
):
    if not pred_csv_paths or not video_paths:
        raise ValueError("pred_csv_paths and video_paths are required for multi-view mode.")
    if len(pred_csv_paths) != len(video_paths):
        raise ValueError("pred_csv_paths and video_paths must have the same length.")
    if gt_sheet_tokens is None:
        gt_sheet_tokens = [None] * len(pred_csv_paths)
    if len(gt_sheet_tokens) != len(pred_csv_paths):
        raise ValueError("gt_sheet_tokens must have the same length as pred_csv_paths when provided.")

    def _infer_view_id(path_like):
        s = str(path_like or "")
        # Prefer explicit tokens commonly used in this project
        m = re.search(r"(PAN[_\-]?V[12]|CAM16[_\-]?V[12]|OLD[_\-]?PAN|NEW[_\-]?PAN|OLD[_\-]?CAM|NEW[_\-]?CAM)", s, flags=re.I)
        if m:
            return m.group(1).upper().replace("-", "_")
        return Path(s).stem

    details_list = []
    # Per-view metadata aligned with details_list indices.
    view_meta = []
    failed_views = []
    for idx, (pred_csv_path, video_path, gt_sheet_token) in enumerate(zip(pred_csv_paths, video_paths, gt_sheet_tokens), start=1):
        view_id = _infer_view_id(video_path)
        video_stem = Path(video_path).stem
        meta = {
            "view_id": view_id,
            "video_stem": video_stem,
            "pred_csv_path": str(pred_csv_path),
            "video_path": str(video_path),
            "gt_sheet_token": gt_sheet_token,
        }
        try:
            details = predicted_intervals_only(
                pred_csv_path,
                obj_id=obj_id,
                verb_id=verb_id,
                sigma=sigma,
                merge_gap_s=merge_gap_s,
                min_dur_s=min_dur_s,
                fps_override=fps_override,
                show=False,
                title=f"[{pred_csv_path}]",
                gt_csv_path=gt_csv_path,
                gt_interaction=gt_interaction,
                sheet_token_override=gt_sheet_token,
                video_path=video_path,
                demo_outdir=None,
                demo_fps=None,
                all_objs=all_objs,
                enable_video_demos=False,
                box_format="xyxy",
                box_base_w=None,
                box_base_h=None,
                debug_gt=debug_gt,
                emit_summary=False,
                return_details=True,
                apply_iv_propaq_resolution=False,
            )
            view_meta.append(meta)
            details_list.append(details)
            if demo_outdir:
                save_processed_plots(
                    details.get("processed", {}),
                    details.get("gt_by_label", {}),
                    details["fps"],
                    demo_outdir,
                    first_interaction_frame=details.get("first_interaction_frame"),
                    title=f"[{pred_csv_path}]",
                    plot_prefix=f"view{idx}_",
                )
        except Exception as e:
            import traceback
            failed_views.append({**meta, "error": str(e)})
            print("\n[ERROR] Failed processing one multi-view input; skipping this view and continuing.")
            print(f"  pred_csv_path: {pred_csv_path}")
            print(f"  video_path:    {video_path}")
            print(f"  gt_sheet:      {gt_sheet_token}")
            print(f"  error:         {e}")
            print(traceback.format_exc())
            continue

    if failed_views:
        print(f"\n[WARN] Skipped {len(failed_views)} failing view(s) in this simulation group.")
        for fv in failed_views:
            print(f"  - {fv['view_id']}: pred={fv['pred_csv_path']} | video={fv['video_path']} | gt={fv['gt_sheet_token']}")

    if not details_list:
        raise RuntimeError("All views failed for this simulation group; no valid inputs left to combine.")

    fps_values = [float(d["fps"]) for d in details_list if d.get("fps") is not None]
    fps = fps_values[0] if fps_values else 4.0
    if fps_values and any(abs(v - fps) > 1e-6 for v in fps_values[1:]):
        print(f"Warning: FPS differs across runs {fps_values}; using {fps:.3f} for combined metrics.")

    mgap = int(round(float(merge_gap_s) * float(fps)))
    mind = int(round(float(min_dur_s) * float(fps)))
    # Step 1: combine spans across views (overlap-only merge, no gap filling yet)
    gt_by_label_all = merge_gt_dicts(
        [d.get("gt_by_label_all", {}) for d in details_list],
        merge_gap_frames=0,
        min_dur_frames=1,
    )
    short_gt_rows_fused = []
    gt_by_label = {}
    for lbl, spans in gt_by_label_all.items():
        # Step 2: remove short spans first → uncertain CSV
        short_spans = [(gs, ge) for (gs, ge) in spans if int(ge - gs) < mind]
        for (gs, ge) in short_spans:
            short_gt_rows_fused.append({
                "simulation_id": None,  # filled later
                "view_id": view_ids_joined if 'view_ids_joined' in locals() else None,
                "video_id": None,  # filled later
                "gt_sheet_token": "|".join([str(t) for t in gt_sheet_tokens if t is not None]),
                "object_label": str(lbl),
                "start_frame": int(gs),
                "end_frame": int(ge),
                "start_time": frame_to_mmss(int(gs), fps),
                "end_time": frame_to_mmss(int(ge), fps),
                "duration_frames": int(ge - gs + 1),
                "duration_seconds": _interval_duration_seconds(int(gs), int(ge), fps),
            })
        long_spans = [(gs, ge) for (gs, ge) in spans if int(ge - gs) >= mind]
        # Step 3: merge remaining long spans whose gap is <= mgap
        kept = merge_sorted_intervals(long_spans, merge_gap_frames=mgap, min_dur_frames=1)
        if kept:
            gt_by_label[lbl] = kept
    if gt_by_label:
        print("Combined GT intervals (frames and mm:ss):")
        for lbl, spans in gt_by_label.items():
            for (gs, ge) in spans:
                print(f"  {lbl}: {gs}-{ge}  ({frame_to_mmss(gs, fps)}–{frame_to_mmss(ge, fps)})")
    else:
        print("Combined GT intervals: none parsed.")

    first_candidates = [d.get("first_interaction_frame") for d in details_list if d.get("first_interaction_frame") is not None]
    first_interaction_frame = min(first_candidates) if first_candidates else None
    # Step 1: fuse across views (overlap-only merge, no gap filling yet)
    fused_processed_all = fuse_processed_runs(
        [d.get("processed", {}) for d in details_list],
        fps=fps,
        merge_gap_s=0,
        min_dur_s=0,
    )
    if first_interaction_frame is not None and first_interaction_frame > 0:
        for oid, data in fused_processed_all.items():
            merged = [(max(s, first_interaction_frame), e) for (s, e) in data.get("merged", []) if e >= first_interaction_frame]
            data["merged"] = merge_sorted_intervals(merged, merge_gap_frames=0, min_dur_frames=1)

    fused_processed = {}
    short_pred_rows_fused = []
    for oid, data in fused_processed_all.items():
        # Step 2: remove short segments first → uncertain CSV
        kept = []
        for (s, e) in data.get("merged", []):
            dur_elapsed = int(e - s)
            dur_frames  = int(e - s + 1)
            if dur_elapsed < mind:
                short_pred_rows_fused.append({
                    "simulation_id": None,  # filled later
                    "view_id": None,        # filled later
                    "video_id": None,       # filled later
                    "source_pred_csv": "|".join([str(vm.get("pred_csv_path")) for vm in view_meta]),
                    "object_class": int(oid),
                    "object_label": oid_to_label(int(oid)),
                    "object_group": OBJ_GROUP.get(int(oid), oid_to_label(int(oid))),
                    "start_frame": int(s),
                    "end_frame": int(e),
                    "start_time": frame_to_mmss(int(s), fps),
                    "end_time": frame_to_mmss(int(e), fps),
                    "duration_frames": dur_frames,
                    "duration_seconds": _interval_duration_seconds(int(s), int(e), fps),
                    "threshold_used": float(data.get("thresh")) if data.get("thresh") is not None else None,
                    "sigma_used": int(data.get("sigma")) if data.get("sigma") is not None else None,
                })
            else:
                kept.append((int(s), int(e)))
        # Step 3: merge remaining long segments whose gap is <= mgap
        kept = merge_sorted_intervals(kept, merge_gap_frames=mgap, min_dur_frames=1)
        new_data = dict(data)
        new_data["merged"] = kept
        fused_processed[oid] = new_data

    # Step 4: resolve IV/PROPAQ conflicts — only long (>=5s) IV segments can trim PROPAQ
    _resolve_iv_propaq_conflicts_predicted(fused_processed)

    # Step 5: remove PROPAQ remnants that became <5s after conflict resolution trimming
    propaq_oids = [oid for oid in fused_processed if oid_to_label(int(oid)) in ("DAVE_PROPAQ", "CHUCK_PROPAQ")]
    for oid in propaq_oids:
        data = fused_processed[oid]
        kept, remnants = [], []
        for (s, e) in data.get("merged", []):
            (kept if int(e - s) >= mind else remnants).append((int(s), int(e)))
        for (s, e) in remnants:
            short_pred_rows_fused.append({
                "simulation_id": None,
                "view_id": None,
                "video_id": None,
                "source_pred_csv": "|".join([str(vm.get("pred_csv_path")) for vm in view_meta]),
                "object_class": int(oid),
                "object_label": oid_to_label(int(oid)),
                "object_group": OBJ_GROUP.get(int(oid), oid_to_label(int(oid))),
                "start_frame": int(s),
                "end_frame": int(e),
                "start_time": frame_to_mmss(int(s), fps),
                "end_time": frame_to_mmss(int(e), fps),
                "duration_frames": int(e - s + 1),
                "duration_seconds": _interval_duration_seconds(int(s), int(e), fps),
                "threshold_used": float(data.get("thresh")) if data.get("thresh") is not None else None,
                "sigma_used": int(data.get("sigma")) if data.get("sigma") is not None else None,
            })
        fused_processed[oid]["merged"] = kept

    if demo_outdir:
        save_processed_plots(
            fused_processed,
            gt_by_label,
            fps,
            demo_outdir,
            first_interaction_frame=first_interaction_frame,
            title="[combined]",
            plot_prefix="combined_",
        )

    combined_summary = None
    if gt_by_label:
        combined_summary = emit_metric_summary(
            "Combined ",
            fused_processed,
            gt_by_label,
            OBJ_TO_INTERACTION,
            cam_prefix=None,
            available_ids=None,
            start_frame=first_interaction_frame,
            fps=fps,
        )
    else:
        print("Skipping metric summary because no GT was provided/parsing yielded no GT intervals.")

    segs = []
    # Metadata for this combined multi-view run
    sim_tokens = [t for t in (gt_sheet_tokens or []) if t is not None]
    if sim_tokens:
        sim_keys = sorted({_simulation_key_from_token(t) for t in sim_tokens})
        simulation_id = sim_keys[0] if len(sim_keys) == 1 else "|".join(sim_keys)
    else:
        simulation_id = "combined_simulation"
    video_ids_joined = "|".join([Path(vp).stem for vp in video_paths]) if video_paths else ""
    view_ids_joined = "|".join([_infer_view_id(vp) for vp in video_paths]) if video_paths else ""

    for oid, data in fused_processed.items():
        oid_int = int(oid)
        object_instance_name = OBJID_TO_INSTANCE_NAME.get(oid_int, oid_to_label(oid_int))
        object_group = OBJ_GROUP.get(oid_int, oid_to_label(oid_int))
        threshold_used = data.get("thresh")
        sigma_used = data.get("sigma")
        for (s, e) in data.get("merged", []):
            dur_frames = int(e - s + 1)
            dur_seconds = float(dur_frames / float(fps)) if fps and fps > 0 else None
            segs.append({
                "simulation_id": simulation_id,
                "video_id": video_ids_joined,
                "view_id": view_ids_joined,
                "object_class": oid,
                "object_instance_name": object_instance_name,
                "object_group": object_group,
                "start_frame": s,
                "end_frame": e,
                "start_time": frame_to_mmss(s, fps),
                "end_time": frame_to_mmss(e, fps),
                "duration_frames": dur_frames,
                "duration_seconds": dur_seconds,
                "threshold_used": float(threshold_used) if threshold_used is not None else None,
                "sigma_used": int(sigma_used) if sigma_used is not None else None,
            })
    if demo_outdir:
        out_csv = Path(demo_outdir) / "combined_segments.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Build prediction-only, group-level merged intervals for CSV export.
        group_raw = {}
        for oid, data in fused_processed.items():
            oid_int = int(oid)
            grp = OBJ_GROUP.get(oid_int, oid_to_label(oid_int))
            th = data.get("thresh")
            sg = data.get("sigma")
            for (s, e) in data.get("merged", []):
                group_raw.setdefault(grp, []).append({
                    "s": int(s),
                    "e": int(e),
                    "oid": oid_int,
                    "thr": float(th) if th is not None else None,
                    "sig": int(sg) if sg is not None else None,
                })

        csv_rows = []
        for grp, rows in group_raw.items():
            if not rows:
                continue
            rows_sorted = sorted(rows, key=lambda r: (r["s"], r["e"]))
            cur_s = rows_sorted[0]["s"]
            cur_e = rows_sorted[0]["e"]
            cls_set = {rows_sorted[0]["oid"]}
            thr_set = {rows_sorted[0]["thr"]} if rows_sorted[0]["thr"] is not None else set()
            sig_set = {rows_sorted[0]["sig"]} if rows_sorted[0]["sig"] is not None else set()

            def _collect_person_boxes_for_interval(ss, ee, cls_s, *, all_frames=False):
                candidates = []
                # Collect every available subject/object box from each individual view
                # so downstream person-ID can match frame-by-frame rather than only
                # representative interval samples.
                for oid_i in sorted(cls_s):
                    oid_int = int(oid_i)
                    for v_idx, details in enumerate(details_list):
                        p_view = (details.get("processed", {}) or {}).get(oid_int, {})
                        f2s = p_view.get("frame_to_subj", {}) or {}
                        f2o = p_view.get("frame_to_box", {}) or {}
                        vm = view_meta[v_idx] if v_idx < len(view_meta) else {}
                        for fid in sorted(f2s.keys()):
                            fi = int(fid)
                            if ss <= fi <= ee:
                                box = f2s.get(fid)
                                if box is None:
                                    continue
                                try:
                                    box_xywh = [float(v) for v in box]
                                except Exception:
                                    continue
                                object_box = f2o.get(fid)
                                object_box_xywh = None
                                if object_box is not None:
                                    try:
                                        object_box_xywh = [float(v) for v in object_box]
                                    except Exception:
                                        object_box_xywh = None
                                candidates.append({
                                    "frame_id": fi,
                                    "object_class": oid_int,
                                    "box_xywh": box_xywh,
                                    "object_box_xywh": object_box_xywh,
                                    "view_id": vm.get("view_id"),
                                    "video_stem": vm.get("video_stem"),
                                    "source_pred_csv": vm.get("pred_csv_path"),
                                })
                if all_frames or len(candidates) <= MAX_PERSON_BOXES_PER_INTERVAL:
                    return sorted(candidates, key=lambda x: (x["frame_id"], str(x.get("view_id", "")), x["object_class"]))

                candidates = sorted(candidates, key=lambda x: (x["frame_id"], str(x.get("view_id", "")), x["object_class"]))
                target_frames = np.linspace(int(ss), int(ee), MAX_PERSON_BOXES_PER_INTERVAL)
                selected = []
                used = set()
                for target in target_frames:
                    best_idx = None
                    best_key = None
                    for idx, cand in enumerate(candidates):
                        if idx in used:
                            continue
                        key = (
                            abs(float(cand["frame_id"]) - float(target)),
                            cand["frame_id"],
                            str(cand.get("view_id", "")),
                            cand["object_class"],
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_idx = idx
                    if best_idx is not None:
                        used.add(best_idx)
                        selected.append(candidates[best_idx])
                return sorted(selected, key=lambda x: (x["frame_id"], str(x.get("view_id", "")), x["object_class"]))

            def _flush_segment(ss, ee, cls_s, thr_s, sig_s):
                dur_frames = int(ee - ss + 1)
                dur_seconds = float(dur_frames / float(fps)) if fps and fps > 0 else None
                person_boxes = _collect_person_boxes_for_interval(int(ss), int(ee), cls_s, all_frames=False)
                person_boxes_all_frames = _collect_person_boxes_for_interval(int(ss), int(ee), cls_s, all_frames=True)
                csv_rows.append({
                    "simulation_id": simulation_id,
                    "video_id": video_ids_joined,
                    "object_group": grp,
                    "start_frame": int(ss),
                    "end_frame": int(ee),
                    "start_time": frame_to_mmss(int(ss), fps),
                    "end_time": frame_to_mmss(int(ee), fps),
                    "duration_frames": dur_frames,
                    "duration_seconds": dur_seconds,
                    "source_object_classes": ",".join(map(str, sorted(cls_s))),
                    "thresholds_used": ",".join(map(str, sorted(thr_s))) if thr_s else "",
                    "sigmas_used": ",".join(map(str, sorted(sig_s))) if sig_s else "",
                    "person_bounding_boxes": json.dumps(person_boxes),
                    "person_bounding_boxes_all_frames": json.dumps(person_boxes_all_frames),
                })

            for r in rows_sorted[1:]:
                # Merge overlapping or touching intervals at group level
                if r["s"] <= cur_e + 1:
                    cur_e = max(cur_e, r["e"])
                    cls_set.add(r["oid"])
                    if r["thr"] is not None:
                        thr_set.add(r["thr"])
                    if r["sig"] is not None:
                        sig_set.add(r["sig"])
                else:
                    _flush_segment(cur_s, cur_e, cls_set, thr_set, sig_set)
                    cur_s, cur_e = r["s"], r["e"]
                    cls_set = {r["oid"]}
                    thr_set = {r["thr"]} if r["thr"] is not None else set()
                    sig_set = {r["sig"]} if r["sig"] is not None else set()
            _flush_segment(cur_s, cur_e, cls_set, thr_set, sig_set)

        cols = [
            "simulation_id",
            "video_id",
            "object_group",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "duration_frames",
            "duration_seconds",
            "source_object_classes",
            "thresholds_used",
            "sigmas_used",
            "person_bounding_boxes",
            "person_bounding_boxes_all_frames",
        ]
        pd.DataFrame(csv_rows, columns=cols).to_csv(out_csv, index=False)
        print(f"Saved combined segments → {out_csv}")

        for row in short_gt_rows_fused:
            row["simulation_id"] = simulation_id
            row["view_id"] = view_ids_joined
            row["video_id"] = video_ids_joined
        for row in short_pred_rows_fused:
            row["simulation_id"] = simulation_id
            row["view_id"] = view_ids_joined
            row["video_id"] = video_ids_joined

        _write_short_interval_csv(
            Path(demo_outdir) / "lessthan5secon_uncertrain_segments_GT.csv",
            short_gt_rows_fused,
            [
                "simulation_id",
                "view_id",
                "video_id",
                "gt_sheet_token",
                "object_label",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "duration_frames",
                "duration_seconds",
            ],
        )
        _write_short_interval_csv(
            Path(demo_outdir) / "lessthan5secon_uncertrain_segments_predicted.csv",
            short_pred_rows_fused,
            [
                "simulation_id",
                "view_id",
                "video_id",
                "source_pred_csv",
                "object_class",
                "object_label",
                "object_group",
                "start_frame",
                "end_frame",
                "start_time",
                "end_time",
                "duration_frames",
                "duration_seconds",
                "threshold_used",
                "sigma_used",
            ],
        )
    if return_summary:
        return segs, fps, combined_summary
    return segs, fps

# ---------- grid search ----------
def _parse_float_csv(x):
    return [float(v.strip()) for v in str(x).split(",") if str(v).strip() != ""]

def _parse_int_csv(x):
    return [int(float(v.strip())) for v in str(x).split(",") if str(v).strip() != ""]

def _build_float_range(vmin, vmax, vstep):
    vals = []
    cur = float(vmin)
    vmax = float(vmax)
    vstep = float(vstep)
    if vstep <= 0:
        raise ValueError("step must be > 0")
    # include endpoint with tolerance
    while cur <= vmax + 1e-12:
        vals.append(round(cur, 6))
        cur += vstep
    return vals

def _simulation_key_from_token(tok):
    s = str(tok or "").strip()
    if not s:
        return "unknown_sim"
    # Example: 2024C_Alpha2 -> 2024C_Alpha ; 2024A_Delta1 -> 2024A_Delta
    s2 = re.sub(r"\d+$", "", s).rstrip("_")
    return s2 if s2 else s

def _interval_duration_seconds(start_frame, end_frame, fps):
    dur_frames = int(end_frame - start_frame + 1)
    return float(dur_frames / float(fps)) if fps and fps > 0 else None

def _write_short_interval_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    print(f"Saved short-duration segments → {path}")

def _build_simulation_groups(pred_csvs, video_paths, gt_sheet_tokens):
    groups = {}
    for i, (pc, vp, gt) in enumerate(zip(pred_csvs, video_paths, gt_sheet_tokens)):
        key = _simulation_key_from_token(gt if gt is not None else f"sim_{i}")
        g = groups.setdefault(key, {"pred_csvs": [], "video_paths": [], "gt_sheet_tokens": [], "indices": []})
        g["pred_csvs"].append(pc)
        g["video_paths"].append(vp)
        g["gt_sheet_tokens"].append(gt)
        g["indices"].append(i)
    return groups

def _family_from_group_key(group_key: str):
    s = str(group_key or "").upper()
    if s.endswith("_IV"):
        return "IV"
    if s.endswith("_MV"):
        return "MV"
    if s.endswith("_PROPAQ"):
        return "PROPAQ"
    return None

def _family_balanced_from_summary(summary):
    fam = {"IV": None, "MV": None, "PROPAQ": None}
    if not summary:
        return fam
    by_group = summary.get("f1_group", {}) or {}
    vals = {"IV": [], "MV": [], "PROPAQ": []}
    for gk, gv in by_group.items():
        fam_key = _family_from_group_key(gk)
        if fam_key is None or not isinstance(gv, dict):
            continue
        v = gv.get("balanced")
        if v is not None:
            vals[fam_key].append(float(v))
    for k in fam.keys():
        fam[k] = float(np.mean(vals[k])) if vals[k] else None
    return fam

def _print_cross_sim_family_averages(sim_to_summary, header_prefix=""):
    if not sim_to_summary:
        return
    fam_across = {"IV": [], "MV": [], "PROPAQ": []}
    print(f"{header_prefix}Cross-simulation family macro F1 (balanced pos/neg):")
    for sim_key, summary in sim_to_summary.items():
        fam = _family_balanced_from_summary(summary)
        print(f"  {sim_key}: IV={fam['IV']}  MV={fam['MV']}  PROPAQ={fam['PROPAQ']}")
        for k in fam_across.keys():
            if fam[k] is not None:
                fam_across[k].append(float(fam[k]))
    print(f"{header_prefix}Average across simulations:")
    print(f"  IV={float(np.mean(fam_across['IV'])) if fam_across['IV'] else None}")
    print(f"  MV={float(np.mean(fam_across['MV'])) if fam_across['MV'] else None}")
    print(f"  PROPAQ={float(np.mean(fam_across['PROPAQ'])) if fam_across['PROPAQ'] else None}")

def _cross_sim_group_hier_macro(sim_to_summary):
    """
    Hierarchical aggregation:
      1) For each simulation and group, use group macro F1 (balanced pos/neg).
      2) For each group, average across simulations.
      3) Final score = average of those per-group averages.
    """
    group_to_vals = {}
    for _sim_key, summary in (sim_to_summary or {}).items():
        by_group = (summary or {}).get("f1_group", {}) or {}
        for gk, gv in by_group.items():
            if not isinstance(gv, dict):
                continue
            v = gv.get("balanced")
            if v is None:
                continue
            group_to_vals.setdefault(str(gk), []).append(float(v))
    per_group_avg = {gk: float(np.mean(vals)) for gk, vals in group_to_vals.items() if vals}
    overall = float(np.mean(list(per_group_avg.values()))) if per_group_avg else None
    return per_group_avg, overall

def _scale_default_params(thr_scale=1.0, sigma_scale=1.0, sigma_add=0):
    new_thr = {}
    for oid, t in DEFAULT_OBJ_THRESH.items():
        v = max(0.01, min(0.99, float(t) * float(thr_scale)))
        new_thr[int(oid)] = round(v, 4)

    new_sig = {}
    for oid, s in DEFAULT_OBJ_SIGMA.items():
        v = int(round(float(s) * float(sigma_scale))) + int(sigma_add)
        new_sig[int(oid)] = max(1, v)
    return new_thr, new_sig

def _apply_default_params(thresh_map, sigma_map):
    DEFAULT_OBJ_THRESH.clear()
    DEFAULT_OBJ_THRESH.update({int(k): float(v) for k, v in thresh_map.items()})
    DEFAULT_OBJ_SIGMA.clear()
    DEFAULT_OBJ_SIGMA.update({int(k): int(v) for k, v in sigma_map.items()})

def _load_best_params_from_json(json_path):
    with open(json_path, "r") as f:
        payload = json.load(f)

    best = payload.get("best", payload) if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        raise ValueError(f"--load_best_json expected a dict-like JSON, got: {type(payload)}")

    thresh_map = best.get("best_thresh_map")
    sigma_map = best.get("best_sigma_map")
    if thresh_map is None or sigma_map is None:
        raise ValueError(
            f"--load_best_json missing best_thresh_map/best_sigma_map in: {json_path}"
        )

    _apply_default_params(thresh_map, sigma_map)
    return len(thresh_map), len(sigma_map)

def _load_manual_params_from_json(json_path):
    with open(json_path, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"--load_params_json expected a JSON object, got: {type(payload)}")

    thresh_map = payload.get("thresholds")
    sigma_map = payload.get("sigmas")
    if thresh_map is None or sigma_map is None:
        raise ValueError(
            f"--load_params_json requires keys 'thresholds' and 'sigmas' in: {json_path}"
        )

    _apply_default_params(thresh_map, sigma_map)
    return len(thresh_map), len(sigma_map)

def _constant_default_params(thr_value=0.5, sigma_value=30):
    new_thr = {}
    for oid in DEFAULT_OBJ_THRESH.keys():
        new_thr[int(oid)] = max(0.01, min(0.99, float(thr_value)))
    new_sig = {}
    for oid in DEFAULT_OBJ_SIGMA.keys():
        new_sig[int(oid)] = max(1, int(round(float(sigma_value))))
    return new_thr, new_sig

def _grid_eval_group_candidate_task(task):
    """
    Process-safe evaluation of one candidate map on one group subset.
    """
    thresh_map = task["thresh_map"]
    sigma_map = task["sigma_map"]
    pred_csvs = task["pred_csvs"]
    video_paths = task["video_paths"]
    gt_csv = task["gt_csv"]
    gt_sheet_tokens = task["gt_sheet_tokens"]
    subset_oids = task["subset_oids"]
    grp = task["group_key"]
    obj_id = task["obj_id"]
    verb_id = task["verb_id"]
    merge_gap_s = task["merge_gap_s"]
    min_dur_s = task["min_dur_s"]
    fps_override = task["fps_override"]
    gt_interaction = task["gt_interaction"]
    all_objs = task["all_objs"]
    debug_gt = task["debug_gt"]
    metric_name = task["metric_name"]

    # Preserve process-local defaults
    old_thr = dict(DEFAULT_OBJ_THRESH)
    old_sig = dict(DEFAULT_OBJ_SIGMA)
    try:
        _apply_default_params(thresh_map, sigma_map)
        sim_groups = _build_simulation_groups(pred_csvs, video_paths, gt_sheet_tokens)
        sim_scores = {}
        sim_summaries = {}
        use_subset = subset_oids is not None and len(subset_oids) > 0
        eval_obj_ids = [int(x) for x in subset_oids] if use_subset else obj_id
        eval_all_objs = False if use_subset else all_objs
        for sim_key, gd in sim_groups.items():
            _, _, summary = predicted_intervals_multi(
                gd["pred_csvs"],
                gd["video_paths"],
                gt_csv,
                gd["gt_sheet_tokens"],
                obj_id=eval_obj_ids,
                verb_id=verb_id,
                sigma=None,
                merge_gap_s=merge_gap_s,
                min_dur_s=min_dur_s,
                fps_override=fps_override,
                gt_interaction=gt_interaction,
                demo_outdir=None,
                all_objs=eval_all_objs,
                debug_gt=debug_gt,
                return_summary=True,
            )
            sim_scores[sim_key] = summary.get(metric_name) if summary else None
            sim_summaries[sim_key] = summary
        per_group_avg_across_sims, hierarchical_overall = _cross_sim_group_hier_macro(sim_summaries)
        grp_score = per_group_avg_across_sims.get(grp)
        grp_score = float(grp_score) if grp_score is not None else -1.0
        return {
            "group_score": grp_score,
            "metric_value": float(hierarchical_overall) if hierarchical_overall is not None else None,
            "per_group_avg_across_simulations": per_group_avg_across_sims,
            "per_simulation_metric": sim_scores,
        }
    finally:
        _apply_default_params(old_thr, old_sig)

def run_simple_grid_search(
    pred_csvs,
    video_paths,
    gt_csv,
    gt_sheet_tokens,
    *,
    obj_id,
    verb_id,
    merge_gap_s,
    min_dur_s,
    fps_override=None,
    gt_interaction=None,
    all_objs=False,
    debug_gt=False,
    thr_scales=(0.9, 1.0, 1.1),
    sigma_scales=(0.8, 1.0, 1.2),
    sigma_adds=(-2, 0, 2),
    grid_mode="scale",
    thr_values=None,
    sigma_values=None,
    metric_name="f1_group_macro_all",
    grid_workers=1,
):
    orig_thr = dict(DEFAULT_OBJ_THRESH)
    orig_sig = dict(DEFAULT_OBJ_SIGMA)
    results = []

    def _evaluate_maps(thresh_map, sigma_map, subset_oids=None):
        _apply_default_params(thresh_map, sigma_map)
        sim_groups = _build_simulation_groups(pred_csvs, video_paths, gt_sheet_tokens)
        sim_scores = {}
        sim_summaries = {}
        use_subset = subset_oids is not None and len(subset_oids) > 0
        eval_obj_ids = [int(x) for x in subset_oids] if use_subset else obj_id
        eval_all_objs = False if use_subset else all_objs
        for sim_key, gd in sim_groups.items():
            _, _, summary = predicted_intervals_multi(
                gd["pred_csvs"],
                gd["video_paths"],
                gt_csv,
                gd["gt_sheet_tokens"],
                obj_id=eval_obj_ids,
                verb_id=verb_id,
                sigma=None,
                merge_gap_s=merge_gap_s,
                min_dur_s=min_dur_s,
                fps_override=fps_override,
                gt_interaction=gt_interaction,
                demo_outdir=None,
                all_objs=eval_all_objs,
                debug_gt=debug_gt,
                return_summary=True,
            )
            sim_scores[sim_key] = summary.get(metric_name) if summary else None
            sim_summaries[sim_key] = summary
        per_group_avg_across_sims, hierarchical_overall = _cross_sim_group_hier_macro(sim_summaries)
        fam_vals = {"IV": [], "MV": [], "PROPAQ": []}
        for _sim_key, _summary in sim_summaries.items():
            ff = _family_balanced_from_summary(_summary)
            for kk in fam_vals.keys():
                if ff[kk] is not None:
                    fam_vals[kk].append(float(ff[kk]))
        fam_iv_mv_prop = {
            "IV": float(np.mean(fam_vals["IV"])) if fam_vals["IV"] else None,
            "MV": float(np.mean(fam_vals["MV"])) if fam_vals["MV"] else None,
            "PROPAQ": float(np.mean(fam_vals["PROPAQ"])) if fam_vals["PROPAQ"] else None,
        }
        score = float(hierarchical_overall) if hierarchical_overall is not None else -1.0
        return score, hierarchical_overall, sim_scores, per_group_avg_across_sims, fam_iv_mv_prop, sim_summaries

    def _aggregate_all_sim_metrics(sim_summaries):
        # Frame-level F1 per group aggregated across simulations
        group_f1_vals = {}
        # Interval per-group metrics aggregated across simulations
        group_interval_vals = {}
        for _sim_key, _summary in (sim_summaries or {}).items():
            if not isinstance(_summary, dict):
                continue
            for gk, gv in (_summary.get("f1_group", {}) or {}).items():
                if not isinstance(gv, dict):
                    continue
                row = group_f1_vals.setdefault(str(gk), {"pos": [], "weighted": [], "balanced": []})
                for kk in ("pos", "weighted", "balanced"):
                    vv = gv.get(kk)
                    if vv is not None:
                        row[kk].append(float(vv))
            for gk, mv in (_summary.get("interval_group", {}) or {}).items():
                if not isinstance(mv, dict):
                    continue
                row = group_interval_vals.setdefault(str(gk), {
                    "gt_pred_overlap_ratio": [],
                    "false_interactions_count": [],
                    "false_interactions_count_pct": [],
                    "false_interaction_pred_duration_pct": [],
                    "false_interaction_pred_duration_pct_per_video_duration": [],
                    "model_start_latency_s": [],
                })
                for kk in row.keys():
                    vv = mv.get(kk)
                    if vv is not None:
                        row[kk].append(float(vv))

        f1_group_agg = {}
        for gk, vals in group_f1_vals.items():
            f1_group_agg[gk] = {
                "pos": float(np.mean(vals["pos"])) if vals["pos"] else None,
                "weighted": float(np.mean(vals["weighted"])) if vals["weighted"] else None,
                "balanced": float(np.mean(vals["balanced"])) if vals["balanced"] else None,
            }

        # Macro F1 over groups
        macro_pos = [v["pos"] for v in f1_group_agg.values() if v.get("pos") is not None]
        macro_weighted = [v["weighted"] for v in f1_group_agg.values() if v.get("weighted") is not None]
        macro_balanced = [v["balanced"] for v in f1_group_agg.values() if v.get("balanced") is not None]

        interval_group_agg = {}
        for gk, vals in group_interval_vals.items():
            interval_group_agg[gk] = {
                "gt_pred_overlap_ratio": float(np.mean(vals["gt_pred_overlap_ratio"])) if vals["gt_pred_overlap_ratio"] else None,
                "false_interactions_count": float(np.mean(vals["false_interactions_count"])) if vals["false_interactions_count"] else None,
                "false_interactions_count_pct": float(np.mean(vals["false_interactions_count_pct"])) if vals["false_interactions_count_pct"] else None,
                "false_interaction_pred_duration_pct": float(np.mean(vals["false_interaction_pred_duration_pct"])) if vals["false_interaction_pred_duration_pct"] else None,
                "false_interaction_pred_duration_pct_per_video_duration": float(np.mean(vals["false_interaction_pred_duration_pct_per_video_duration"])) if vals["false_interaction_pred_duration_pct_per_video_duration"] else None,
                "model_start_latency_s": float(np.mean(vals["model_start_latency_s"])) if vals["model_start_latency_s"] else None,
            }

        # Macro interval metrics over groups
        def _macro_from_group(metric_key):
            xs = [v.get(metric_key) for v in interval_group_agg.values() if v.get(metric_key) is not None]
            return float(np.mean(xs)) if xs else None

        interval_macro_overlap = _macro_from_group("gt_pred_overlap_ratio")
        interval_macro_false_count = _macro_from_group("false_interactions_count")
        interval_macro_false_count_pct = _macro_from_group("false_interactions_count_pct")
        interval_macro_false_pred_dur_pct = _macro_from_group("false_interaction_pred_duration_pct")
        interval_macro_false_pred_dur_pct_video = _macro_from_group("false_interaction_pred_duration_pct_per_video_duration")
        interval_macro_latency = _macro_from_group("model_start_latency_s")
        interval_total_false_count = int(round(sum(v.get("false_interactions_count") or 0.0 for v in interval_group_agg.values())))

        return {
            "frame_level_f1_per_group": f1_group_agg,
            "frame_level_macro_f1_group_pos": float(np.mean(macro_pos)) if macro_pos else None,
            "frame_level_macro_f1_group_weighted": float(np.mean(macro_weighted)) if macro_weighted else None,
            "frame_level_macro_f1_group_balanced": float(np.mean(macro_balanced)) if macro_balanced else None,
            "interval_metrics_per_group": interval_group_agg,
            "interval_macro_overlap_ratio_group": interval_macro_overlap,
            "interval_macro_false_interactions_count_group": interval_macro_false_count,
            "interval_macro_false_interactions_count_pct_group": interval_macro_false_count_pct,
            "interval_total_false_interactions_count_group": interval_total_false_count,
            "interval_macro_false_interaction_pred_duration_pct_group": interval_macro_false_pred_dur_pct,
            "interval_macro_false_interaction_pred_duration_pct_per_video_duration_group": interval_macro_false_pred_dur_pct_video,
            "interval_macro_model_start_latency_s_group": interval_macro_latency,
        }

    try:
        current_thr = dict(orig_thr)
        current_sig = dict(orig_sig)
        trial_count = 0

        group_order = ["DAVE_IV", "DAVE_MV", "DAVE_PROPAQ", "CHUCK_IV", "CHUCK_MV", "CHUCK_PROPAQ"]
        group_to_oids = {}
        for oid in sorted(current_thr.keys()):
            grp = OBJ_GROUP.get(int(oid))
            if grp:
                group_to_oids.setdefault(grp, []).append(int(oid))

        print("[grid] starting single-run object-group tuning")
        # Baseline (full evaluation) for reference in logs
        base_score, base_metric, base_sim_scores, base_group_avg, base_family_avg, _ = _evaluate_maps(
            current_thr, current_sig, subset_oids=None
        )
        print("\n" + "=" * 88)
        print(f"[grid] baseline full score={base_score} (hier_group_macro_all={base_metric})")
        print(f"[grid] baseline per_group_avg_across_simulations: {base_group_avg}")

        selected_by_group = {}
        selected_by_object = {}
        group_tuning_history = {}
        for grp in group_order:
            oids = group_to_oids.get(grp, [])
            if not oids:
                continue
            print("\n" + "=" * 88)
            print(f"[grid] tuning group {grp} with objects {oids}")

            if grid_mode == "absolute":
                if not thr_values or not sigma_values:
                    raise ValueError("absolute grid mode requires thr_values and sigma_values.")
                combo_iter = [("absolute", tv, sv, 0) for tv, sv in itertools.product(thr_values, sigma_values)]
            else:
                combo_iter = [("scale", ts, ss, sa) for ts, ss, sa in itertools.product(thr_scales, sigma_scales, sigma_adds)]

            group_trial_rows = []
            print(f"[grid] group-scoped metric: {grp}_group_balanced_macro_F1")

            # tune one object at a time using the group's macro F1 as objective
            for target_oid in oids:
                print(f"[grid] tuning object {target_oid} within group {grp}")
                # baseline score for this object step
                _, _, _, step_group_avg, _, _ = _evaluate_maps(current_thr, current_sig, subset_oids=oids)
                step_best_score = step_group_avg.get(grp)
                step_best_score = float(step_best_score) if step_best_score is not None else -1.0
                step_best_thr = float(current_thr[target_oid])
                step_best_sig = int(current_sig[target_oid])

                trial_specs = []
                for mode, a, b, c in combo_iter:
                    cand_thr = dict(current_thr)
                    cand_sig = dict(current_sig)

                    if mode == "absolute":
                        tv, sv = float(a), int(round(float(b)))
                        cand_thr[target_oid] = max(0.01, min(0.99, tv))
                        cand_sig[target_oid] = max(1, sv)
                    else:
                        ts, ss, sa = float(a), float(b), int(c)
                        cand_thr[target_oid] = max(0.01, min(0.99, float(current_thr[target_oid]) * ts))
                        cand_sig[target_oid] = max(1, int(round(float(current_sig[target_oid]) * ss)) + sa)
                    trial_count += 1
                    trial_spec = {
                        "mode": mode, "a": a, "b": b, "c": c,
                        "cand_thr": cand_thr, "cand_sig": cand_sig
                    }
                    trial_specs.append(trial_spec)

                eval_outputs = []
                if int(grid_workers) > 1 and len(trial_specs) > 1:
                    task_payloads = []
                    for spec in trial_specs:
                        task_payloads.append({
                            "thresh_map": spec["cand_thr"],
                            "sigma_map": spec["cand_sig"],
                            "pred_csvs": pred_csvs,
                            "video_paths": video_paths,
                            "gt_csv": gt_csv,
                            "gt_sheet_tokens": gt_sheet_tokens,
                            "subset_oids": oids,
                            "group_key": grp,
                            "obj_id": obj_id,
                            "verb_id": verb_id,
                            "merge_gap_s": merge_gap_s,
                            "min_dur_s": min_dur_s,
                            "fps_override": fps_override,
                            "gt_interaction": gt_interaction,
                            "all_objs": all_objs,
                            "debug_gt": debug_gt,
                            "metric_name": metric_name,
                        })
                    with cf.ProcessPoolExecutor(max_workers=int(grid_workers)) as ex:
                        eval_outputs = list(ex.map(_grid_eval_group_candidate_task, task_payloads))
                else:
                    for spec in trial_specs:
                        _, _, _, group_avg_only, _, _ = _evaluate_maps(spec["cand_thr"], spec["cand_sig"], subset_oids=oids)
                        grp_score = group_avg_only.get(grp)
                        grp_score = float(grp_score) if grp_score is not None else -1.0
                        eval_outputs.append({
                            "group_score": grp_score,
                            "metric_value": None,
                            "per_group_avg_across_simulations": group_avg_only,
                            "per_simulation_metric": {},
                        })

                for spec, out_eval in zip(trial_specs, eval_outputs):
                    mode, a, b, c = spec["mode"], spec["a"], spec["b"], spec["c"]
                    cand_thr, cand_sig = spec["cand_thr"], spec["cand_sig"]
                    grp_score = float(out_eval.get("group_score", -1.0))

                    trial_row = {
                        "group_tuned": grp,
                        "object_tuned": int(target_oid),
                        "object_ids_in_group": [int(x) for x in oids],
                        "group_score": grp_score,
                        "metric_name": f"{grp}_group_balanced_macro_F1",
                        "grid_mode": "absolute_object_with_group_eval" if mode == "absolute" else "scale_object_with_group_eval",
                        "thr_scale": float(a) if mode == "scale" else None,
                        "sigma_scale": float(b) if mode == "scale" else None,
                        "sigma_add": int(c) if mode == "scale" else None,
                        "thr_value": float(a) if mode == "absolute" else None,
                        "sigma_value": float(b) if mode == "absolute" else None,
                        "candidate_threshold_for_object": float(cand_thr[target_oid]),
                        "candidate_sigma_for_object": int(cand_sig[target_oid]),
                    }
                    results.append(trial_row)
                    group_trial_rows.append(trial_row)
                    print(
                        f"[grid] trial object={target_oid} "
                        f"threshold={cand_thr[target_oid]:.4f} sigma={cand_sig[target_oid]} "
                        f"group_score={grp_score}"
                    )

                    if grp_score > step_best_score:
                        step_best_score = grp_score
                        step_best_thr = float(cand_thr[target_oid])
                        step_best_sig = int(cand_sig[target_oid])

                current_thr[target_oid] = step_best_thr
                current_sig[target_oid] = step_best_sig
                selected_by_object[str(int(target_oid))] = {
                    "group": grp,
                    "threshold": float(step_best_thr),
                    "sigma": int(step_best_sig),
                    "selection_metric": f"{grp}_group_balanced_macro_F1",
                    "selection_score": float(step_best_score),
                }
                print(
                    f"[grid] selected object={target_oid} "
                    f"threshold={step_best_thr:.4f} sigma={step_best_sig} "
                    f"group_score={step_best_score}"
                )

            # end-of-group score after all objects in this group were tuned
            _, _, _, final_group_avg, _, _ = _evaluate_maps(current_thr, current_sig, subset_oids=oids)
            final_group_score = final_group_avg.get(grp)
            final_group_score = float(final_group_score) if final_group_score is not None else -1.0
            selected_by_group[grp] = {
                "group_score_after_tuning": final_group_score,
                "objects": [int(x) for x in oids],
                "selected_thresholds": {str(int(oid)): float(current_thr[oid]) for oid in oids},
                "selected_sigmas": {str(int(oid)): int(current_sig[oid]) for oid in oids},
                # Backward-compatible aliases
                "best_thresholds": {str(int(oid)): float(current_thr[oid]) for oid in oids},
                "best_sigmas": {str(int(oid)): int(current_sig[oid]) for oid in oids},
            }
            group_tuning_history[grp] = group_trial_rows
            print(f"[grid] completed group {grp} group_score_after_tuning={final_group_score}")

        # Final full evaluation across all groups using selected params
        final_score, final_metric, final_sim_scores, final_group_avg, final_family_avg, final_sim_summaries = _evaluate_maps(
            current_thr, current_sig, subset_oids=None
        )
        print("\n" + "=" * 88)
        print("[grid] running final full evaluation with selected per-object parameters")
        print(f"[grid] final full evaluation score={final_score} (hier_group_macro_all={final_metric})")
        print(f"[grid] final overall group macro F1={final_metric}")
        print(f"[grid] final full per_group_avg_across_simulations: {final_group_avg}")
        print(f"[grid] final aggregated overall group macro F1 across all provided simulations = {final_metric}")
        print(f"[grid] final aggregated per-group macro F1 across all provided simulations = {final_group_avg}")

        final_all_sim_metrics = _aggregate_all_sim_metrics(final_sim_summaries)
        print("[grid] final all-simulation aggregated metrics across all provided simulations/views")
        print("Frame-level F1 per group:")
        for gk in ["DAVE_IV", "DAVE_MV", "DAVE_PROPAQ", "CHUCK_IV", "CHUCK_MV", "CHUCK_PROPAQ"]:
            row = (final_all_sim_metrics.get("frame_level_f1_per_group", {}) or {}).get(gk, {})
            print(f"  {gk}: pos={row.get('pos')} weighted={row.get('weighted')} balanced={row.get('balanced')}")
        print(f"Frame-level macro F1 (group pos): {final_all_sim_metrics.get('frame_level_macro_f1_group_pos')}")
        print(f"Frame-level macro F1 (group weighted incl. no_interaction): {final_all_sim_metrics.get('frame_level_macro_f1_group_weighted')}")
        print(f"Frame-level macro F1 (group equal pos/neg): {final_all_sim_metrics.get('frame_level_macro_f1_group_balanced')}")
        print(f"Interval macro overlap ratio (group): {final_all_sim_metrics.get('interval_macro_overlap_ratio_group')}")
        print(f"Interval macro falsely predicted interactions count (group): {final_all_sim_metrics.get('interval_macro_false_interactions_count_group')}")
        print(f"Interval macro falsely predicted interactions count pct (group): {final_all_sim_metrics.get('interval_macro_false_interactions_count_pct_group')}")
        print(f"Interval total falsely predicted interactions count (group): {final_all_sim_metrics.get('interval_total_false_interactions_count_group')}")
        print(f"Interval macro false interaction prediction duration pct (group): {final_all_sim_metrics.get('interval_macro_false_interaction_pred_duration_pct_group')}")
        print(f"Interval macro false interaction prediction duration pct per total video duration (group): {final_all_sim_metrics.get('interval_macro_false_interaction_pred_duration_pct_per_video_duration_group')}")
        print(f"Interval macro model start latency s (group): {final_all_sim_metrics.get('interval_macro_model_start_latency_s_group')}")
        print("=" * 100)

        best = {
            "score": final_score,
            "metric_name": "hier_group_macro_all",
            "metric_value": final_metric,
            "per_simulation_metric": final_sim_scores,
            "per_group_avg_across_simulations": final_group_avg,
            "family_macro_f1_avg_across_simulations": final_family_avg,
            "grid_mode": "objectwise_with_group_eval_" + str(grid_mode),
            "selected_by_group": selected_by_group,
            "selected_by_object": selected_by_object,
            "group_tuning_history": group_tuning_history,
            "best_thresh_map": {int(k): float(v) for k, v in current_thr.items()},
            "best_sigma_map": {int(k): int(v) for k, v in current_sig.items()},
            "trial_count": int(trial_count),
            "final_full_evaluation": {
                "score": final_score,
                "metric_name": "hier_group_macro_all",
                "metric_value": final_metric,
                "overall_group_macro_F1": final_metric,
                "per_group_scores": final_group_avg,
                "per_simulation_metric": final_sim_scores,
                "per_group_avg_across_simulations": final_group_avg,
                "family_macro_f1_avg_across_simulations": final_family_avg,
            },
            "final_aggregated_overall_group_macro_F1_across_all_simulations": final_metric,
            "final_aggregated_per_group_macro_F1_across_all_simulations": final_group_avg,
            "final_all_simulation_aggregated_metrics": final_all_sim_metrics,
        }
    finally:
        _apply_default_params(orig_thr, orig_sig)

    ranked = sorted(results, key=lambda x: x.get("group_score", -1.0), reverse=True)
    return best, ranked

# ---------- CLI ----------
def _require_file(path, flag_name):
    """Fail fast with a clear message instead of a deep pandas/cv2 traceback."""
    if path and not Path(path).is_file():
        raise FileNotFoundError(f"{flag_name} points to a file that does not exist: {path}")


def _require_files(paths, flag_name):
    if not paths:
        return
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{flag_name} has {len(missing)} missing file(s):\n  " + "\n  ".join(missing)
        )


def _validate_inputs(args):
    """Check that every input path the user passed actually exists."""
    _require_file(args.pred_csv, "--pred_csv")
    _require_files(args.pred_csvs, "--pred_csvs")
    _require_file(args.video_path, "--video_path")
    _require_files(args.video_paths, "--video_paths")
    _require_file(args.gt_csv, "--gt_csv")
    _require_file(args.load_best_json, "--load_best_json")
    _require_file(args.load_params_json, "--load_params_json")
    if args.pred_csvs and args.video_paths and len(args.pred_csvs) != len(args.video_paths):
        raise ValueError(
            f"--pred_csvs has {len(args.pred_csvs)} entries but --video_paths has "
            f"{len(args.video_paths)}; pass them in the same order, one per camera view."
        )
    if args.grid_search and args.pred_csvs and args.gt_sheet_tokens and \
            len(args.gt_sheet_tokens) != len(args.pred_csvs):
        raise ValueError(
            f"--gt_sheet_tokens has {len(args.gt_sheet_tokens)} entries but --pred_csvs has "
            f"{len(args.pred_csvs)}; --grid_search requires one GT sheet token per view."
        )


def _print_dry_run_report(args):
    print("[dry_run] All referenced input files were found. Nothing was processed. Summary:")
    if args.pred_csv:
        print(f"  single-view pred_csv: {args.pred_csv}")
    if args.pred_csvs:
        print(f"  multi-view pred_csvs ({len(args.pred_csvs)}):")
        video_paths = args.video_paths or []
        for i, pc in enumerate(args.pred_csvs):
            vp = video_paths[i] if i < len(video_paths) else "(none given)"
            print(f"    {pc}\n      <-> video: {vp}")
    if args.gt_csv:
        print(f"  gt_csv: {args.gt_csv}")
    if args.demo_outdir:
        print(f"  demo_outdir (created if missing): {args.demo_outdir}")
    if args.save_segments:
        print(f"  segments CSV will be written to: {args.save_segments}")
    if args.grid_search:
        print("  mode: --grid_search (hyperparameter tuning against ground truth)")
    print("[dry_run] Re-run the same command without --dry_run to actually generate intervals.")


def main():
    p = argparse.ArgumentParser(description="Create temporal predicted HOI intervals from one or more simulation views using the same smoothing and segment-normalization logic as the visualization pipeline.")
    p.add_argument("--all_objs", action="store_true", help="Process all object_class IDs present in the prediction CSV")
    p.add_argument("--pred_csv", default=None, help="Path to *_df_preds.csv")
    p.add_argument("--pred_csvs", nargs="+", default=None, help="Multi-view mode: one or more *_df_preds.csv paths")
    p.add_argument("--gt_csv", default=None, help="Path to GT annotations CSV (optional)")
    p.add_argument("--gt_interaction", default=None, help="Optional; if omitted, inferred from --obj using built-in mapping")
    p.add_argument("--gt_sheet_token", default=None, help="Optional; override patient token used for GT filtering (e.g., '2024c_alpha2').")
    p.add_argument("--gt_sheet_tokens", nargs="+", default=None, help="Multi-view mode: optional GT sheet tokens aligned with --pred_csvs")
    p.add_argument("--obj", nargs="+", type=int, default=[101], help="Object class id(s); one or more (e.g., 85 90)")
    p.add_argument("--verb", type=int, default=117, help="Verb class id (default=117)")
    p.add_argument("--sigma", type=int, default=None, help="Gaussian sigma (frames). If omitted, use per-object defaults.")
    p.add_argument("--merge_gap_s", type=float, default=4, help="Merge gaps ≤ seconds (default: 4s)")
    p.add_argument("--min_dur_s", type=float, default=5, help="Keep segments >= seconds; < threshold goes to uncertain output (default: 5s)")
    p.add_argument("--fps", type=float, default=None, help="Override FPS (else auto-infer)")
    p.add_argument("--save_segments", default=None, help="Path to save segments CSV")
    p.add_argument("--no_show", action="store_true", help="Do not display the plot")
    p.add_argument("--video_path", default=None, help="Path to source .mp4 to render demo overlays (optional)")
    p.add_argument("--video_paths", nargs="+", default=None, help="Multi-view mode: source .mp4 paths aligned with --pred_csvs")
    p.add_argument("--demo_outdir", default=None, help="Output directory for demo videos (optional)")
    p.add_argument("--no_video_demos", action="store_true", help="Skip writing demo videos even if video_path is provided (only plots + F1).")
    p.add_argument("--demo_fps", type=float, default=4.0, help="Override FPS for demo video writing (default: 4.0)")
    p.add_argument("--box_base_w", type=float, default=None, help="If preds were generated on resized frames, give their width to rescale boxes to video size.")
    p.add_argument("--box_base_h", type=float, default=None, help="If preds were generated on resized frames, give their height to rescale boxes to video size.")
    p.add_argument("--debug_gt", action="store_true", help="Print GT parsing diagnostics (found columns, sheet names, interval counts).")
    p.add_argument("--grid_search", action="store_true", help="Run simple grid search over scaled DEFAULT_OBJ_THRESH and DEFAULT_OBJ_SIGMA (multi-view mode only).")
    p.add_argument("--grid_thr_scales", default="0.9,1.0,1.1", help="Comma-separated threshold scale factors for grid search.")
    p.add_argument("--grid_sigma_scales", default="0.8,1.0,1.2", help="Comma-separated sigma scale factors for grid search.")
    p.add_argument("--grid_sigma_adds", default="-2,0,2", help="Comma-separated additive sigma offsets (after scaling).")
    p.add_argument("--grid_mode", default="scale", choices=["scale", "absolute"], help="Grid-search mode: scale current defaults, or absolute global thr/sigma.")
    p.add_argument("--grid_thr_values", default=None, help="Absolute mode: comma-separated threshold values (e.g., '0.2,0.3,0.4').")
    p.add_argument("--grid_sigma_values", default=None, help="Absolute mode: comma-separated sigma values (e.g., '10,20,30').")
    p.add_argument("--grid_thr_min", type=float, default=None, help="Absolute mode: threshold range min.")
    p.add_argument("--grid_thr_max", type=float, default=None, help="Absolute mode: threshold range max.")
    p.add_argument("--grid_thr_step", type=float, default=None, help="Absolute mode: threshold range step.")
    p.add_argument("--grid_sigma_min", type=float, default=None, help="Absolute mode: sigma range min.")
    p.add_argument("--grid_sigma_max", type=float, default=None, help="Absolute mode: sigma range max.")
    p.add_argument("--grid_sigma_step", type=float, default=None, help="Absolute mode: sigma range step.")
    p.add_argument("--grid_out_json", default=None, help="Optional path to write full grid-search results JSON.")
    p.add_argument("--grid_apply_best", action="store_true", help="After --grid_search, apply best params and run final segment generation.")
    p.add_argument("--grid_workers", type=int, default=1, help="Number of parallel workers for candidate trials within each tuned object (default: 1).")
    p.add_argument("--load_best_json", default=None, help="Optional path to a saved gridsearch JSON; applies best_thresh_map/best_sigma_map before running.")
    p.add_argument("--load_params_json", default=None, help="Optional path to manual params JSON with keys: thresholds, sigmas.")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate that all input files/paths exist and print a summary of what "
                        "would run, without generating any intervals or writing any output. "
                        "Use this first to catch typo'd paths before a long run.")
    args = p.parse_args()

    # Expand ~ so paths work whether passed as "~/..." or "$HOME/..."
    _eu = os.path.expanduser
    if args.pred_csv:      args.pred_csv      = _eu(args.pred_csv)
    if args.pred_csvs:     args.pred_csvs     = [_eu(p) for p in args.pred_csvs]
    if args.video_path:    args.video_path    = _eu(args.video_path)
    if args.video_paths:   args.video_paths   = [_eu(p) for p in args.video_paths]
    if args.gt_csv:        args.gt_csv        = _eu(args.gt_csv)
    if args.demo_outdir:   args.demo_outdir   = _eu(args.demo_outdir)
    if args.grid_out_json: args.grid_out_json = _eu(args.grid_out_json)
    if args.load_best_json:  args.load_best_json  = _eu(args.load_best_json)
    if args.load_params_json: args.load_params_json = _eu(args.load_params_json)

    _validate_inputs(args)
    if args.dry_run:
        _print_dry_run_report(args)
        return

    if args.load_best_json and args.load_params_json:
        raise ValueError("Use only one of --load_best_json or --load_params_json.")

    if args.load_best_json:
        n_thr, n_sig = _load_best_params_from_json(args.load_best_json)
        print(f"[load_best_json] Applied tuned defaults from {args.load_best_json} (thresholds={n_thr}, sigmas={n_sig})")
    if args.load_params_json:
        n_thr, n_sig = _load_manual_params_from_json(args.load_params_json)
        print(f"[load_params_json] Applied manual defaults from {args.load_params_json} (thresholds={n_thr}, sigmas={n_sig})")

    if args.grid_search:
        if not args.pred_csvs or not args.video_paths:
            raise ValueError("--grid_search requires multi-view inputs: --pred_csvs and --video_paths.")
        if not args.gt_csv or not args.gt_sheet_tokens:
            raise ValueError("--grid_search requires GT: --gt_csv and --gt_sheet_tokens.")
        if len(args.pred_csvs) != len(args.video_paths) or len(args.pred_csvs) != len(args.gt_sheet_tokens):
            raise ValueError("For --grid_search, --pred_csvs/--video_paths/--gt_sheet_tokens must have equal lengths.")

        thr_values = None
        sigma_values = None
        if args.grid_mode == "absolute":
            if args.grid_thr_values and args.grid_sigma_values:
                thr_values = _parse_float_csv(args.grid_thr_values)
                sigma_values = _parse_float_csv(args.grid_sigma_values)
            elif None not in (args.grid_thr_min, args.grid_thr_max, args.grid_thr_step, args.grid_sigma_min, args.grid_sigma_max, args.grid_sigma_step):
                thr_values = _build_float_range(args.grid_thr_min, args.grid_thr_max, args.grid_thr_step)
                sigma_values = _build_float_range(args.grid_sigma_min, args.grid_sigma_max, args.grid_sigma_step)
            else:
                raise ValueError(
                    "absolute mode requires either --grid_thr_values/--grid_sigma_values or full min/max/step ranges."
                )

        best, ranked = run_simple_grid_search(
            args.pred_csvs,
            args.video_paths,
            args.gt_csv,
            args.gt_sheet_tokens,
            obj_id=args.obj,
            verb_id=args.verb,
            merge_gap_s=args.merge_gap_s,
            min_dur_s=args.min_dur_s,
            fps_override=args.fps,
            gt_interaction=args.gt_interaction,
            all_objs=args.all_objs,
            debug_gt=args.debug_gt,
            thr_scales=_parse_float_csv(args.grid_thr_scales),
            sigma_scales=_parse_float_csv(args.grid_sigma_scales),
            sigma_adds=_parse_int_csv(args.grid_sigma_adds),
            grid_mode=args.grid_mode,
            thr_values=thr_values,
            sigma_values=sigma_values,
            metric_name="f1_group_macro_all",
            grid_workers=max(1, int(args.grid_workers)),
        )

        print("\n[grid] Best combination:")
        print(json.dumps(best, indent=2))
        print("\n[grid] Top 5 combinations:")
        for row in ranked[:5]:
            print(json.dumps(row, indent=2))

        if args.grid_out_json:
            out_obj = {"best": best, "top5": ranked[:5], "all_results": ranked}
            Path(args.grid_out_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.grid_out_json, "w") as f:
                json.dump(out_obj, f, indent=2)
            print(f"[grid] Saved results JSON -> {args.grid_out_json}")

        if args.grid_apply_best and best is not None:
            if best.get("best_thresh_map") is not None and best.get("best_sigma_map") is not None:
                tuned_thr = best.get("best_thresh_map", {})
                tuned_sig = best.get("best_sigma_map", {})
                _apply_default_params(tuned_thr, tuned_sig)
            elif best.get("grid_mode") == "absolute":
                tuned_thr, tuned_sig = _constant_default_params(
                    thr_value=best["thr_value"],
                    sigma_value=best["sigma_value"],
                )
                _apply_default_params(tuned_thr, tuned_sig)
            else:
                tuned_thr, tuned_sig = _scale_default_params(
                    thr_scale=best["thr_scale"],
                    sigma_scale=best["sigma_scale"],
                    sigma_add=best["sigma_add"],
                )
                _apply_default_params(tuned_thr, tuned_sig)
            print("[grid] Applied best parameters to defaults for this run.")
        elif not args.grid_apply_best:
            print("[grid] Grid search completed. Re-run with --grid_apply_best to execute final run with tuned params.")
            return

    if args.pred_csvs:
        if not args.video_paths:
            raise ValueError("Multi-view mode requires --video_paths with --pred_csvs.")
        if args.gt_sheet_tokens and len(args.gt_sheet_tokens) == len(args.pred_csvs):
            sim_groups = _build_simulation_groups(args.pred_csvs, args.video_paths, args.gt_sheet_tokens)
        else:
            sim_groups = {"combined": {"pred_csvs": args.pred_csvs, "video_paths": args.video_paths, "gt_sheet_tokens": args.gt_sheet_tokens or [None] * len(args.pred_csvs)}}

        segs_all = []
        fps = None
        sim_summaries = {}
        for sim_key, gd in sim_groups.items():
            sim_outdir = args.demo_outdir
            if args.demo_outdir:
                sim_outdir = str(Path(args.demo_outdir) / sim_key)
            print(f"\n=== Running simulation group: {sim_key} (views={len(gd['pred_csvs'])}) ===")
            try:
                segs, fps_sim, sim_summary = predicted_intervals_multi(
                    gd["pred_csvs"],
                    gd["video_paths"],
                    args.gt_csv,
                    gd["gt_sheet_tokens"],
                    obj_id=args.obj,
                    verb_id=args.verb,
                    sigma=args.sigma,
                    merge_gap_s=args.merge_gap_s,
                    min_dur_s=args.min_dur_s,
                    fps_override=args.fps,
                    gt_interaction=args.gt_interaction,
                    demo_outdir=sim_outdir,
                    all_objs=args.all_objs,
                    debug_gt=args.debug_gt,
                    return_summary=True,
                )
            except Exception as e:
                import traceback
                print(f"\n[ERROR] Simulation group failed; continuing to next group: {sim_key}")
                print(f"  error: {e}")
                print(traceback.format_exc())
                continue
            sim_summaries[sim_key] = sim_summary
            for s in segs:
                s["simulation_key"] = sim_key
            segs_all.extend(segs)
            if fps is None:
                fps = fps_sim
        if not segs_all:
            raise RuntimeError("No simulation groups completed successfully.")
        _print_cross_sim_family_averages(sim_summaries)
        per_group_avg_across_sims, overall_hier = _cross_sim_group_hier_macro(sim_summaries)
        print("Cross-simulation per-group macro F1 averages:")
        for gk, gv in sorted(per_group_avg_across_sims.items()):
            print(f"  {gk}: {gv}")
        print(f"Cross-simulation overall average (avg over groups): {overall_hier}")
        segs = segs_all
    else:
        if not args.pred_csv:
            raise ValueError("Single-view mode requires --pred_csv.")
        segs, fps = predicted_intervals_only(
            args.pred_csv, obj_id=args.obj, verb_id=args.verb,
            sigma=args.sigma,
            merge_gap_s=args.merge_gap_s, min_dur_s=args.min_dur_s,
            fps_override=args.fps,
            show=not args.no_show,
            title=f"[{args.pred_csv}]",
            gt_csv_path=args.gt_csv,
            sheet_token_override=args.gt_sheet_token,
            gt_interaction=args.gt_interaction,
            video_path=args.video_path, demo_outdir=args.demo_outdir, demo_fps=args.demo_fps,
            enable_video_demos=not args.no_video_demos,
            all_objs=args.all_objs,
            box_base_w=args.box_base_w, box_base_h=args.box_base_h,
            debug_gt=args.debug_gt
        )
    print(f"Segments (FPS={fps:.3f}): {len(segs)}")
    for s in segs: print(s)
    if args.save_segments:
        pd.DataFrame(segs).to_csv(args.save_segments, index=False)
        print(f"Saved segments → {args.save_segments}")

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)
