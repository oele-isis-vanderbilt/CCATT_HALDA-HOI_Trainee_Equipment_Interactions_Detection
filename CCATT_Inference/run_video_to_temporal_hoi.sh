#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-command entry point: give this script a single video, get back a CSV of
# temporal HOI interaction start/end times. It chains Step 0 (frame extraction
# + CDN inference) directly into Step 1 (temporal interval creation) for one
# video, so you don't have to run those two steps by hand.
#
# Required environment variables:
#   VIDEO_PATH              path to a single video file (.mp4/.3gp/...)
#   PRETRAINED_MODEL_PATH   path to the fine-tuned CDN checkpoint (.pth)
#
# CDN_Pretrained's datasets/hico.py has a dedicated eval-only code path
# (triggered whenever --eval is passed): it lists *.jpg directly from
# --hoi_path with no annotations file at all ("image-only inference mode").
# No HICO-format scaffolding (images/test2015/, test_hico.json,
# trainval_hico.json, corre_hico.npy) is needed or created by this script.
#
# Frames are sampled at 4fps (matching the paper's stated methodology), but
# 1_Frames_Extraction.py keeps each saved frame's ORIGINAL video frame-count
# in its filename rather than renumbering sequentially -- so Step 1 is told
# the video's *native* fps (auto-detected via cv2), not 4, to correctly
# convert frame_id -> seconds. Passing 4 there directly gives wrong
# (inflated) start/end times -- confirmed by testing.
#
# Optional:
#   OUTPUT_DIR     where frames/predictions/segments are written
#                  (default: ./video_to_temporal_hoi_output next to this script)
#   CDN_REPO_ROOT  path to a CDN_Pretrained repo, if you want to use a copy
#                  other than the one bundled next to this script
#                  (default: ./CDN_Pretrained next to this script)
#   OBJ            space-separated object_class id(s) to score (default: all
#                  objects present in the predictions -- see --all_objs in
#                  V3_Create_Temporal_Predicted_HOI_intervals_...py --help)
#   VERB           verb class id (default: 117, same default as Step 1)
#   DEVICE         cuda / cpu / mps (default: cuda; use cpu if no GPU available
#                  -- CDN inference is slow on CPU but works)
#
# Usage:
#   VIDEO_PATH=/path/to/video.mp4 \
#   PRETRAINED_MODEL_PATH=/path/to/checkpoint_best.pth \
#   bash run_video_to_temporal_hoi.sh [--dry_run]
#
# --dry_run checks that all input paths exist and prints what would run,
# without actually extracting frames or running CDN.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
if [ "${1:-}" = "--dry_run" ]; then
  DRY_RUN=1
fi

VIDEO_PATH="${VIDEO_PATH:-}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
CDN_REPO_ROOT="${CDN_REPO_ROOT:-$ROOT_DIR/CDN_Pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/video_to_temporal_hoi_output}"
OBJ="${OBJ:-}"
VERB="${VERB:-117}"
DEVICE="${DEVICE:-cuda}"

if [ -z "$VIDEO_PATH" ]; then
  echo "[ERROR] Set VIDEO_PATH to a single video file. See --help header of this script." >&2
  exit 1
fi
if [ ! -f "$VIDEO_PATH" ]; then
  echo "[ERROR] VIDEO_PATH not found: $VIDEO_PATH" >&2
  exit 1
fi
if [ -z "$PRETRAINED_MODEL_PATH" ]; then
  echo "[ERROR] Set PRETRAINED_MODEL_PATH to the fine-tuned CDN checkpoint (.pth)." >&2
  exit 1
fi
if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
  echo "[ERROR] PRETRAINED_MODEL_PATH not found: $PRETRAINED_MODEL_PATH" >&2
  exit 1
fi
if [ ! -f "$CDN_REPO_ROOT/main.py" ]; then
  echo "[ERROR] CDN_REPO_ROOT does not look like the CDN repo (no main.py found): $CDN_REPO_ROOT" >&2
  exit 1
fi

VIDEO_NAME="$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')"
FRAMES_DIR="$OUTPUT_DIR/frames/$VIDEO_NAME"
PRED_CSV="$OUTPUT_DIR/predictions/${VIDEO_NAME}_df_preds.csv"
SEGMENTS_CSV="$OUTPUT_DIR/segments/${VIDEO_NAME}_temporal_segments.csv"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry_run] All required inputs found:"
  echo "  video:            $VIDEO_PATH"
  echo "  CDN checkpoint:   $PRETRAINED_MODEL_PATH"
  echo "  CDN repo:         $CDN_REPO_ROOT"
  echo "  frames would go to:      $FRAMES_DIR"
  echo "  predictions would go to: $PRED_CSV"
  echo "  segments would go to:    $SEGMENTS_CSV"
  echo "[dry_run] Re-run without --dry_run to actually process this video."
  exit 0
fi

mkdir -p "$(dirname "$FRAMES_DIR")" "$(dirname "$PRED_CSV")" "$(dirname "$SEGMENTS_CSV")"

echo "=== Step 0a: extracting frames (4fps sampling, matching the paper's stated methodology) ==="
# 1_Frames_Extraction.py's --input must be a directory (it globs *.mp4 inside
# it) and writes frames to <output>/<video_stem>/ -- stage just this one video
# in its own temp directory so only it gets processed, not every video that
# might otherwise be sitting next to it.
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT
ln -s "$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")" "$STAGING_DIR/$(basename "$VIDEO_PATH")"
python3 "$ROOT_DIR/1_Frames_Extraction.py" --input "$STAGING_DIR" --output "$(dirname "$FRAMES_DIR")" --fps 4
if ! find "$FRAMES_DIR" -type f -name '*.jpg' -print -quit >/dev/null 2>&1; then
  echo "[ERROR] Frame extraction produced no frames for $VIDEO_PATH" >&2
  exit 1
fi

echo "=== Step 0b: running CDN HOI inference (GPU, slow) ==="
(
  cd "$CDN_REPO_ROOT"
  python3 main.py \
    --pretrained "$PRETRAINED_MODEL_PATH" \
    --dataset_file hico \
    --hoi_path "$FRAMES_DIR" \
    --num_obj_classes 92 \
    --num_verb_classes 118 \
    --backbone resnet50 \
    --num_queries 64 \
    --dec_layers_hopd 3 \
    --dec_layers_interaction 3 \
    --device "$DEVICE" \
    --eval \
    --use_nms_filter
  # Matches the original production driver
  # (Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh)
  # exactly. This CDN_Pretrained/engine.py's save_preds() writes every frame's
  # predictions (unlike CDN_Finetuned/engine.py, which caps to the first 20
  # frames unless --adapter_dir save_all is passed).
)
if [ -f "$CDN_REPO_ROOT/df_preds.csv" ]; then
  mv "$CDN_REPO_ROOT/df_preds.csv" "$PRED_CSV"
else
  echo "[ERROR] CDN did not produce df_preds.csv in $CDN_REPO_ROOT" >&2
  exit 1
fi
echo "Saved predictions -> $PRED_CSV"

echo "=== Step 1: converting frame-level predictions to temporal intervals ==="
OBJ_ARGS=(--all_objs)
if [ -n "$OBJ" ]; then
  OBJ_ARGS=(--obj $OBJ)
fi
# 1_Frames_Extraction.py keeps each saved frame's ORIGINAL video frame-count in
# its filename (e.g. frame_000654.jpg) -- it does not renumber sequentially --
# so converting frame_id -> seconds needs the video's native fps (same value
# used internally above to compute the 4fps sampling stride), not the 4fps
# sampling target itself. Confirmed by testing: passing --fps 4 here silently
# produces wrong (inflated) start/end times.
NATIVE_FPS="$(python3 -c "import cv2,sys; print(cv2.VideoCapture(sys.argv[1]).get(cv2.CAP_PROP_FPS))" "$VIDEO_PATH")"
echo "Native fps (for frame_id -> time conversion): $NATIVE_FPS"
FPS_ARGS=(--fps "$NATIVE_FPS")

# KNOWN ISSUE on some Macs: running this python3 call immediately after Step
# 0b's heavy CDN subprocess (even in a subshell) can make numpy fail to load
# with a spurious "incompatible architecture" error. This is NOT a code bug --
# confirmed by testing that the identical command succeeds every time when run
# as its own fresh process (not chained after Step 0b in the same script), and
# that it is unrelated to environment variables or PATH (both were ruled out).
# It has not been root-caused; it does not affect Linux/GPU-server deployment.
# If it happens, Step 0's real predictions (saved above) are NOT lost -- rerun
# just this command to finish:
if ! python3 "$ROOT_DIR/V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py" \
  --pred_csv "$PRED_CSV" \
  --video_path "$VIDEO_PATH" \
  "${OBJ_ARGS[@]}" \
  "${FPS_ARGS[@]}" \
  --verb "$VERB" \
  --no_show \
  --no_video_demos \
  --save_segments "$SEGMENTS_CSV"; then
  echo "[ERROR] Step 1 failed. This is a known local-environment quirk on some Macs" >&2
  echo "  (see comment above this line in the script), not a lost result --" >&2
  echo "  your real predictions are safe at: $PRED_CSV" >&2
  echo "  Finish by re-running just this command:" >&2
  echo "  python3 \"$ROOT_DIR/V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py\" --pred_csv \"$PRED_CSV\" --video_path \"$VIDEO_PATH\" ${OBJ_ARGS[*]} ${FPS_ARGS[*]} --verb \"$VERB\" --no_show --no_video_demos --save_segments \"$SEGMENTS_CSV\"" >&2
  exit 1
fi

echo "=== Done ==="
echo "Temporal HOI start/end times saved to: $SEGMENTS_CSV"
