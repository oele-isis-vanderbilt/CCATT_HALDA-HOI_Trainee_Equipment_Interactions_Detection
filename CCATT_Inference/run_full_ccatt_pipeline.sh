#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-command entry point for the full CCATT inference pipeline: give it the
# synchronized CAM + PAN videos of one simulation, get back a CSV of who used
# what equipment and when. It chains all 4 steps (Advanced_FineTuned_Model_..
# .sh, V3_Create_Temporal_..., generate_role_assignment_csvs.py,
# person_identification_v4.py) so you don't have to run them by hand.
#
# Required environment variables:
#   PRETRAINED_MODEL_PATH  path to the fine-tuned CDN checkpoint (.pth) --
#                          this is the equipment/action-detection model
#   PERSON_ID_WEIGHTS      path to the trained person/role identification
#                          model's weights (.pt) -- this is a SEPARATE model
#                          from PRETRAINED_MODEL_PATH above; ask the ML team
#                          for it
#   DEMO_ROLE_SCRIPT       path to role_contrastive/demo_role.py (part of the
#                          person/role identification model) -- ask the ML
#                          team for this
#   PYTHON_BIN             python3 interpreter that has the person/role
#                          identification model's environment's dependencies
#                          installed -- ask the ML team for this
#
# Optional:
#   INPUT_DIR       folder holding exactly the 2 videos (CAM + PAN) for one
#                   simulation (default: ./Video_Input next to this script)
#   OUTPUT_ROOT     where all intermediate + final results are written
#                   (default: ./ccatt_pipeline_output next to this script)
#   CDN_REPO_ROOT   path to a CDN_Pretrained repo, if you want to use a copy
#                   other than the one bundled next to this script
#                   (default: ./CDN_Pretrained next to this script)
#   JOB_NUMBER      label used to name the Step 0 results subfolder
#                   (default: CCATT_Run)
#   CONF            role-model detection confidence threshold (default: 0.10)
#   DEVICE          cuda / cpu / mps for CDN + role-model inference
#                   (default: cuda; use cpu if no GPU available)
#
# Usage:
#   PRETRAINED_MODEL_PATH=/path/to/checkpoint_best.pth \
#   PERSON_ID_WEIGHTS=/path/to/role_contrastive/weights/best.pt \
#   DEMO_ROLE_SCRIPT=/path/to/role_contrastive/demo_role.py \
#   PYTHON_BIN=/path/to/role_yolo_env/bin/python3 \
#   bash run_full_ccatt_pipeline.sh [--dry_run]
#
# --dry_run checks that all input paths exist (including that exactly 2
# videos are present in INPUT_DIR) and prints what would run, without
# actually processing anything.
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
if [ "${1:-}" = "--dry_run" ]; then
  DRY_RUN=1
fi

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
PERSON_ID_WEIGHTS="${PERSON_ID_WEIGHTS:-}"
DEMO_ROLE_SCRIPT="${DEMO_ROLE_SCRIPT:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/Video_Input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/ccatt_pipeline_output}"
CDN_REPO_ROOT="${CDN_REPO_ROOT:-$ROOT_DIR/CDN_Pretrained}"
JOB_NUMBER="${JOB_NUMBER:-CCATT_Run}"
CONF="${CONF:-0.10}"
DEVICE="${DEVICE:-cuda}"

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
if [ -z "$PERSON_ID_WEIGHTS" ] || [ -z "$DEMO_ROLE_SCRIPT" ] || [ -z "$PYTHON_BIN" ]; then
  echo "[ERROR] Set PERSON_ID_WEIGHTS, DEMO_ROLE_SCRIPT, and PYTHON_BIN to your copy of the" >&2
  echo "  role identification model (ask the ML team for it -- see README.md)." >&2
  exit 1
fi
if [ ! -f "$PERSON_ID_WEIGHTS" ]; then
  echo "[ERROR] PERSON_ID_WEIGHTS not found: $PERSON_ID_WEIGHTS" >&2
  exit 1
fi
if [ ! -f "$DEMO_ROLE_SCRIPT" ]; then
  echo "[ERROR] DEMO_ROLE_SCRIPT not found: $DEMO_ROLE_SCRIPT" >&2
  exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
  echo "[ERROR] INPUT_DIR not found: $INPUT_DIR" >&2
  exit 1
fi

# Only the extensions Advanced_FineTuned_Model_..._Actv_L.sh (Step 0) itself
# looks for -- keep this in sync with that script so what we count here is
# exactly what Step 0 will process.
shopt -s nullglob nocaseglob
VIDEOS=("$INPUT_DIR"/*.mp4 "$INPUT_DIR"/*.3gp)
shopt -u nullglob nocaseglob

if [ "${#VIDEOS[@]}" -ne 2 ]; then
  echo "[ERROR] Expected exactly 2 videos (synchronized CAM + PAN views of one" >&2
  echo "  simulation) in INPUT_DIR, found ${#VIDEOS[@]}: $INPUT_DIR" >&2
  echo "  See the 'What to provide' section in README.md." >&2
  exit 1
fi

STEP1_OUTDIR="$OUTPUT_ROOT/step1_segments"
ROLE_CSV_ROOT="$OUTPUT_ROOT/role_assignment_csvs"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry_run] All required inputs found:"
  echo "  videos (CAM + PAN):    ${VIDEOS[0]}"
  echo "                         ${VIDEOS[1]}"
  echo "  CDN checkpoint:        $PRETRAINED_MODEL_PATH"
  echo "  CDN repo:              $CDN_REPO_ROOT"
  echo "  person ID weights:     $PERSON_ID_WEIGHTS"
  echo "  role demo script:      $DEMO_ROLE_SCRIPT"
  echo "  role python bin:       $PYTHON_BIN"
  echo "  Step 0 results go to:  $OUTPUT_ROOT/hoi_results_${JOB_NUMBER}/"
  echo "  Step 1 results go to:  $STEP1_OUTDIR/combined/combined_segments.csv"
  echo "  Step 2 results go to:  $ROLE_CSV_ROOT/"
  echo "  final result goes to:  $STEP1_OUTDIR/combined/CCATT_Trainee_Equipment_Interactions.csv"
  echo "[dry_run] Re-run without --dry_run to actually process this simulation."
  exit 0
fi

echo "=== Step 0: extracting frames + running CDN HOI inference on both videos (GPU, slow) ==="
PRETRAINED_MODEL_PATH="$PRETRAINED_MODEL_PATH" \
CDN_REPO_ROOT="$CDN_REPO_ROOT" \
INPUT_DIR="$INPUT_DIR" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
JOB_NUMBER="$JOB_NUMBER" \
  bash "$ROOT_DIR/Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh"

PRED_CSVS=()
VIDEO_PATHS=()
for VIDEO_PATH in "${VIDEOS[@]}"; do
  VIDEO_NAME="$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')"
  PRED_CSV="$(find "$OUTPUT_ROOT/hoi_results_${JOB_NUMBER}" -type f -name "${VIDEO_NAME}_df_preds.csv" -print -quit)"
  if [ -z "$PRED_CSV" ]; then
    echo "[ERROR] Step 0 did not produce a df_preds.csv for $VIDEO_NAME" >&2
    exit 1
  fi
  PRED_CSVS+=("$PRED_CSV")
  VIDEO_PATHS+=("$VIDEO_PATH")
done

echo "=== Step 1: converting frame-level predictions to temporal intervals (both views combined) ==="
python3 "$ROOT_DIR/V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py" \
  --pred_csvs "${PRED_CSVS[@]}" \
  --video_paths "${VIDEO_PATHS[@]}" \
  --all_objs \
  --demo_outdir "$STEP1_OUTDIR" \
  --no_video_demos

if [ ! -f "$STEP1_OUTDIR/combined/combined_segments.csv" ]; then
  echo "[ERROR] Step 1 did not produce $STEP1_OUTDIR/combined/combined_segments.csv" >&2
  exit 1
fi

echo "=== Step 2: figuring out who (Nurse/Doctor/RT) is where (GPU, slow) ==="
python3 "$ROOT_DIR/generate_role_assignment_csvs.py" \
  --input_root "$STEP1_OUTDIR" \
  --video_roots "$INPUT_DIR" \
  --role_csv_root "$ROLE_CSV_ROOT" \
  --weights "$PERSON_ID_WEIGHTS" \
  --demo_role_script "$DEMO_ROLE_SCRIPT" \
  --python_bin "$PYTHON_BIN" \
  --conf "$CONF" \
  --device "$DEVICE"

echo "=== Step 3: combining equipment interactions + role info into the final table ==="
python3 "$ROOT_DIR/person_identification_v4.py" \
  --input_root "$STEP1_OUTDIR" \
  --role_csv_root "$ROLE_CSV_ROOT" \
  --hoi_sampled_fps 4.0

FINAL_CSV="$STEP1_OUTDIR/combined/CCATT_Trainee_Equipment_Interactions.csv"
echo "=== Done ==="
if [ -f "$FINAL_CSV" ]; then
  echo "Result saved to: $FINAL_CSV"
else
  echo "[WARNING] Expected final result not found at: $FINAL_CSV" >&2
  echo "  Check the Step 3 output above for details." >&2
fi
