#nohup bash advPhase2_SemiAutomtic_Data_PostCleaning.sh > log_advphase5_postcleaning.out 2>&1 &

# next local runs
#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3.12}

CLEAR_ROOT="${CLEAR_ROOT:-$HOME/CCAT_Opensource_work/Advanced/data_full_videos_frames/clear_frames_phase5_sample}"
STATE_DIR="${STATE_DIR:-$HOME/CCAT_Opensource_work/Advanced/phase2_cdn_autolabel_v2_phase5}"

OUT_LABELS="${OUT_LABELS:-$STATE_DIR/cdn_labels.csv}"
OUT_HIGH_FRAMES="${OUT_HIGH_FRAMES:-$STATE_DIR/high_conf_frames.txt}"
OUT_LOW_FRAMES="${OUT_LOW_FRAMES:-$STATE_DIR/low_conf_frames.txt}"
OUT_HIGH_LABELS="${OUT_HIGH_LABELS:-$STATE_DIR/high_conf_labels.csv}"
OUT_HIGH_LABELS_NMS_INPUT="${OUT_HIGH_LABELS_NMS_INPUT:-$STATE_DIR/high_conf_labels_for_nms.csv}"
OUT_HIGH_LABELS_NMS="${OUT_HIGH_LABELS_NMS:-$STATE_DIR/high_conf_labels_nms.csv}"
ADVANCED_PSEUDO_LABELS="${ADVANCED_PSEUDO_LABELS:-$STATE_DIR/advanced_psuedo_labels.csv}"

VIZ_DIR="${VIZ_DIR:-$STATE_DIR/postcleaning_viz}"
VIZ_BEFORE_DIR="${VIZ_BEFORE_DIR:-$VIZ_DIR/pre_nms}"
VIZ_AFTER_DIR="${VIZ_AFTER_DIR:-$VIZ_DIR/post_nms}"
VIZ_SAMPLE_CSV_BEFORE="${VIZ_SAMPLE_CSV_BEFORE:-$VIZ_DIR/sample_pre_nms.csv}"
VIZ_SAMPLE_CSV_AFTER="${VIZ_SAMPLE_CSV_AFTER:-$VIZ_DIR/sample_post_nms.csv}"
VIZ_SAMPLE_FRAMES="${VIZ_SAMPLE_FRAMES:-$VIZ_DIR/sample_frames.txt}"
VIZ_SAMPLES="${VIZ_SAMPLES:-50}"
VIZ_SEED="${VIZ_SEED:-123}"
VIZ_MIN_POSITIVE_FRAMES="${VIZ_MIN_POSITIVE_FRAMES:-20}"
VIZ_POSITIVE_VERBS="${VIZ_POSITIVE_VERBS:-117 118}"

REVIEW_DIR="${REVIEW_DIR:-$STATE_DIR/low_conf_review_bundle}"
REVIEW_SAMPLES="${REVIEW_SAMPLES:-320}"
REVIEW_SEED="${REVIEW_SEED:-42}"

mkdir -p "$STATE_DIR" "$VIZ_DIR"

log() { echo "[$(date +'%F %T')] $*"; }

if [ ! -f "$OUT_HIGH_LABELS" ]; then
  log "ERROR: High-confidence labels not found: $OUT_HIGH_LABELS"
  exit 1
fi

if [ ! -f "$OUT_HIGH_FRAMES" ]; then
  log "ERROR: High-confidence frames not found: $OUT_HIGH_FRAMES"
  exit 1
fi

# Safety: prevent truncation from accidental in-place overwrite.
NMS_IN_RESOLVED="$(python3 -c 'import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "$OUT_HIGH_LABELS_NMS_INPUT")"
NMS_OUT_RESOLVED="$(python3 -c 'import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "$OUT_HIGH_LABELS_NMS")"
if [ "$NMS_IN_RESOLVED" = "$NMS_OUT_RESOLVED" ]; then
  log "ERROR: OUT_HIGH_LABELS_NMS_INPUT and OUT_HIGH_LABELS_NMS resolve to the same path."
  exit 1
fi

log "Sampling ${VIZ_SAMPLES} high-confidence frames before NMS"
"$PYTHON" "$ROOT_DIR/advPhase2_sample_labels_for_viz.py" \
  --labels_csv "$OUT_HIGH_LABELS" \
  --frames_txt "$OUT_HIGH_FRAMES" \
  --output_csv "$VIZ_SAMPLE_CSV_BEFORE" \
  --output_frames_txt "$VIZ_SAMPLE_FRAMES" \
  --samples "$VIZ_SAMPLES" \
  --min-positive-frames "$VIZ_MIN_POSITIVE_FRAMES" \
  --positive-verbs $VIZ_POSITIVE_VERBS \
  --seed "$VIZ_SEED"

log "Visualizing sampled labels before NMS"
"$PYTHON" "$ROOT_DIR/5.3_visualize_phase2_annotations.py" \
  --state_dir "$STATE_DIR" \
  --frames_dir "$CLEAR_ROOT" \
  --labels_csv "$VIZ_SAMPLE_CSV_BEFORE" \
  --output_dir "$VIZ_BEFORE_DIR"

log "Preparing NMS input from OUT_HIGH_LABELS"
"$PYTHON" "$ROOT_DIR/advPhase2_prepare_nms_input.py" \
  --input_csv "$OUT_HIGH_LABELS" \
  --frames_txt "$OUT_HIGH_FRAMES" \
  --output_csv "$OUT_HIGH_LABELS_NMS_INPUT"

log "Removing true duplicate HOIs (same interaction + overlapping subject/object) with union-box representative"
"$PYTHON" "$ROOT_DIR/5.5_remove_hoi_duplicates.py" \
  -i "$OUT_HIGH_LABELS_NMS_INPUT" \
  -o "$OUT_HIGH_LABELS_NMS" \
  --person-iou 0.5 \
  --roi-iou 0.5 \
  --score-field verb_score \
  --drop-duplicates

log "Appending high-confidence pseudo labels after NMS"
"$PYTHON" "$ROOT_DIR/advPhase2_append_pseudo_labels.py" \
  --source "$OUT_HIGH_LABELS_NMS" \
  --high_conf_frames "$OUT_HIGH_FRAMES" \
  --dest "$ADVANCED_PSEUDO_LABELS"

log "Sampling ${VIZ_SAMPLES} high-confidence frames after NMS"
"$PYTHON" "$ROOT_DIR/advPhase2_sample_labels_for_viz.py" \
  --labels_csv "$OUT_HIGH_LABELS_NMS" \
  --frames_txt "$VIZ_SAMPLE_FRAMES" \
  --output_csv "$VIZ_SAMPLE_CSV_AFTER" \
  --samples "$VIZ_SAMPLES" \
  --min-positive-frames "$VIZ_MIN_POSITIVE_FRAMES" \
  --positive-verbs $VIZ_POSITIVE_VERBS \
  --seed "$VIZ_SEED"

log "Visualizing sampled labels after NMS"
"$PYTHON" "$ROOT_DIR/5.3_visualize_phase2_annotations.py" \
  --state_dir "$STATE_DIR" \
  --frames_dir "$CLEAR_ROOT" \
  --labels_csv "$VIZ_SAMPLE_CSV_AFTER" \
  --output_dir "$VIZ_AFTER_DIR"

log "Exporting low-confidence review bundle"
"$PYTHON" "$ROOT_DIR/advPhase2_export_low_conf_review_bundle.py" \
  --clear_root "$CLEAR_ROOT" \
  --low_conf_frames "$OUT_LOW_FRAMES" \
  --labels_csv "$OUT_LABELS" \
  --output_dir "$REVIEW_DIR" \
  --sample_size "$REVIEW_SAMPLES" \
  --seed "$REVIEW_SEED"

log "Post-cleaning done."
