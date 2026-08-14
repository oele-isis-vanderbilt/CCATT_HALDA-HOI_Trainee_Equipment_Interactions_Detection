#nohup env CHUNK_SIZE=5000 bash advPhase2_SemiAutomtic_Data_Labeling.sh > log_advPhase2_SemiAutomatic_Data_Labeling_phase5.out 2>&1 &
# if raw CDN outputs availab;e dont want to run CDN again
#nohup env REUSE_CHUNK_OUTPUTS=1 CHUNK_SIZE=5000 bash advPhase2_SemiAutomtic_Data_Labeling.sh > log_advPhase2_SemiAutomatic_Data_Labeling_phase5_1.out 2>&1 &

#!/bin/bash
# Phase 2 CDN auto-labeling:
# 1) Run CDN inference on clear frames
# 2) Apply HOI rules for verb 117 vs no_interaction (57)
# 3) Save labeled annotations + low/high confidence frame lists

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
PYTHON=${PYTHON:-python3.12}

# Inputs
CLEAR_ROOT_ORIGINAL="${CLEAR_ROOT_ORIGINAL:-$HOME/CCAT_Opensource_work/Advanced/data_full_videos_frames/clear_frames}"
# sample 800*3 images from CLEAR_ROOT_original and copy them to CLEAR_ROOT
CLEAR_ROOT="${CLEAR_ROOT:-$HOME/CCAT_Opensource_work/Advanced/data_full_videos_frames/clear_frames_phase5_sample}"
STATE_DIR="${STATE_DIR:-$HOME/CCAT_Opensource_work/Advanced/phase2_cdn_autolabel_v2_phase5}"
EXCLUDE_HICO_JSON="${EXCLUDE_HICO_JSON:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/Annotations/annotations/trainval_hico.json}"

# CDN settings (point to your repo and weights)
CDN_REPO="${CDN_REPO:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN}"
#Phase1 weights
#CDN_WEIGHTS="${CDN_WEIGHTS:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase1/logs_stage2_v2/checkpoint_best.pth}"
# #Phase2 weights
# CDN_WEIGHTS="${CDN_WEIGHTS:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase2/logs_stage2_v2/checkpoint_best.pth}"
#Phase3 weights
#CDN_WEIGHTS="${CDN_WEIGHTS:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase3/logs_stage2_v2/checkpoint_best.pth}"
#Phase4 weights
CDN_WEIGHTS="${CDN_WEIGHTS:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/logs_stage2_v2/checkpoint_best.pth}"

DEVICE="${DEVICE:-cuda}"
CDN_EXTRA_ARGS="${CDN_EXTRA_ARGS:---device ${DEVICE} --batch_size 32}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
REUSE_CHUNK_OUTPUTS="${REUSE_CHUNK_OUTPUTS:-0}"

# Labeling thresholds
# #Phase 2
# SCORE_THRESH="${SCORE_THRESH:-0.5}"      # min object confidence score
# SUBJECT_SCORE_THRESH="${SUBJECT_SCORE_THRESH:-0.01}" 
# #Phase 3 & phase 4 &
SCORE_THRESH="${SCORE_THRESH:-0.00001}"      # min object confidence score
SUBJECT_SCORE_THRESH="${SUBJECT_SCORE_THRESH:-0.001}"  # keep only rows with subject/person score > this
POS_VERB_THRESH="${POS_VERB_THRESH:-0.80}"  # verb 117 => positive
NEG_VERB_THRESH="${NEG_VERB_THRESH:-0.10}"  # verb 117 => negative
DIST_THRESH="${DIST_THRESH:-150}"        # pixels; if no overlap and distance>thr => negative
NUM_OBJ_CLASSES=92
NUM_VERB_CLASSES="${NUM_VERB_CLASSES:-118}"

OUT_PRED="${OUT_PRED:-$STATE_DIR/cdn_preds_raw.csv}"
OUT_STD="${OUT_STD:-$STATE_DIR/cdn_preds_std.csv}"
OUT_LABELS="${OUT_LABELS:-$STATE_DIR/cdn_labels.csv}"
OUT_LOW_FRAMES="${OUT_LOW_FRAMES:-$STATE_DIR/low_conf_frames.txt}"
OUT_HIGH_FRAMES="${OUT_HIGH_FRAMES:-$STATE_DIR/high_conf_frames.txt}"
OUT_HIGH_LABELS="${OUT_HIGH_LABELS:-$STATE_DIR/high_conf_labels.csv}"
# ADVANCED_PSEUDO_LABELS="${ADVANCED_PSEUDO_LABELS:-$HOME/CCAT_Opensource_work/Advanced/advanced_psuedo_labels.csv}"
OUT_SAMPLE_DIR="${OUT_SAMPLE_DIR:-$STATE_DIR/sampled_low_conf}"

mkdir -p "$STATE_DIR"

log() { echo "[$(date +'%F %T')] $*"; }

log "Frames   : $CLEAR_ROOT"
log "CDN repo : $CDN_REPO"
log "Weights  : $CDN_WEIGHTS"
log "Out dir  : $STATE_DIR"
log "Chunk sz : $CHUNK_SIZE"
log "Reuse chunk outputs: $REUSE_CHUNK_OUTPUTS"

if [[ "$REUSE_CHUNK_OUTPUTS" == "1" ]]; then
  log "Rebuilding labels from existing chunk_outputs (skip CDN inference)"
  "$PYTHON" "$ROOT_DIR/advPhase2_rebuild_labels_from_chunks.py" \
    --state_dir "$STATE_DIR" \
    --score_thresh "$SCORE_THRESH" \
    --subject_score_thresh "$SUBJECT_SCORE_THRESH" \
    --pos_verb_thresh "$POS_VERB_THRESH" \
    --neg_verb_thresh "$NEG_VERB_THRESH" \
    --dist_thresh "$DIST_THRESH" \
    --out_labels "$OUT_LABELS" \
    --out_high_labels "$OUT_HIGH_LABELS" \
    --out_low_frames "$OUT_LOW_FRAMES" \
    --out_high_frames "$OUT_HIGH_FRAMES"
else
  #Phase 4 samples-6400
  #phase 5 samples-12800
  log "Sampling frames from CLEAR_ROOT_original -> CLEAR_ROOT"
  "$PYTHON" "$ROOT_DIR/advPhase2_sample_frames.py" \
    --input_root "$CLEAR_ROOT_ORIGINAL" \
    --output_root "$CLEAR_ROOT" \
    --sample_size "${PHASE2_SAMPLE_SIZE:-12800}" \
    --seed "${PHASE2_SAMPLE_SEED:-42}" \
    --exclude_hico_json "$EXCLUDE_HICO_JSON" \
    --clean_output

  # Sanity: ensure frames exist
  if ! find -L "$CLEAR_ROOT" -type f -name '*.jpg' -print -quit | grep -q .; then
    log "ERROR: No .jpg frames found under $CLEAR_ROOT. Lower blur threshold or point CLEAR_ROOT to existing frames."
    exit 1
  fi

  "$PYTHON" "$ROOT_DIR/advPhase2_cdn_autolabel.py" \
    --clear_root "$CLEAR_ROOT" \
    --state_dir "$STATE_DIR" \
    --cdn_repo "$CDN_REPO" \
    --cdn_weights "$CDN_WEIGHTS" \
    --device "$DEVICE" \
    --cdn_extra_args "$CDN_EXTRA_ARGS" \
    --num_obj_classes "$NUM_OBJ_CLASSES" \
    --num_verb_classes "$NUM_VERB_CLASSES" \
    --score_thresh "$SCORE_THRESH" \
    --subject_score_thresh "$SUBJECT_SCORE_THRESH" \
    --pos_verb_thresh "$POS_VERB_THRESH" \
    --neg_verb_thresh "$NEG_VERB_THRESH" \
    --dist_thresh "$DIST_THRESH" \
    --chunk_size "$CHUNK_SIZE" \
    --out_pred "$OUT_PRED" \
    --out_std "$OUT_STD" \
    --out_labels "$OUT_LABELS" \
    --out_low_frames "$OUT_LOW_FRAMES" \
    --out_high_frames "$OUT_HIGH_FRAMES" \
    --out_high_labels "$OUT_HIGH_LABELS"
fi

# Phase2 -
#   --subj_thresh "${SUBJ_THRESH:-0.5}" \
#   --obj_thresh "${OBJ_THRESH:-0.5}" \
# Phase3-
#   --subj_thresh "${SUBJ_THRESH:-0.3}" \
#   --obj_thresh "${OBJ_THRESH:-0.3}" \
# Phase4-
#   --subj_thresh "${SUBJ_THRESH:-0.3}" \
#   --obj_thresh "${OBJ_THRESH:-0.3}" \
# 1280 - samples
# Phase5-
#   --subj_thresh "${SUBJ_THRESH:-0.3}" \
#   --obj_thresh "${OBJ_THRESH:-0.3}" \
#
log "Sampling low-confidence frames/labels"
"$PYTHON" "$ROOT_DIR/advPhase2_sample_low_conf.py" \
  --low_conf_frames "$OUT_LOW_FRAMES" \
  --labels_csv "$OUT_LABELS" \
  --output_dir "$OUT_SAMPLE_DIR" \
  --sample_size "${SAMPLE_SIZE:-2560}" \
  --pos_thr "$POS_VERB_THRESH" \
  --neg_thr "$NEG_VERB_THRESH" \
  --dist_thr "$DIST_THRESH" \
  --subj_thresh "${SUBJ_THRESH:-0.3}" \
  --obj_thresh "${OBJ_THRESH:-0.3}" \
  --seed "${SEED:-42}"

log "Done."
