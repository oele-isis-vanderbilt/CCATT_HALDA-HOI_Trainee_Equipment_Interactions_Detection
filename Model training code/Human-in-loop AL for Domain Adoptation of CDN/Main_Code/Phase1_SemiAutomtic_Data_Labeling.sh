#!/bin/bash
# Unified entrypoint orchestrating per-step scripts (one file per logic).
# Steps:
#   0) Frame extraction (1_Frames_Extraction.py)
#   1) Blur removal + sampling (2_sample_nonblurry.py)
#   2) YOLO person detection + fixed ROIs (3_yolo_person_detect_and_attach_rois.py)
#   3) Bootstrap (4_bootstrap_agree_labels.py)
#   4) Active learning (optional) (5_active_learning_iteration.py)


set -euo pipefail

#################################
# Paths and core configuration  #
#################################
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3.12}
# Conda activation (optional). Set USE_OPENMM_ENV=1 to force openmm-cuda-env; default uses HOICLIP if available.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base)
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh" || true
  if [ "${USE_OPENMM_ENV:-0}" = "1" ]; then
    if conda activate openmm-cuda-env 2>/dev/null; then
      echo "[Env] Activated conda env: openmm-cuda-env"
    else
      echo "[Env] WARNING: openmm-cuda-env not available; continuing with current env"
    fi
  else
    if conda activate HOICLIP 2>/dev/null; then
      echo "[Env] Activated conda env: HOICLIP"
    else
      echo "[Env] WARNING: HOICLIP env not available; continuing with current env"
    fi
  fi
else
  echo "WARNING: conda not found; ensure required packages are available" >&2
fi
# GPU selection (set CUDA_DEVICE env var; maps to CUDA_VISIBLE_DEVICES and CDN device arg)
CUDA_DEVICE=${CUDA_DEVICE:-0}

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# ---- Edit these to match your environment ----
# Camera-specific roots (runs both 1-4 and 1-1 by default; override with CAMERA_VIEWS env)
# Note: default splits into two array entries; set CAMERA_VIEWS="1-4" to limit to one.
CAMERA_VIEWS=(${CAMERA_VIEWS:-1-4 1-1})
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$ROOT_DIR/Semiautomaticdata/video_smaples/train}"

set_camera_paths() {
  local view="$1"
  case "$view" in
    "1-4")
      VIDEOS_ROOT="/home/mereddd/CCAT_Opensource_work/Jan_test_runs/data/raw_videos/1-4-Videos/train"  # directory containing videos
      OUTPUT_ROOT="$OUTPUT_ROOT_BASE/1-4"  # where all artifacts are written
      SECOND_EQUIPMENT="MV"
      ;;
    "1-1")
      VIDEOS_ROOT="/home/mereddd/CCAT_Opensource_work/Jan_test_runs/data/raw_videos/train"  # directory containing videos
      OUTPUT_ROOT="$OUTPUT_ROOT_BASE/1-1"  # where all artifacts are written
      SECOND_EQUIPMENT="Propac"
      ;;
    *)
      echo "Unsupported CAMERA_VIEW '$view'. Use 1-4 or 1-1." >&2
      exit 1
      ;;
  esac
  CAMERA_VIEW="$view"
  EQUIPMENT_TYPES=("IV Pump" "$SECOND_EQUIPMENT")
  PHASE0_OUT="$OUTPUT_ROOT/phase0"
  PHASE1_OUT="$OUTPUT_ROOT/phase1"
  LOG_DIR="$OUTPUT_ROOT/logs"
}
ROI_CONFIG="$ROOT_DIR/camera_rois.json"                      # fixed ROI coordinates (edit file)
# Optional zoom override for camera_view=1-1 (use: zoom_level_2/3/4); leave empty for default/auto
ROI_ZOOM_LEVEL="${ROI_ZOOM_LEVEL:-}"
# Optional JSON mapping of video_name -> zoom_level (zoom_level_2/3/4) for 1-1 view
DEFAULT_ZOOM_MAP_FILE="$ROOT_DIR/roi_zoom_map_1-1.json"
if [ -z "${ROI_ZOOM_MAP_FILE:-}" ] && [ -f "$DEFAULT_ZOOM_MAP_FILE" ]; then
  ROI_ZOOM_MAP_FILE="$DEFAULT_ZOOM_MAP_FILE"
else
  ROI_ZOOM_MAP_FILE="${ROI_ZOOM_MAP_FILE:-}"
fi

# Phase 0 settings
FPS=4.0                                                      # target FPS; set 0 to disable and use EVERY_N
EVERY_N=1                                                    # used when FPS==0
BLUR_THRESH=150                                              # Laplacian variance; below is blurry
ROI_BLUR_THRESH=200                                          # Laplacian variance inside equipment ROIs (more aggressive)
SAMPLE_FRAMES=100 #100                                            # per-video sample count (after blur). Set 0 to keep all.
SAMPLE_SEED=12
YOLO_WEIGHTS="${YOLO_WEIGHTS:-/home/mereddd/CCAT_Opensource_work/yolo11l.pt}"
YOLO_CONF=0.25
YOLO_IMGSZ=640

# Phase 1 (bootstrap) CDN + MMPose
CDN_REPO="/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/"                                      # CDN repository path
CDN_PRETRAINED="/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/pretrained/hico_cdn_s.pth"          # CDN HICO weights
# Override via CDN_EXTRA_ARGS env; default uses selected CUDA_DEVICE
CDN_EXTRA_ARGS=${CDN_EXTRA_ARGS:---device cuda:${CUDA_DEVICE}}
# CDN inference helper inputs
OBJECT_THRESHOLDS="${OBJECT_THRESHOLDS:-$ROOT_DIR/object_thresholds.json}"
OBJECT_CROPS="${OBJECT_CROPS:-$ROOT_DIR/object_crops.json}"
CDN_POS_VERBS=${CDN_POS_VERBS:-"9 37"}            # HICO hold/carry
CDN_NEG_VERBS=${CDN_NEG_VERBS:-"113 58"}          # watch, no_interaction
# MMPose (required): defaults pointed at your MMPose repo; override as needed
MMPOSE_ROOT=${MMPOSE_ROOT:-"/home/mereddd/HandDetection_opensource_work/Embodied_Learning/BJED2026Jan/RTMPose_MMPOSE_based action recog_Dec302025/mmpose"}
if [ -z "${MMPOSE_CONFIG:-}" ] && [ -n "$MMPOSE_ROOT" ]; then
  # Try expected path first
  CAND_CFG="$MMPOSE_ROOT/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
  if [ -f "$CAND_CFG" ]; then
    MMPOSE_CONFIG="$CAND_CFG"
  else
    # Attempt auto-discovery of the config inside MMPOSE_ROOT
    FOUND_CFG=$(find "$MMPOSE_ROOT" -type f -name "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py" -print -quit 2>/dev/null || true)
    if [ -n "$FOUND_CFG" ]; then
      MMPOSE_CONFIG="$FOUND_CFG"
    fi
  fi
fi
# If MMPOSE_CONFIG is set but missing, attempt fallback search under MMPOSE_ROOT.
if [ -n "${MMPOSE_CONFIG:-}" ] && [ ! -f "$MMPOSE_CONFIG" ] && [ -n "$MMPOSE_ROOT" ]; then
  FOUND_CFG=$(find "$MMPOSE_ROOT" -type f -name "td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py" -print -quit 2>/dev/null || true)
  [ -n "$FOUND_CFG" ] && MMPOSE_CONFIG="$FOUND_CFG"
fi
MMPOSE_CONFIG=${MMPOSE_CONFIG:-""}
if [ -z "${MMPOSE_CHECKPOINT:-}" ]; then
  LOCAL_PTH="$ROOT_DIR/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
  if [ -f "$LOCAL_PTH" ]; then
    MMPOSE_CHECKPOINT="$LOCAL_PTH"
  else
    MMPOSE_CHECKPOINT="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
  fi
fi
MMPOSE_DEVICE=${MMPOSE_DEVICE:-cuda:${CUDA_DEVICE}}
MMPOSE_MIN_CONF=${MMPOSE_MIN_CONF:-0.25}
MMPOSE_MIN_HITS=${MMPOSE_MIN_HITS:-1}
MMPOSE_CMD=${MMPOSE_CMD:-"conda run -n openmm-cuda-env python"}
# Optional: set MMPOSE_PREDS_CSV to merge precomputed MMPose labels for agreement.
NO_INTERACTION_ID=58
VALID_INTERACTION_ID=118
CDN_POS_THRESH=0.00000
CDN_NEG_THRESH=0.00000
SUBJ_IOU=0.6
OBJ_IOU=0.3

#################################
# Helpers
#################################
log() { echo "[$(date +'%F %T')] $*"; }

#################################
# Phase 0
#################################
run_phase0() {
  log "Phase 0: frames → blur filter → YOLO persons → fixed ROIs"
  mkdir -p "$LOG_DIR" "$PHASE0_OUT"

  FRAMES_DIR="$PHASE0_OUT/frames"
  CLEAR_DIR="$PHASE0_OUT/2_clear_frames"
  SAMPLE_DIR="$PHASE0_OUT/2_sampled_frames"
  DET_DIR="$PHASE0_OUT/3_detections"
  mkdir -p "$DET_DIR"

  # Frame extraction via standalone script (skip only if all videos already extracted)
  need_extract=1
  if [ -d "$FRAMES_DIR" ]; then
    need_extract=0
    shopt -s nullglob
    video_files=("$VIDEOS_ROOT"/*.{mp4,MP4,3gp,3GP,mov,MOV,mkv,MKV,avi,AVI})
    shopt -u nullglob
    for vf in "${video_files[@]}"; do
      stem="$(basename "${vf%.*}")"
      if ! compgen -G "$FRAMES_DIR/$stem/*.jpg" > /dev/null; then
        need_extract=1
        break
      fi
    done
    if [ "$need_extract" -eq 0 ]; then
      log "Phase 0: skipping frame extraction, found frames for all videos at $FRAMES_DIR"
    fi
  fi

  if [ "$need_extract" -eq 1 ]; then
    log "Phase 0: extracting frames with 1_Frames_Extraction.py"
    EXTRACT_ARGS=(--input "$VIDEOS_ROOT" --output "$FRAMES_DIR")
    if [ "$FPS" != "0" ] && [ "$FPS" != "0.0" ]; then
      EXTRACT_ARGS+=(--fps "$FPS")
    else
      EXTRACT_ARGS+=(--every_n "$EVERY_N")
    fi
    $PYTHON "$ROOT_DIR/1_Frames_Extraction.py" "${EXTRACT_ARGS[@]}" | tee -a "$LOG_FILE"
  fi

  # Quick visualization: copy a few extracted frames per video (1_viz_copy_samples.py)
  VIZ_FRAMES_DIR="$PHASE0_OUT/1_viz_frames"
  mkdir -p "$VIZ_FRAMES_DIR"
  if compgen -G "$VIZ_FRAMES_DIR/*/*.jpg" > /dev/null; then
    log "Phase 0: skipping viz_frames copy (1_viz_copy_samples.py; already exists)"
  else
    $PYTHON "$ROOT_DIR/1_viz_copy_samples.py" \
      --input_root "$FRAMES_DIR" \
      --output_root "$VIZ_FRAMES_DIR" \
      --count 3 | tee -a "$LOG_FILE"
  fi

  # Blur removal + sampling
  if compgen -G "$SAMPLE_DIR/*/*.jpg" > /dev/null; then
    log "Phase 0: skipping blur+sample (found sampled frames in $SAMPLE_DIR)"
  else
    log "Phase 0: removing blurry frames and sampling ($SAMPLE_FRAMES per video)"
    $PYTHON "$ROOT_DIR/2_sample_nonblurry.py" \
      --input_root "$FRAMES_DIR" \
      --output_root "$CLEAR_DIR" \
      --sample_out_root "$SAMPLE_DIR" \
      --threshold "$BLUR_THRESH" \
      --roi_config "$ROI_CONFIG" \
      --camera_view "$CAMERA_VIEW" \
      --equipment_types "${EQUIPMENT_TYPES[@]}" \
      --roi_threshold "$ROI_BLUR_THRESH" \
      --sample "$SAMPLE_FRAMES" \
      --seed "$SAMPLE_SEED" | tee -a "$LOG_FILE"
  fi

  # Quick visualization: copy a few sampled frames per video (2_viz_copy_samples.py)
  VIZ_SAMPLED_DIR="$PHASE0_OUT/2_viz_sampled"
  mkdir -p "$VIZ_SAMPLED_DIR"
  if compgen -G "$VIZ_SAMPLED_DIR/*/*.jpg" > /dev/null; then
    log "Phase 0: skipping viz_sampled copy (2_viz_copy_samples.py; already exists)"
  else
    $PYTHON "$ROOT_DIR/2_viz_copy_samples.py" \
      --input_root "$SAMPLE_DIR" \
      --output_root "$VIZ_SAMPLED_DIR" \
      --count 3 | tee -a "$LOG_FILE"
  fi

  # YOLO person detection + attach fixed ROIs per video
  for VIDEO_PATH in "$SAMPLE_DIR"/*; do
    [ -d "$VIDEO_PATH" ] || continue
    VIDEO_NAME="$(basename "$VIDEO_PATH")"
    OUT_JSON="$DET_DIR/${VIDEO_NAME}.json"
    zoom_arg=()
    zoom_map_arg=()
    if [ "$CAMERA_VIEW" = "1-1" ] && [ -n "$ROI_ZOOM_LEVEL" ]; then
      zoom_arg=(--zoom_level "$ROI_ZOOM_LEVEL")
    fi
    if [ "$CAMERA_VIEW" = "1-1" ] && [ -n "$ROI_ZOOM_MAP_FILE" ]; then
      zoom_map_arg=(--zoom_map "$ROI_ZOOM_MAP_FILE")
    fi
    if [ -s "$OUT_JSON" ]; then
      log "Phase 0: skipping detection for $VIDEO_NAME (exists)"
      continue
    fi
    $PYTHON "$ROOT_DIR/3_yolo_person_detect_and_attach_rois.py" \
      --image_dir "$VIDEO_PATH" \
      --output_json "$OUT_JSON" \
      --camera_view "$CAMERA_VIEW" \
      --video_name "$VIDEO_NAME" \
      --roi_config "$ROI_CONFIG" \
      --equipment_types "${EQUIPMENT_TYPES[@]}" \
      "${zoom_arg[@]}" \
      "${zoom_map_arg[@]}" \
      --yolo_weights "$YOLO_WEIGHTS" \
      --conf "$YOLO_CONF" \
      --imgsz "$YOLO_IMGSZ" | tee -a "$LOG_FILE"
  done

  # Visualization of detections and ROIs (first few frames per video) (3_viz_draw_detections.py)
  VIZ_DET_DIR="$PHASE0_OUT/3_viz_detections"
  mkdir -p "$VIZ_DET_DIR"
  if compgen -G "$VIZ_DET_DIR/*/*.jpg" > /dev/null; then
    log "Phase 0: skipping viz_detections (3_viz_draw_detections.py; already exists)"
  else
    $PYTHON "$ROOT_DIR/3_viz_draw_detections.py" \
      --frames_root "$SAMPLE_DIR" \
      --detections_root "$DET_DIR" \
      --output_root "$VIZ_DET_DIR" \
      --count 3 | tee -a "$LOG_FILE"
  fi
}

#################################
# Phase 1 (bootstrap)
#################################
run_bootstrap_all() {
  log "Phase 1: CDN + MMPose (agreement when both available; fallback to CDN)"
  shopt -s nullglob
  if [ -z "$MMPOSE_CONFIG" ] || [ -z "$MMPOSE_CHECKPOINT" ]; then
    log "ERROR: MMPOSE_CONFIG and MMPOSE_CHECKPOINT must be set before running bootstrap."
    log "Hint: export MMPOSE_ROOT to your mmpose repo (e.g., /Users/.../RTMPose_MMPOSE_based action recog_Dec302025/mmpose) or set MMPOSE_CONFIG directly."
    exit 1
  fi
  if [ ! -f "$MMPOSE_CONFIG" ]; then
    case "$MMPOSE_CONFIG" in
      http://*|https://*)
        log "WARNING: MMPOSE_CONFIG is a URL; please provide a local config path."
        ;;
      *)
        log "ERROR: MMPOSE_CONFIG not found at $MMPOSE_CONFIG. Set MMPOSE_CONFIG to a valid MMPose config (e.g., td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py in your mmpose repo)."
        ;;
    esac
    exit 1
  fi
  case "$MMPOSE_CHECKPOINT" in
    http://*|https://*)
      log "WARNING: MMPOSE_CHECKPOINT is a URL; it will be downloaded at runtime. If network is blocked, set MMPOSE_CHECKPOINT to a local .pth file instead."
      ;;
    *)
      if [ ! -f "$MMPOSE_CHECKPOINT" ]; then
        log "ERROR: MMPOSE_CHECKPOINT not found at $MMPOSE_CHECKPOINT. Please provide a local .pth checkpoint."
        exit 1
      fi
      ;;
  esac

  local det_root="$PHASE0_OUT/3_detections"
  local frames_root="$PHASE0_OUT/2_sampled_frames"

  for DET_JSON in "$det_root"/*.json; do
    VIDEO_NAME="$(basename "$DET_JSON" .json)"

    if [ -d "$frames_root/$VIDEO_NAME" ]; then
      FRAMES_DIR="$frames_root/$VIDEO_NAME"
    elif [ -d "$PHASE0_OUT/2_clear_frames/$VIDEO_NAME" ]; then
      FRAMES_DIR="$PHASE0_OUT/2_clear_frames/$VIDEO_NAME"
    else
      FRAMES_DIR="$PHASE0_OUT/frames/$VIDEO_NAME"
    fi

    OUT_DIR="$PHASE1_OUT/4_$VIDEO_NAME"
    if [ -s "$OUT_DIR/bootstrap_summary.json" ]; then
      log "Bootstrap: skipping $VIDEO_NAME (existing summary at $OUT_DIR/bootstrap_summary.json)"
      continue
    fi
    mkdir -p "$OUT_DIR"
    CDN_PREDS="$OUT_DIR/cdn_preds.csv"

    log "Bootstrap -> $VIDEO_NAME (frames: $FRAMES_DIR)"
    # Run CDN predictions via helper script (required)
    $PYTHON "$ROOT_DIR/4.1_cdn_predictions.py" \
      --frames_dir "$FRAMES_DIR" \
      --detections_json "$DET_JSON" \
      --roi_config "$ROI_CONFIG" \
      --camera_view "$CAMERA_VIEW" \
      --equipment_types "${EQUIPMENT_TYPES[@]}" \
      --cdn_repo "$CDN_REPO" \
      --cdn_weights "$CDN_PRETRAINED" \
      --cdn_extra_args "$CDN_EXTRA_ARGS" \
      --object_thresholds "$OBJECT_THRESHOLDS" \
      --object_crops "$OBJECT_CROPS" \
      --positive_verb_ids ${CDN_POS_VERBS} \
      --negative_verb_ids ${CDN_NEG_VERBS} \
      --subj_iou_thresh "$SUBJ_IOU" \
      --obj_iou_thresh "$OBJ_IOU" \
      --output_csv "$CDN_PREDS" | tee -a "$LOG_FILE"

    CMD=(
      "$PYTHON" "$ROOT_DIR/4_bootstrap_agree_labels.py"
      --frames_dir "$FRAMES_DIR"
      --detections_json "$DET_JSON"
      --roi_config "$ROI_CONFIG"
      --camera_view "$CAMERA_VIEW"
      --equipment_types "${EQUIPMENT_TYPES[@]}"
      --output "$OUT_DIR"
      --cdn_preds_csv "$CDN_PREDS"
      --no_interaction_id "$NO_INTERACTION_ID"
      --cdn_pos_thresh "$CDN_POS_THRESH"
      --cdn_neg_thresh "$CDN_NEG_THRESH"
      --subj_iou_thresh "$SUBJ_IOU"
      --obj_iou_thresh "$OBJ_IOU"
      --mmpose_config "$MMPOSE_CONFIG"
      --mmpose_checkpoint "$MMPOSE_CHECKPOINT"
      --mmpose_device "$MMPOSE_DEVICE"
      --mmpose_min_conf "$MMPOSE_MIN_CONF"
      --mmpose_min_hits "$MMPOSE_MIN_HITS"
      --mmpose_cmd "$MMPOSE_CMD"
      --mmpose_vis_dir "$OUT_DIR/4_viz_mmpose_skeletons"
    )
    if [ -n "${MMPOSE_PREDS_CSV:-}" ]; then
      CMD+=(--mmpose_preds_csv "$MMPOSE_PREDS_CSV")
    fi
    {
      printf "Bootstrap command:"
      printf " %q" "${CMD[@]}"
      printf "\n"
    } | tee -a "$LOG_FILE"
    "${CMD[@]}" | tee -a "$LOG_FILE"

    # Visualization of bootstrap labels (first N frames) (4_viz_bootstrap_labels.py).
  VIZ_BOOT_DIR="$OUT_DIR/4_viz_labels"
  if [ ! -d "$VIZ_BOOT_DIR" ] || ! compgen -G "$VIZ_BOOT_DIR/*.jpg" > /dev/null; then
    $PYTHON "$ROOT_DIR/4_viz_bootstrap_labels.py" \
      --frames_dir "$FRAMES_DIR" \
      --candidates_csv "$OUT_DIR/bootstrap_candidates.csv" \
      --output_dir "$VIZ_BOOT_DIR" \
      --viz_cdn_dir "$OUT_DIR/4_viz_cdn" \
      --viz_mmpose_dir "$OUT_DIR/4_viz_mmpose" \
      --limit 20 | tee -a "$LOG_FILE"
  else
    log "Bootstrap viz: skipping (already exists at $VIZ_BOOT_DIR)"
  fi
  done
  shopt -u nullglob

  # Phase 1.2: Export merged HICO-DET annotations (skips frames with disagreements).
  if [ -d "$PHASE1_OUT" ]; then
    $PYTHON "$ROOT_DIR/export_hico_annotations.py" \
      --phase1_root "$PHASE1_OUT" | tee -a "$LOG_FILE"
  fi
}

#################################
# Entry point
#################################
# If executed directly, run Phase0 + Phase1 for all configured camera views.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  for VIEW in "${CAMERA_VIEWS[@]}"; do
    set_camera_paths "$VIEW"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/pipeline_${CAMERA_VIEW}_$TIMESTAMP.log"

    log "Camera view $CAMERA_VIEW: starting pipeline"
    run_phase0
    run_bootstrap_all
  done
  log "Pipeline complete. Logs written under each camera's OUTPUT_ROOT/logs directory."
fi
