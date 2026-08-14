#!/bin/bash
# bash "advPhase1_SemiAutomtic_Data_Labeling_FramesExtraction.sh"
# Advanced Phase 1: extract frames (4 fps) from full videos and filter out blurry frames.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
PYTHON=${PYTHON:-python3.12}

# ---- Paths (override via env) ----
VIDEOS_INPUT="${VIDEOS_INPUT:-/home/mereddd/CCAT_Opensource_work/Advanced/data_full_videos}"
FRAMES_ROOT="${FRAMES_ROOT:-/home/mereddd/CCAT_Opensource_work/Advanced/data_full_videos_frames/complete_frames}"

CLEAR_ROOT="${CLEAR_ROOT:-/home/mereddd/CCAT_Opensource_work/Advanced/data_full_videos_frames/clear_frames}"

# ---- Extraction settings ----
TARGET_FPS=${TARGET_FPS:-4}

# ---- Blur filter settings ----
THRESHOLD=${THRESHOLD:-400}
ROI_THRESHOLD=${ROI_THRESHOLD:-400}
SEED=${SEED:-42}
BLUR_WORKERS=${BLUR_WORKERS:-4}

log() { echo "[$(date +'%F %T')] $*"; }

#############
# 1) Extract frames @ ~4 fps
#############
extract_frames() {
  local input_root="$VIDEOS_INPUT"
  local output_root="$FRAMES_ROOT"

  input_root=${input_root%/}
  output_root=${output_root%/}

  mkdir -p "$output_root"

  # Skip extraction if frames already exist
  if find "$output_root" -type f -name '*.jpg' -print -quit | grep -q .; then
    log "[Extract] Skipping: frames already present under $output_root"
    return
  fi

  mapfile -t video_dirs < <(
    find "$input_root" -type f \
      \( -iname '*.mp4' -o -iname '*.3gp' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) |
      sed 's#/[^/]*$##' |
      sort -u
  )

  if [ ${#video_dirs[@]} -eq 0 ]; then
    echo "No videos found under $input_root" >&2
    exit 1
  fi

  log "[Extract] found ${#video_dirs[@]} video dirs under $input_root"

  for dir in "${video_dirs[@]}"; do
    local rel_dir=${dir#"$input_root"}
    rel_dir=${rel_dir#/}
    local out_dir="$output_root"
    if [ -n "$rel_dir" ]; then
      out_dir="$output_root/$rel_dir"
    fi
    mkdir -p "$out_dir"
    log "[Extract] $dir -> $out_dir (fps=$TARGET_FPS)"
    "$PYTHON" "$ROOT_DIR/1_Frames_Extraction.py" --input "$dir" --output "$out_dir" --fps "$TARGET_FPS"
  done
}

#############
# 2) Remove blurry frames (optional sampling)
#############
remove_blur() {
  args=(
    --input_root "$FRAMES_ROOT"
    --output_root "$CLEAR_ROOT"
    --threshold "$THRESHOLD"
    --roi_threshold "$ROI_THRESHOLD"
    --seed "$SEED"
    --workers "$BLUR_WORKERS"
  )
  log "[Blur] input=$FRAMES_ROOT clear=$CLEAR_ROOT (ROI checks disabled; sampling disabled)"
  "$PYTHON" "$ROOT_DIR/2_sample_nonblurry.py" "${args[@]}"
}

main() {
  extract_frames
  remove_blur
  log "All steps finished."
}

main "$@"
