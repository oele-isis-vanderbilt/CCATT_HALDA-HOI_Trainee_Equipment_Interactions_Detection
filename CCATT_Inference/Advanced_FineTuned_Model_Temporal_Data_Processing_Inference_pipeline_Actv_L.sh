#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Step 0 of the CCATT inference pipeline: raw video -> frame-level HOI
# predictions (*_df_preds.csv), the input Step 1
# (V3_Create_Temporal_Predicted_HOI_intervals_...py) needs to produce
# temporal interaction start/end times.
#
# For each .mp4/.3gp video under --input_dir, this script:
#   1. Extracts frames (1_Frames_Extraction.py), skipping videos already
#      extracted.
#   2. Runs the CDN HOI-detection model (--eval mode) over those frames,
#      producing a df_preds.csv with the exact columns Step 1 expects
#      (filename, subject_box, object_box, subject_class, object_class,
#      verb_class, score, obj_scores, verb_scores_index_decoder).
#   3. Moves/renames that CSV to <output_dir>/hoi_results_<job_name>/
#      <camera>/<video_name>_df_preds.csv, skipping videos already done.
#
# This step needs a GPU and the CDN Python environment (torch/torchvision
# matching CDN/requirements.txt) -- it is the slow step, analogous to
# generate_role_assignment_csvs.py. Run Step 1 next on its output.
#
# Configure via environment variables (all have defaults below):
#   PRETRAINED_MODEL_PATH  path to the fine-tuned CDN checkpoint (.pth)
#   CDN_REPO_ROOT           path to the CDN repo (this project's CDN/ folder)
#   INPUT_DIR               folder of source videos
#   OUTPUT_ROOT             where extracted frames + results are written
#   JOB_NUMBER              label used to name the results subfolder
#
# Example:
#   PRETRAINED_MODEL_PATH=/path/to/checkpoint_best.pth \
#   CDN_REPO_ROOT=/path/to/CDN \
#   INPUT_DIR=/path/to/videos \
#   OUTPUT_ROOT=/path/to/output \
#   bash Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh
# -----------------------------------------------------------------------------

# Notes:
# - Results are bucketed by detected camera tag (e.g., CAM16_V2, CAM16_V1, PAN_V2, PAN_V1).
# - The pipeline predicts HOI scores for all objects; equipment-specific labels are no longer used.
# ======================================
# Configuration Parameters (override any of these as environment variables)
# ======================================
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

JOB_NUMBER="${JOB_NUMBER:-Task_adv4_phase4}"  # Change this for different jobs
CDN_REPO_ROOT="${CDN_REPO_ROOT:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4/logs_stage2_v2/checkpoint_best.pth}"
INPUT_DIR="${INPUT_DIR:-$HOME/CCAT_Opensource_work/Advanced/Training_Data/Shaggys_Videos_One_Video_at_A_Time}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/CCAT_Opensource_work/Advanced/Training_Data/Shaggys_Videos/Output}"

echo "🔧 Using Job Number: $JOB_NUMBER"
echo "🔧 Model Path: $PRETRAINED_MODEL_PATH"
echo "========================================"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
    echo "❌ Error: Pretrained model not found at $PRETRAINED_MODEL_PATH"
    exit 1
fi

# Function to run the complete processing pipeline
run_processing_pipeline() {
    local IITSEC_Testing=$1
    local INPUT_DIR=$2
    local EQUIPMENT_TYPE="Generic"
    local CAMERA_VIEW=$3
    local OUTPUT_ROOT=$4
    
    echo "========================================"
    echo "Processing: $EQUIPMENT_TYPE ($CAMERA_VIEW)"
    echo "Input: $INPUT_DIR"
    echo "Output: $OUTPUT_ROOT"
    echo "========================================"
    
    OUTPUT_ROOT_ROOT="$OUTPUT_ROOT"

    # Robust globbing: don't leave unmatched patterns literal; match case-insensitively
    shopt -s nullglob nocaseglob

    # --- Helper: detect camera tag from filename ---
    detect_camera_from_name() {
        # Sets DETECTED_CAMERA_LABEL and DETECTED_VIEW based on filename tokens
        local nm="$1"
        DETECTED_CAMERA_LABEL=""
        DETECTED_VIEW=""
        if echo "$nm" | grep -Eqi 'CAM16[_-]?V?2'; then
            DETECTED_CAMERA_LABEL="CAM16_V2"
            DETECTED_VIEW="view2_v2"
        elif echo "$nm" | grep -Eqi 'CAM16[_-]?V?1'; then
            DETECTED_CAMERA_LABEL="CAM16_V1"
            DETECTED_VIEW="view2_v1"
        elif echo "$nm" | grep -Eqi 'PAN[_-]?V?2'; then
            DETECTED_CAMERA_LABEL="PAN_V2"
            DETECTED_VIEW="view1_v2"
        elif echo "$nm" | grep -Eqi 'PAN[_-]?V?1'; then
            DETECTED_CAMERA_LABEL="PAN_V1"
            DETECTED_VIEW="view1_v1"
        fi
    }
    
    # Create all necessary directories upfront
    mkdir -p "$OUTPUT_ROOT"
    mkdir -p "$OUTPUT_ROOT/hoi_results_${JOB_NUMBER}"
    mkdir -p "$OUTPUT_ROOT/logs"
    mkdir -p "$OUTPUT_ROOT_ROOT/train_6_minutes_samples_Extracted_frames"

    case="${CAMERA_VIEW}-Videos_${EQUIPMENT_TYPE}"
    LOG_DIR="$OUTPUT_ROOT/logs"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/${case}pipeline_$TIMESTAMP.log"
    SUMMARY_FILE="$OUTPUT_ROOT/${case}summary_metrics_$TIMESTAMP.csv"
    echo "video_name,macro_f1_score" > "$SUMMARY_FILE"

    echo "🚀 Starting Temporal Inference Pipeline" | tee -a "$LOG_FILE"
    echo "Job Number: $JOB_NUMBER" | tee -a "$LOG_FILE"
    echo "Model Path: $PRETRAINED_MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # STEP 1: Extract frames
    VIDEO_COUNT=0
    echo "➡️  Step 1: Extract frames" | tee -a "$LOG_FILE"

    # Robust frame extraction condition: detect any JPGs under the frames root
    TARGET_FRAMES_DIR="$OUTPUT_ROOT_ROOT/train_6_minutes_samples_Extracted_frames"
    if [ -d "$TARGET_FRAMES_DIR" ] && find "$TARGET_FRAMES_DIR" -type f -name '*.jpg' -print -quit >/dev/null 2>&1; then
        echo "✅ Skipping Step 1: Frames already extracted" | tee -a "$LOG_FILE"
    else
        echo "🔄 Extracting frames..." | tee -a "$LOG_FILE"
        python "$ROOT_DIR/1_Frames_Extraction.py" --input "$INPUT_DIR" --output "$TARGET_FRAMES_DIR" | tee -a "$LOG_FILE"
        # Verify extraction
        if ! find "$TARGET_FRAMES_DIR" -type f -name '*.jpg' -print -quit >/dev/null 2>&1; then
            echo "❌ Extraction produced no frames. Check INPUT_DIR and extractor." | tee -a "$LOG_FILE"
            return 1
        fi
    fi

    # === STEP 2-3: Run per video ===
    echo "➡️  Step 2-3: Processing videos..." | tee -a "$LOG_FILE"
    
    for VIDEO_PATH in "$INPUT_DIR"/*.mp4 "$INPUT_DIR"/*.MP4 "$INPUT_DIR"/*.3gp "$INPUT_DIR"/*.3GP; do
        [ -e "$VIDEO_PATH" ] || continue
        echo "Checking: $VIDEO_PATH" | tee -a "$LOG_FILE"
        
        VIDEO_COUNT=$((VIDEO_COUNT + 1))
        VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
        # Detect camera family/version from filename (e.g., CAM16_V2, CAM16_V1, PAN_V1, PAN_V2)
        detect_camera_from_name "$VIDEO_NAME"
        if [ -n "${DETECTED_CAMERA_LABEL:-}" ]; then
            echo "📷 Camera detected: $DETECTED_CAMERA_LABEL  → view: ${DETECTED_VIEW:-n/a}" | tee -a "$LOG_FILE"
        else
            echo "📷 Camera detected: unknown (falling back to declared CAMERA_VIEW=$CAMERA_VIEW)" | tee -a "$LOG_FILE"
        fi
        # Resolve frame path robustly (handles extractor sanitization of names)
        TARGET_FRAMES_DIR="$OUTPUT_ROOT_ROOT/train_6_minutes_samples_Extracted_frames"
        CANDIDATES=(
          "$TARGET_FRAMES_DIR/$VIDEO_NAME"
          "$TARGET_FRAMES_DIR/$(echo "$VIDEO_NAME" | tr ' ' '_' )"
          "$TARGET_FRAMES_DIR/$(echo "$VIDEO_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
          "$TARGET_FRAMES_DIR/$(echo "$VIDEO_NAME" | sed -E 's/_V[12]\b//I')"
          "$TARGET_FRAMES_DIR/$(echo "$VIDEO_NAME" | tr ' ' '_' | sed -E 's/_V[12]\b//I')"
        )
        FRAME_PATH=""
        for c in "${CANDIDATES[@]}"; do
          if [ -d "$c" ] && find "$c" -type f -name '*.jpg' -print -quit >/dev/null; then
            FRAME_PATH="$c"
            break
          fi
        done
        # Fallback: try to find a near-match directory (case-insensitive)
        if [ -z "$FRAME_PATH" ]; then
          FRAME_PATH="$(find "$TARGET_FRAMES_DIR" -maxdepth 1 -type d -iname "*$(echo "$VIDEO_NAME" | tr ' ' '_')*" -print -quit || true)"
        fi
        if [ -z "$FRAME_PATH" ] || ! find "$FRAME_PATH" -type f -name '*.jpg' -print -quit >/dev/null; then
          echo "📸 Frames missing for $VIDEO_NAME → extracting now..." | tee -a "$LOG_FILE"
          # Normalize a target folder name (remove spaces and version tokens)
          SAFE_VIDEO_NAME="$(echo "$VIDEO_NAME" | tr ' ' '_' | sed -E 's/_V[12]\b//I')"
          FRAME_PATH="$TARGET_FRAMES_DIR/$SAFE_VIDEO_NAME"
          mkdir -p "$FRAME_PATH"
          # Try python extractor limited to this file; fallback to ffmpeg
          if python "$ROOT_DIR/1_Frames_Extraction.py" --input "$VIDEO_PATH" --output "$FRAME_PATH" 2>>"$LOG_FILE"; then
            :
          else
            # ffmpeg -hide_banner -loglevel error -y -i "$VIDEO_PATH" -vf "fps=4,scale=-2:720" "$FRAME_PATH/%06d.jpg" || true
            ffmpeg -hide_banner -loglevel error -y -i "$VIDEO_PATH" -vf "fps=4" "$FRAME_PATH/%06d.jpg"
          fi
          # Verify again
          if ! find "$FRAME_PATH" -type f -name '*.jpg' -print -quit >/dev/null; then
            echo "❌ Still no frames for $VIDEO_NAME at $FRAME_PATH" | tee -a "$LOG_FILE"
            echo "----------------------------------" | tee -a "$LOG_FILE"
            continue
          fi
        fi
        echo "🧩 Using frames from: $FRAME_PATH" | tee -a "$LOG_FILE"
        
        OUTPUT_FOLDER="$OUTPUT_ROOT/$VIDEO_NAME"
        mkdir -p "$OUTPUT_FOLDER"
        # Prepare results directory (bucket by detected camera label if available)
        if [ -n "${DETECTED_CAMERA_LABEL:-}" ]; then
            RESULT_DIR="$OUTPUT_ROOT/hoi_results_${JOB_NUMBER}/${DETECTED_CAMERA_LABEL}"
        else
            RESULT_DIR="$OUTPUT_ROOT/hoi_results_${JOB_NUMBER}/UnknownCamera"
        fi
        mkdir -p "$RESULT_DIR"

        echo "🔄 Processing Video $VIDEO_COUNT: $VIDEO_NAME" | tee -a "$LOG_FILE"
        
        # Check if already processed
        if [ -s "$RESULT_DIR/${VIDEO_NAME}_df_preds.csv" ]; then
            echo "✅ Skipping: HOI detection already done for $VIDEO_NAME" | tee -a "$LOG_FILE"
        else
            echo "🔄 Running CDN HOI detection for $VIDEO_NAME..." | tee -a "$LOG_FILE"
            
            # Change to CDN directory
            cd "$CDN_REPO_ROOT"
            # Run CDN inference
            python main.py \
              --pretrained "$PRETRAINED_MODEL_PATH" \
              --dataset_file hico \
              --hoi_path "$FRAME_PATH" \
              --num_obj_classes 92 \
              --num_verb_classes 118 \
              --backbone resnet50 \
              --num_queries 64 \
              --dec_layers_hopd 3 \
              --dec_layers_interaction 3 \
              --eval \
              --use_nms_filter 2>&1 | tee -a "$LOG_FILE"

            # Move and rename prediction CSV
            DEST_CSV="$RESULT_DIR/${VIDEO_NAME}_df_preds.csv"
            # Some cluster setups write df_preds.csv to a fixed scratch path instead of CDN's
            # own working directory; set TMP_DF_PREDS to that path if yours does the same.
            TMP_DF_PREDS="${TMP_DF_PREDS:-}"
            MOVE_OK=0
            if [ -n "$TMP_DF_PREDS" ] && [ -f "$TMP_DF_PREDS" ]; then
                echo "➡️  Moving prediction CSV for $VIDEO_NAME" | tee -a "$LOG_FILE"
                mv "$TMP_DF_PREDS" "$DEST_CSV" && MOVE_OK=1
            elif [ -f "./df_preds.csv" ]; then
                echo "➡️  Moving prediction CSV for $VIDEO_NAME" | tee -a "$LOG_FILE"
                mv "./df_preds.csv" "$DEST_CSV" && MOVE_OK=1
            else
                echo "❌ Error: No prediction file generated for $VIDEO_NAME" | tee -a "$LOG_FILE"
            fi

            if [ "$MOVE_OK" -eq 1 ] && [ -s "$DEST_CSV" ]; then
                echo "✅ Completed: $VIDEO_NAME" | tee -a "$LOG_FILE"
                if [ -d "$FRAME_PATH" ]; then
                    rm -rf "$FRAME_PATH"
                    echo "🧹 Deleted extracted frames for $VIDEO_NAME: $FRAME_PATH" | tee -a "$LOG_FILE"
                fi
            elif [ "$MOVE_OK" -eq 1 ]; then
                echo "⚠️ Moved CSV is empty for $VIDEO_NAME: $DEST_CSV" | tee -a "$LOG_FILE"
            fi
            
            # Return to original directory
            cd "$ROOT_DIR"
        fi

        echo "Output saved to: $RESULT_DIR" | tee -a "$LOG_FILE"
        echo "----------------------------------" | tee -a "$LOG_FILE"
    done

    # Final summary
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "❌ No .mp4 or .3gp videos found in $INPUT_DIR" | tee -a "$LOG_FILE"
    else
        echo "✅ Processed $VIDEO_COUNT video(s) for $EQUIPMENT_TYPE." | tee -a "$LOG_FILE"
    fi
    
    echo "📊 Summary CSV: $SUMMARY_FILE" | tee -a "$LOG_FILE"
    echo "📁 Results saved to: $OUTPUT_ROOT/hoi_results_${JOB_NUMBER}/" | tee -a "$LOG_FILE"
}

# ======================================
# Main Execution
# ======================================

echo "🚀 Starting pipeline with $JOB_NUMBER model..."
echo "========================================"

run_processing_pipeline \
    1 \
    "$INPUT_DIR" \
    '2' \
    "$OUTPUT_ROOT"

echo "✅ Pipeline completed!"
echo "========================================"

echo "🎉 All processing completed successfully!"
echo "📊 Check individual log files for detailed results."