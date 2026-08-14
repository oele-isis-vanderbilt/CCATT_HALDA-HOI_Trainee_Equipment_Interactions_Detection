#---------------------------------------------#
# mkdir -p /home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/Annotations/{images,phase4.1_Labeled_Data,annotations}
# scp "/Users/divyamereddy/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2_phase4/merged_hico_annotations_merged.json" mereddd@hyper13.isis.vanderbilt.edu:"/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/Annotations/phase4.1_Labeled_Data"
# nohup bash "advPhase2_Training_Dataprep & Training.sh" > log_advphase4.1_Dataprep.out 2>&1 & 
# tail -f log_advphase4.1_Dataprep.out

#---------------------------------------------#
#!/bin/bash

# after this merge phase1 and phase2 annotations 
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

# Phase-2 run root: keep all data for this run in one place.
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/Annotations}"
ANNOT_JSON="${ANNOT_JSON:-$ANNOTATIONS_ROOT/phase4.1_Labeled_Data/merged_hico_annotations_merged.json}"
# PHASE1_JSON="${PHASE1_JSON:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase1/Annotations/annotations/trainval_hico.json}"
#PHASE2_JSON="${PHASE2_JSON:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase2/Annotations/annotations/trainval_hico.json}"
PHASE3_JSON="${PHASE3_JSON:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase3/Annotations/annotations/trainval_hico.json}"

ANNOT_DIR="$(dirname "$ANNOT_JSON")"
ANNOT_OUT="${ANNOT_OUT:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_with_aug.json}"
ANNOT_SANITIZED="${ANNOT_SANITIZED:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_verb118.json}"

# Images root (override if your train images are elsewhere).
IMAGES_ROOT="${IMAGES_ROOT:-$ANNOTATIONS_ROOT/images/train2015}"
PHASE4_SAMPLED_ROOT="${PHASE4_SAMPLED_ROOT:-$HOME/CCAT_Opensource_work/Advanced/data_full_videos_frames/clear_frames_phase4_sample}"
SYNC_PHASE4_IMAGES="${SYNC_PHASE4_IMAGES:-1}"
RUN_REMOTE_MERGE="${RUN_REMOTE_MERGE:-1}"
PHASE4_AUG_JSON="${PHASE4_AUG_JSON:-$ANNOT_OUT}"
MERGED_TRAINVAL_JSON="${MERGED_TRAINVAL_JSON:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase4.1/Annotations/trainval_hico.json}"
RUN_FINAL_VIZ="${RUN_FINAL_VIZ:-1}"
FINAL_VIZ_SAMPLES="${FINAL_VIZ_SAMPLES:-100}"
FINAL_VIZ_MIN_POSITIVE="${FINAL_VIZ_MIN_POSITIVE:-30}"
FINAL_VIZ_SEED="${FINAL_VIZ_SEED:-123}"
FINAL_VIZ_OUTPUT_DIR="${FINAL_VIZ_OUTPUT_DIR:-$ANNOTATIONS_ROOT/viz_sample_final_annotations}"

# Augmentation settings (same script as Phase 1).
# AUG_PER_IMAGE=${AUG_PER_IMAGE:-2}
#phase4
AUG_PER_IMAGE=${AUG_PER_IMAGE:-4}
AUG_SEED=${AUG_SEED:-42}
VERB_ID=${VERB_ID:-118}  # Treat verb_id 117 as 118 (remapped below); selection uses this id.
NO_INTERACTION_VERB_ID=${NO_INTERACTION_VERB_ID:-58}
FILL_MISSING_NO_INTERACTION=${FILL_MISSING_NO_INTERACTION:-1}
APPLY_OBJECT_NMS=${APPLY_OBJECT_NMS:-1}
OBJECT_NMS_IOU=${OBJECT_NMS_IOU:-0.7}
OBJECT_NMS_PER_CATEGORY=${OBJECT_NMS_PER_CATEGORY:-1}
PROPAC_COUNT=${PROPAC_COUNT:-0}
MV_COUNT=${MV_COUNT:-0}
OUTPUT_DIR="${OUTPUT_DIR:-$ANNOT_DIR/selection}"
COPY_IMAGES=${COPY_IMAGES:-0}
COPY_FLAG=()
if [ "$COPY_IMAGES" = "1" ]; then
  COPY_FLAG+=(--copy_images)
fi

if [ ! -f "$ANNOT_JSON" ]; then
  echo "Annotations not found: $ANNOT_JSON" >&2
  exit 1
fi

if [ "$SYNC_PHASE4_IMAGES" = "1" ]; then
  if [ ! -d "$PHASE4_SAMPLED_ROOT" ]; then
    echo "Phase4 sampled frames root not found: $PHASE4_SAMPLED_ROOT" >&2
    exit 1
  fi
  echo "Syncing sampled frames into training images root:"
  echo "  source = $PHASE4_SAMPLED_ROOT"
  echo "  dest   = $IMAGES_ROOT"
  mkdir -p "$IMAGES_ROOT"
  rsync -a --info=progress2 \
    "$PHASE4_SAMPLED_ROOT/" \
    "$IMAGES_ROOT/"
fi

if [ ! -d "$IMAGES_ROOT" ]; then
  echo "Images root not found: $IMAGES_ROOT" >&2
  exit 1
fi

# Normalize verb ids (117 -> 118) before selection/augmentation cloning.
echo "Normalizing verb ids (117 -> 118):" ---
echo "  source json = $ANNOT_JSON"
echo "  sanitized   = $ANNOT_SANITIZED"
$PYTHON - "$ANNOT_JSON" "$ANNOT_SANITIZED" <<'PYCODE'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r") as f:
    entries = json.load(f)

if not isinstance(entries, list):
    raise SystemExit(f"Expected list in {src}")

changed = 0
for entry in entries:
    if not isinstance(entry, dict):
        continue
    for hoi in entry.get("hoi_annotation") or []:
        if not isinstance(hoi, dict):
            continue
        if hoi.get("category_id") == 117:
            hoi["category_id"] = 118
            changed += 1

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w") as f:
    json.dump(entries, f, indent=2)
print(f"Remapped {changed} verb_id 117 -> 118; wrote {len(entries)} entries to {dst}")
PYCODE

ANNOT_SRC="$ANNOT_SANITIZED"

# Remove duplicate/suppressed object boxes before pair-filling logic.
if [ "$APPLY_OBJECT_NMS" = "1" ]; then
  echo "Applying object-box NMS on annotations:"
  echo "  source json = $ANNOT_SRC"
  echo "  iou_thresh  = $OBJECT_NMS_IOU"
  NMS_ARGS=()
  if [ "$OBJECT_NMS_PER_CATEGORY" = "1" ]; then
    NMS_ARGS+=(--per-category)
  fi
  $PYTHON "$ROOT_DIR/5.704_apply_object_nms_hico.py" \
    --input-json "$ANNOT_SRC" \
    --output-json "$ANNOT_SRC" \
    --iou-thresh "$OBJECT_NMS_IOU" \
    "${NMS_ARGS[@]}"
fi

# Ensure every human-object pair has at least one HOI.
# If a pair has no HOI, add no_interaction (verb_id=58).
if [ "$FILL_MISSING_NO_INTERACTION" = "1" ]; then
  echo "Filling missing human-object HOIs with no_interaction:"
  echo "  source json = $ANNOT_SRC"
  echo "  verb_id     = $NO_INTERACTION_VERB_ID"
  $PYTHON "$ROOT_DIR/5.703_fill_missing_no_interaction_pairs.py" \
    --input-json "$ANNOT_SRC" \
    --output-json "$ANNOT_SRC" \
    --no-interaction-verb "$NO_INTERACTION_VERB_ID"
fi

echo "Augmenting images in place:"
echo "  root       = $IMAGES_ROOT"
echo "  per_image  = $AUG_PER_IMAGE"
$PYTHON "$ROOT_DIR/T1_augment_images.py" \
  --input_root "$IMAGES_ROOT" \
  --output_root "$IMAGES_ROOT" \
  --per_image "$AUG_PER_IMAGE" \
  --seed "$AUG_SEED" \
  --annotations_in "$ANNOT_SRC" \
  --annotations_out "$ANNOT_OUT"

# Use the augmented annotations as the source for downstream selection/balancing.
ANNOT_SRC="$ANNOT_OUT"

echo "Updating annotations with augmented entries:"
echo "  source json = $ANNOT_SRC"
echo "  output json = $ANNOT_OUT"
$PYTHON "$ROOT_DIR/T1_select_balance_interactions.py" \
  --annotations "$ANNOT_SRC" \
  --verb_id "$VERB_ID" \
  --frames_root "$IMAGES_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --propac_count "$PROPAC_COUNT" \
  --mv_count "$MV_COUNT" \
  --annotations_out "$ANNOT_OUT" \
  --augment_root "$IMAGES_ROOT" \
  --keep_masked_boxes \
  "${COPY_FLAG[@]}"

# Normalize paths inside the final annotations (replace images/train2015 -> annotations/images).
echo "Rewriting file_name paths in $ANNOT_OUT (images/train2015 -> annotations/images)"
$PYTHON - "$ANNOT_OUT" "$ANNOT_OUT" <<'PYCODE'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r") as f:
    entries = json.load(f)

if not isinstance(entries, list):
    raise SystemExit(f"Expected list in {src}")

changed = 0
for entry in entries:
    if not isinstance(entry, dict):
        continue
    fname = entry.get("file_name")
    if not fname:
        continue
    new_fname = fname.replace("images/train2015", "annotations/images")
    if new_fname != fname:
        entry["file_name"] = new_fname
        changed += 1

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w") as f:
    json.dump(entries, f, indent=2)
print(f"Rewrote {changed} file_name entries; saved to {dst}")
PYCODE

# Show HOI counts (original + updated with augment clones).
if [ -f "$ANNOT_SRC" ]; then
  echo "HOI counts (sanitized original annotations):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$ANNOT_SRC"
fi
if [ -f "$ANNOT_OUT" ]; then
  echo "HOI counts (augmented trainval):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$ANNOT_OUT"
fi

# Merge Phase-1 + newly augmented Phase-2 before model training.
if [ "$RUN_REMOTE_MERGE" = "1" ]; then
  if [ ! -f "$PHASE3_JSON" ]; then
    echo "Phase-2 annotations not found for merge: $PHASE3_JSON" >&2
    exit 1
  fi
  if [ ! -f "$PHASE4_AUG_JSON" ]; then
    echo "Phase-4 augmented annotations not found for merge: $PHASE4_AUG_JSON" >&2
    exit 1
  fi
  echo "Merging Phase-1 + Phase-4 augmented annotations:"
  echo "  phase1 = $PHASE3_JSON"
  echo "  phase2 = $PHASE4_AUG_JSON"
  echo "  output = $MERGED_TRAINVAL_JSON"
  $PYTHON "$ROOT_DIR/5.700_remote_merge.py" \
    --phase1-json "$PHASE3_JSON" \
    --phase2-json "$PHASE4_AUG_JSON" \
    --output-json "$MERGED_TRAINVAL_JSON"
fi

if [ "$RUN_FINAL_VIZ" = "1" ]; then
  VIZ_JSON="$ANNOT_OUT"
  if [ -f "$MERGED_TRAINVAL_JSON" ]; then
    VIZ_JSON="$MERGED_TRAINVAL_JSON"
  fi
  echo "Sampling final visualization set:"
  echo "  annotations = $VIZ_JSON"
  echo "  images_dir  = $IMAGES_ROOT"
  echo "  output_dir  = $FINAL_VIZ_OUTPUT_DIR"
  $PYTHON "$ROOT_DIR/5.702_one_time_sample_hico_viz.py" \
    --images_dir "$IMAGES_ROOT" \
    --annotations_json "$VIZ_JSON" \
    --output_dir "$FINAL_VIZ_OUTPUT_DIR" \
    --samples "$FINAL_VIZ_SAMPLES" \
    --positive-verb 118 \
    --min-positive-frames "$FINAL_VIZ_MIN_POSITIVE" \
    --seed "$FINAL_VIZ_SEED"
fi

echo "Done."

#============================= post processing data =============================#
# Everything below is a manual runbook (image symlinking, then launching CDN
# training) rather than a fully unattended pipeline -- it includes an alternate
# "if the first approach doesn't work" block (kept intentionally, not dead code)
# and was written to be run a step at a time. All paths are now overridable via
# env vars (same convention as the rest of this file); the defaults below match
# the original hyper13 layout, so set these before running elsewhere.

CDN_REPO_ROOT="${CDN_REPO_ROOT:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN}"
RUN_ROOT="${RUN_ROOT:-$CDN_REPO_ROOT/RUNS/Task_adv4_phase4.1}"
PHASE1_RUN_ROOT="${PHASE1_RUN_ROOT:-$CDN_REPO_ROOT/RUNS/Task_adv4_phase1}"
PHASE3_RUN_ROOT="${PHASE3_RUN_ROOT:-$CDN_REPO_ROOT/RUNS/Task_adv4_phase3}"
CDN_BASE_PRETRAINED="${CDN_BASE_PRETRAINED:-$HOME/CCAT_Opensource_work/p2_pretrained_Models/CDN/pretrained/hico_cdn_s.pth}"
CONDA_ENV="${CONDA_ENV:-pvic}"
STAGE1_GPU="${STAGE1_GPU:-1}"
STAGE2_GPU="${STAGE2_GPU:-0}"
REMOTE_HOST="${REMOTE_HOST:-mereddd@hyper13.isis.vanderbilt.edu}"
VIZ_DOWNLOAD_DEST="${VIZ_DOWNLOAD_DEST:-$HOME/Downloads/Adv_Active_Learning_Project/phase2_cdn_autolabel_v2}"

# # ##1) Add phase2 image symlinks into phase3 train2015 (flat by basename)
# cd "$RUN_ROOT/Annotations/"



SOURCE="$PHASE3_RUN_ROOT/Annotations/images/train2015"
TARGET="./images/train2015"

find "$SOURCE" -type f -name "*.jpg" -print0 | while IFS= read -r -d '' f; do
  base=$(basename "$f")
  ln -snf "$(realpath "$f")" "$TARGET/$base"
done

# ####---- repeat for each phase as needed
SOURCE="$PHASE1_RUN_ROOT/Annotations/images/train2015/"
TARGET="./images/train2015"
find "$SOURCE" -type f -name "*.jpg" -print0 | while IFS= read -r -d '' f; do
  base=$(basename "$f")
  ln -snf "$(realpath "$f")" "$TARGET/$base"
done


# delete later
# if above doesnt work it might mean you have + in annotations recheck please. try below if you
SOURCE="$PHASE3_RUN_ROOT/Annotations/images/train2015"
TARGET="./images/train2015"

find "$SOURCE" -type f -name "*.jpg" -print0 | while IFS= read -r -d '' f; do
    rel="${f#$SOURCE/}"
    mkdir -p "$TARGET/$(dirname "$rel")"
    ln -snf "$(realpath "$f")" "$TARGET/$rel"
done


# # 2) Copy corre_hico.npy
cp "$PHASE3_RUN_ROOT/Annotations/annotations/corre_hico.npy" ./annotations/

mv trainval_hico.json annotations/
cp annotations/trainval_hico.json annotations/test_hico.json

# # 3) Ensure test2015 link exists
cd "$RUN_ROOT/Annotations/images"
ln -s train2015 test2015


# ## dont run if not needed.
# #SOURCE="$PHASE3_RUN_ROOT/Annotations/images/train2015"
# #TARGET="train2015"
# # find "$SOURCE" -type f -name "*.jpg" -print0 | while IFS= read -r -d '' f; do
# #     base=$(basename "$f")
# #     ln -s "$(realpath "$f")" "$TARGET/$base"
# # done

# # Validata the data
scp -r "$REMOTE_HOST:$RUN_ROOT/Annotations/viz_sample_final_annotations" "$VIZ_DOWNLOAD_DEST/"

# # ##============================= Train the CDN model u =============================#
cd "$RUN_ROOT"
conda activate "$CONDA_ENV"

nohup bash -c "CUDA_VISIBLE_DEVICES=$STAGE1_GPU  python3.12 '$CDN_REPO_ROOT/main.py'\
        --pretrained '$CDN_BASE_PRETRAINED' \
        --dataset_file hico \
        --hoi_path './Annotations/' \
        --output_dir logs \
        --num_obj_classes 92 \
        --num_verb_classes 118 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 60 \
        --lr_drop 60 \
        --use_nms_filter \
        --batch_size 8 " > stage1_nohup.out 2>&1 &

# # tail -f stage1_nohup.out

# # # # # # # Finetune stage2 encoder frozen

conda activate "$CONDA_ENV"; nohup bash -c "CUDA_VISIBLE_DEVICES=$STAGE2_GPU \
        python3.12 '$CDN_REPO_ROOT/main.py' \
        --pretrained 'logs/checkpoint_best.pth' \
        --output_dir logs_stage2_v2/ \
        --dataset_file hico \
        --hoi_path './Annotations/'  \
        --num_obj_classes 92 \
        --num_verb_classes 118 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 10 \
        --freeze_mode 1 \
        --obj_reweight \
        --verb_reweight \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --use_nms_filter \
        " > stage2_nohup.out 2>&1 &


#------- Model test & val 

# go here 
# /Users/divyamereddy/Documents/Vanderbilt/OELELab/Ccat/CCAT_Primary_Action_Recognition_p2/pretrained_models/Tools_Data_Processing_Generic_forp2_DivyaCCATT/Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_v2.sh
