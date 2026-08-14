# first we should copy the merged_hico_format file. 

#!/bin/bash
# Phase 1 training prep helper.
# Original notes: copy merged HICO annotations, select N Propac interactions, balance MV/IV, augment (blur/brightness/contrast/etc.).
# This script copies the merged annotations, links clear frames, and selects Propac interaction samples.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}
PHASE1_ROOT="${PHASE1_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase1}"
# Auto-detect common multi-site layouts (Semiautomaticdata/1-1/phase1, Semiautomaticdata/1-4/phase1).
AUTO_PHASE1_ROOTS=()
for site in 1-1 1-4; do
  candidate="$ROOT_DIR/Semiautomaticdata/$site/phase1"
  if [ -d "$candidate" ]; then
    AUTO_PHASE1_ROOTS+=("$candidate")
  fi
done
# Optional comma-separated list of phase1 roots (e.g., "path/to/1-1/phase1,path/to/1-4/phase1").
PHASE1_ROOTS_ENV="${PHASE1_ROOTS:-}"
if [ -n "$PHASE1_ROOTS_ENV" ]; then
  OLDIFS="$IFS"
  IFS=',' read -r -a PHASE1_ROOT_ARR <<< "$PHASE1_ROOTS_ENV"
  IFS="$OLDIFS"
elif [ "${#AUTO_PHASE1_ROOTS[@]}" -gt 0 ]; then
  PHASE1_ROOT_ARR=("${AUTO_PHASE1_ROOTS[@]}")
else
  PHASE1_ROOT_ARR=("$PHASE1_ROOT")
fi
ANNOT_ROOT="$ROOT_DIR/Semiautomaticdata/Phase1_Training/annotations"
MERGED_JSON="${MERGED_JSON:-$ANNOT_ROOT/merged_hico_annotations.json}"
TRAINVAL_JSON="$ANNOT_ROOT/trainval_hico.json"
TARGET_DIR="$ROOT_DIR/Semiautomaticdata/Phase1_Training/annotations/images"
TARGET_DIR_FLAT="$ROOT_DIR/Semiautomaticdata/Phase1_Training/annotations/images_flat"
TRAIN_IMG_ROOT="$ROOT_DIR/Semiautomaticdata/Phase1_Training/images/train2015"
PROPAC_OUT="$ROOT_DIR/Semiautomaticdata/Phase1_Training/propac_samples"
PROPAC_COUNT=${PROPAC_COUNT:-1000}
MV_COUNT=${MV_COUNT:-$PROPAC_COUNT}
BALANCE_IV=${BALANCE_IV:-1}
AUGMENT_ENABLE=${AUGMENT_ENABLE:-1}
AUG_PER_IMAGE=${AUG_PER_IMAGE:-3}
# Optional Phase 2+ (active learning) runner — disabled by default.
ACTIVE_ENABLE=${ACTIVE_ENABLE:-0}
ACTIVE_ITERATION=${ACTIVE_ITERATION:-0}
ACTIVE_DETECTIONS_ROOT="${ACTIVE_DETECTIONS_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase0/3_detections}"
ACTIVE_FRAMES_ROOT="${ACTIVE_FRAMES_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase0}"
ACTIVE_STATE_ROOT="${ACTIVE_STATE_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase1}"
ACTIVE_ROI_CONFIG="${ACTIVE_ROI_CONFIG:-$ROOT_DIR/camera_rois.json}"
ACTIVE_CAMERA_VIEW="${ACTIVE_CAMERA_VIEW:-1-4}"
ACTIVE_EQUIPMENT_TYPES=(${ACTIVE_EQUIPMENT_TYPES:-"IV Pump" MV})
ACTIVE_CDN_REPO="${ACTIVE_CDN_REPO:-}"
ACTIVE_CDN_WEIGHTS="${ACTIVE_CDN_WEIGHTS:-}"
ACTIVE_CDN_EXTRA_ARGS="${ACTIVE_CDN_EXTRA_ARGS:---device cuda:0}"
ACTIVE_POS_THRESH="${ACTIVE_POS_THRESH:-0.6}"
ACTIVE_NEG_THRESH="${ACTIVE_NEG_THRESH:-0.6}"
ACTIVE_SUBJ_IOU="${ACTIVE_SUBJ_IOU:-0.6}"
ACTIVE_OBJ_IOU="${ACTIVE_OBJ_IOU:-0.3}"
ACTIVE_BUDGET_RATIO="${ACTIVE_BUDGET_RATIO:-0.2}"
# Source root for clear frames; override via SRC_ROOT env if stored elsewhere.
COPY_FRAMES=${COPY_FRAMES:-1}
COPY_FRAMES_TO="${COPY_FRAMES_TO:-$ROOT_DIR/Semiautomaticdata/Phase1_Training/Training_frames}"
SRC_ROOT="${SRC_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase0/2_clear_frames}"
if [ "$COPY_FRAMES" = "1" ]; then
  SRC_ROOT="$COPY_FRAMES_TO"
fi
AUGMENT_INPUT="${AUGMENT_INPUT:-$SRC_ROOT}"
if [ "$COPY_FRAMES" = "1" ]; then
  AUGMENT_ROOT="${AUGMENT_ROOT:-$ROOT_DIR/Semiautomaticdata/Phase1_Training/Training_frames_aug}"
else
  AUGMENT_ROOT="${AUGMENT_ROOT:-$ROOT_DIR/Semiautomaticdata/video_smaples/train/phase0/2_clear_frames_aug}"
fi

# Export merged HICO annotations upfront (skips frames with disagreements).
valid_roots=()
for r in "${PHASE1_ROOT_ARR[@]}"; do
  if [ -d "$r" ]; then
    valid_roots+=("$r")
  else
    echo "Phase1 root not found at $r; skipping."
  fi
done

if [ "${#valid_roots[@]}" -gt 0 ]; then
  echo "Exporting merged HICO annotations from: ${valid_roots[*]}"
  mkdir -p "$(dirname "$MERGED_JSON")"
  export_args=(--phase1_root "${valid_roots[@]}" --output "$MERGED_JSON")
  if [ "$COPY_FRAMES" = "1" ]; then
    mkdir -p "$COPY_FRAMES_TO"
    export_args+=(--copy_frames_to "$COPY_FRAMES_TO")
  fi
  $PYTHON "$ROOT_DIR/export_hico_annotations.py" "${export_args[@]}"
else
  echo "No valid phase1 roots found; skipping export."
fi

# Copy merged HICO annotations into the Phase1 training annotations folder.
mkdir -p "$ANNOT_ROOT" "$TARGET_DIR"
if [ -f "$MERGED_JSON" ]; then
  cp "$MERGED_JSON" "$TRAINVAL_JSON"
  echo "Copied merged annotations to $TRAINVAL_JSON"
else
  echo "Merged annotations not found at $MERGED_JSON (skip copy)."
fi

if [ ! -d "$SRC_ROOT" ]; then
  echo "Source frames directory not found at: $SRC_ROOT" >&2
  exit 1
fi

shopt -s nullglob
linked=0
for src_dir in "$SRC_ROOT"/*/*/; do
  parent="$(basename "$(dirname "$src_dir")")"
  base="$(basename "$src_dir")"
  dest_dir="$TARGET_DIR/$parent"
  dest="$dest_dir/$base"
  mkdir -p "$dest_dir"
  if [ -L "$dest" ] || [ -e "$dest" ]; then
    echo "Skipping existing entry: $dest"
    continue
  fi
  ln -s "$src_dir" "$dest"
  echo "Linked $dest -> $src_dir"
  linked=$((linked + 1))
done
shopt -u nullglob

if [ "$linked" -eq 0 ]; then
  echo "No new symlinks were created (nothing to link or all already present)."
else
  echo "Created $linked symlink(s) under $TARGET_DIR"
fi

# Build a flat symlink view of all images for easy globbing.
mkdir -p "$TARGET_DIR_FLAT"
flat_created=0
while IFS= read -r file; do
  rel="${file#$TARGET_DIR/}"
  safe_name="${rel//\//__}"
  dest="$TARGET_DIR_FLAT/$safe_name"
  if [ -e "$dest" ]; then
    continue
  fi
  ln -s "$file" "$dest"
  flat_created=$((flat_created + 1))
done < <(find -L "$TARGET_DIR" -path "$TARGET_DIR_FLAT" -prune -o -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print)
echo "Flattened $flat_created image link(s) under $TARGET_DIR_FLAT"

# Expose a train2015-style images root for downstream training pipelines.
if [ ! -e "$TRAIN_IMG_ROOT" ]; then
  mkdir -p "$(dirname "$TRAIN_IMG_ROOT")"
  ln -s "$TARGET_DIR" "$TRAIN_IMG_ROOT"
  echo "Linked $TRAIN_IMG_ROOT -> $TARGET_DIR"
elif [ -L "$TRAIN_IMG_ROOT" ]; then
  echo "train2015 link already exists: $TRAIN_IMG_ROOT"
else
  echo "train2015 path exists and is not a symlink: $TRAIN_IMG_ROOT (leave as is)"
fi

# Select N Propac interactions (and N MV), optionally balance IV by masking extras; copy samples under propac_samples.
if [ -f "$MERGED_JSON" ]; then
  mkdir -p "$PROPAC_OUT"
  balance_flag=()
  # Run augmentation automatically if enabled.
  if [ "$AUGMENT_ENABLE" = "1" ]; then
    mkdir -p "$AUGMENT_ROOT"
    echo "Running augmentations from $AUGMENT_INPUT -> $AUGMENT_ROOT (per_image=$AUG_PER_IMAGE)"
    $PYTHON "$ROOT_DIR/T1_augment_images.py" \
      --input_root "$AUGMENT_INPUT" \
      --output_root "$AUGMENT_ROOT" \
      --per_image "$AUG_PER_IMAGE"
    # Expose augmented frames under Training_frames for downstream loaders expecting that root.
    if [ -d "$COPY_FRAMES_TO" ]; then
      shopt -s nullglob
      for site_dir in "$AUGMENT_ROOT"/*/; do
        base_site="$(basename "$site_dir")"
        dest_site="$COPY_FRAMES_TO/$base_site"
        if [ ! -e "$dest_site" ]; then
          ln -s "$site_dir" "$dest_site"
          echo "Linked augmented site $dest_site -> $site_dir"
        fi
      done
      shopt -u nullglob
    fi
  else
    AUGMENT_ROOT=""
  fi
  anno_args=(--annotations_out "$TRAINVAL_JSON")
  if [ -n "$AUGMENT_ROOT" ]; then
    anno_args+=(--augment_root "$AUGMENT_ROOT")
  fi
  $PYTHON "$ROOT_DIR/T1_select_balance_interactions.py" \
    --annotations "$MERGED_JSON" \
    --frames_root "$TARGET_DIR" \
    --output_dir "$PROPAC_OUT" \
    --propac_count "$PROPAC_COUNT" \
    --mv_count "$MV_COUNT" \
    "${anno_args[@]}" \
    --copy_images
else
  echo "Merged annotations not found at $MERGED_JSON; skipping Propac selection."
fi

# Show HOI counts for interaction/no_interaction per object (original + final).
if [ -f "$MERGED_JSON" ]; then
  echo "HOI counts (original merged annotations):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$MERGED_JSON"
fi
if [ -f "$TRAINVAL_JSON" ]; then
  echo "HOI counts (trainval with aug/clones if provided):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$TRAINVAL_JSON"
fi

# we can find all images in Training_frames folder now
rsync -av /home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Training_frames_aug/ /home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Training_frames/


#============================= Train the CDN model using the prepared data =============================#



# nohup bash -c "CUDA_VISIBLE_DEVICES=0  \
# python3.12 /home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/main.py       
#  --pretrained '/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/pretrained/hico_cdn_s.pth'         
#  --dataset_file hico         
#  --hoi_path '../'         
#  --output_dir logs         
#  --num_obj_classes 83         
#  --num_verb_classes 118         
#  --backbone resnet50         
#  --num_queries 64         
#  --dec_layers_hopd 3         
#  --dec_layers_interaction 3         
#  --epochs 90         
#  --lr_drop 60        
#   --use_nms_filter
#   --batch_size 64 " > stage1_nohup.out 2>&1 &

# # # Finetune stage2 encoder frozen 

# conda activate pvic; nohup bash -c "CUDA_VISIBLE_DEVICES=0  \
#         python3.12 /home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/main.py \
#         --pretrained 'logs/checkpoint_best.pth' \
#         --output_dir logs_stage2_v2/ \
#         --dataset_file hico \
#         --hoi_path '../' \
#         --num_obj_classes 83 \
#         --num_verb_classes 118 \
#         --backbone resnet50 \
#         --num_queries 64 \
#         --dec_layers_hopd 3 \
#         --dec_layers_interaction 3 \
#         --epochs 10 \
#         --freeze_mode 1 \
#         --obj_reweight \
#         --verb_reweight \
#         --lr 1e-5 \
#         --lr_backbone 1e-6 \
#         --use_nms_filter \
#         " > stage2_nohup.out 2>&1 &

