#!/bin/bash
#IMAGES_ROOT='/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase2_Training/annotations/images' # Phase2
#
#IMAGES_ROOT='/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase3_Training/annotation_file/annotations/images/train2015/' # Phase3
IMAGES_ROOT='/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase3_Training/annotation_file/annotations/images/train2015'

#bash Phase2_Training_Dataprep%20%26%20Training.sh


# after this merge phase1 and phase2 annotations 
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

# Fixed Phase 2 annotations path.
ANNOT_JSON="/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase2_Training/annotations/merged_hico_annotations_merged_ph2.json" # phase2
ANNOT_JSON="/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase3_Training/annotation_file/annotations/merged_hico_annotations_merged.json" #Phase3

ANNOT_DIR="$(dirname "$ANNOT_JSON")"
ANNOT_OUT="${ANNOT_OUT:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_with_aug.json}"
ANNOT_SANITIZED="${ANNOT_SANITIZED:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_verb118.json}"

# Images root (override with IMAGES_ROOT if it differs).
IMAGES_ROOT="${IMAGES_ROOT:-$ANNOT_DIR/images}"

# Augmentation settings (same script as Phase 1).
AUG_PER_IMAGE=${AUG_PER_IMAGE:-2}
AUG_SEED=${AUG_SEED:-42}
VERB_ID=${VERB_ID:-118}  # Treat verb_id 117 as 118 (remapped below); selection uses this id.
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

echo "Done."

#============================= Train the CDN model using the prepared data =============================#

conda activate pvic

nohup bash -c "CUDA_VISIBLE_DEVICES=1  python3.12 ../../main.py\
        --pretrained '/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/pretrained/hico_cdn_s.pth' \
        --dataset_file hico \
        --hoi_path '../Task_adv3/Annotations/' \
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

# # # Finetune stage2 encoder frozen 

conda activate pvic; nohup bash -c "CUDA_VISIBLE_DEVICES=1 \
        python3.12 /home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/main.py \
        --pretrained 'logs/checkpoint_best.pth' \
        --output_dir logs_stage2_v2/ \
        --dataset_file hico \
        --hoi_path ../annotation_file/  \
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

