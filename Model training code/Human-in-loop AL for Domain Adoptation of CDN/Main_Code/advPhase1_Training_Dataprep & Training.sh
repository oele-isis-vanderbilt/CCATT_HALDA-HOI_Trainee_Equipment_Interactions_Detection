# Running commands
# bash 'advPhase1_Training_Dataprep & Training.sh'
#!/bin/bash
#================================================================================
# Extract 4 frames per second per images first 
# Then uniformly selected 200 images per view per version device.saved them in below location
#================================================================================
ANNOTATIONS_ROOT='/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/RUNS/Task_adv4_phase1/Annotations' # Phase2
IMAGES_ROOT="${ANNOTATIONS_ROOT}/images/train2015"

# after this merge phase1 and phase2 annotations 
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

# Fixed Phase 2 annotations path.
ANNOT_JSON="${ANNOTATIONS_ROOT}/annotations/trainval_hico.json" # phase2
ANNOT_DIR="$(dirname "$ANNOT_JSON")"
ANNOT_OUT="${ANNOT_OUT:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_with_aug.json}"             # pre-balance (augmented)
ANNOT_BALANCED="${ANNOT_BALANCED:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_with_aug_balanced.json}" # post-balance
ANNOT_SANITIZED="${ANNOT_SANITIZED:-$ANNOT_DIR/$(basename "${ANNOT_JSON%.json}")_verb118.json}"

# Augmentation settings (same script as Phase 1).
AUG_PER_IMAGE=${AUG_PER_IMAGE:-2}
AUG_SEED=${AUG_SEED:-42}
VERB_ID=${VERB_ID:-118}  # Treat verb_id 117 as 118 (remapped below); selection uses this id.
PROPAC_COUNT=${PROPAC_COUNT:-1000}  # Set high to force balancing with propac clones
MV_COUNT=${MV_COUNT:-1000}  # Set high to force balancing with masked versions
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
echo "  balanced   = $ANNOT_BALANCED"
$PYTHON "$ROOT_DIR/T1_select_balance_interactions.py" \
  --annotations "$ANNOT_SRC" \
  --verb_id "$VERB_ID" \
  --frames_root "$IMAGES_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --propac_count "$PROPAC_COUNT" \
  --mv_count "$MV_COUNT" \
  --annotations_out "$ANNOT_BALANCED" \
  --augment_root "$IMAGES_ROOT" \
  "${COPY_FLAG[@]}"

# Normalize paths inside the final annotations (strip any images/train2015 prefixes).
echo "Rewriting file_name paths in $ANNOT_BALANCED (remove images/train2015/ prefixes)"
$PYTHON - "$ANNOT_BALANCED" "$ANNOT_BALANCED" <<'PYCODE'
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
    new_fname = str(fname).replace("\\", "/")
    for pre in ("images/train2015/", "annotations/images/", "images/", "train2015/"):
        if new_fname.startswith(pre):
            new_fname = new_fname[len(pre) :]
            break
    if new_fname != fname:
        entry["file_name"] = new_fname
        changed += 1

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w") as f:
    json.dump(entries, f, indent=2)
print(f"Rewrote {changed} file_name entries; saved to {dst}")
PYCODE

# Show HOI counts (pre- and post-balance).
if [ -f "$ANNOT_OUT" ]; then
  echo "HOI counts (pre-balance augmented):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$ANNOT_OUT"
fi
if [ -f "$ANNOT_BALANCED" ]; then
  echo "HOI counts (balanced trainval):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$ANNOT_BALANCED"
fi

echo "Removing HOI interactions whose object category is 1 (object_id=1) from $ANNOT_BALANCED"
$PYTHON - "$ANNOT_BALANCED" "$ANNOT_BALANCED" <<'PYCODE'
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with src.open("r") as f:
    entries = json.load(f)

if not isinstance(entries, list):
    raise SystemExit(f"Expected list in {src}")

removed = 0
for entry in entries:
    anns = entry.get("annotations") or []
    hois = entry.get("hoi_annotation") or []
    kept = []
    for hoi in hois:
        obj_id = hoi.get("object_id")
        if obj_id is None or obj_id >= len(anns):
            continue
        obj_cat = anns[obj_id].get("category_id")
        if obj_cat == 1:
            removed += 1
            continue
        kept.append(hoi)
    entry["hoi_annotation"] = kept

with dst.open("w") as f:
    json.dump(entries, f, indent=2)
print(f"Removed {removed} HOI interactions tied to object category 1; saved to {dst}")
PYCODE

# Sanity: recount HOIs after removal.
if [ -f "$ANNOT_BALANCED" ]; then
  echo "HOI counts (post object-1 removal):"
  $PYTHON "$ROOT_DIR/T1_count_hoi.py" --annotations "$ANNOT_BALANCED"
fi

echo "Done."

#============================= Train the CDN model using the prepared data =============================#

# nohup bash -c "CUDA_VISIBLE_DEVICES=1  python3.12 ../../main.py\
#         --pretrained '/home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/pretrained/hico_cdn_s.pth' \
#         --dataset_file hico \
#         --hoi_path ./Annotations \
#         --output_dir logs \
#         --num_obj_classes 92 \
#         --num_verb_classes 118 \
#         --backbone resnet50 \
#         --num_queries 64 \
#         --dec_layers_hopd 3 \
#         --dec_layers_interaction 3 \
#         --epochs 60 \
#         --lr_drop 60 \
#         --use_nms_filter \
#         --batch_size 8" > stage1_nohup.out 2>&1 &

# # # # # Finetune stage2 encoder frozen 

# conda activate pvic; nohup bash -c "CUDA_VISIBLE_DEVICES=1 \
#         python3.12 /home/mereddd/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/main.py \
#         --pretrained 'logs/checkpoint_best.pth' \
#         --output_dir logs_stage2_v2/ \
#         --dataset_file hico \
#         --hoi_path  ./Annotations \
#         --num_obj_classes 92 \
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
