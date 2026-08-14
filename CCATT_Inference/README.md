# CCATT Inference Pipeline

This repo takes two wide-angle camera recordings of a CCATT training
simulation as input and outputs which piece of equipment was used (IV,
ventilator, monitor), who used it, and the exact start/stop time of each
interaction. No deep-learning background is required to run it.

## What to provide

Upload the **synchronized** CAM (wide-angle) and PAN (wide-angle) camera
videos for **one simulation at a time** into the `Video_Input/` folder next
to this README -- both videos must be recordings of the same simulation,
started/stopped together. The pipeline processes both camera views of that
simulation together to predict all equipment interactions -- don't mix
videos from different simulations in the same run.

`Video_Input/` is git-ignored, so uploaded videos are never committed to the
repo.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

That covers everything except the "role identification" model, a separate,
already-trained model -- ask the ML team for it if you don't have it, then
point `PERSON_ID_WEIGHTS` / `DEMO_ROLE_SCRIPT` / `PYTHON_BIN` at your copy.

## Easiest way to run it (for Domain Experts)

One command runs all 4 steps and produces the final report -- no programming
experience needed:

```bash
PRETRAINED_MODEL_PATH=/path/to/checkpoint_best.pth \
PERSON_ID_WEIGHTS=/path/to/role_contrastive/weights/best.pt \
DEMO_ROLE_SCRIPT=/path/to/role_contrastive/demo_role.py \
PYTHON_BIN=/path/to/role_yolo_env/bin/python3 \
bash run_full_ccatt_pipeline.sh
```

Add `--dry_run` to the end first to check your paths (and that `Video_Input/`
has exactly the 2 videos it expects) before it actually runs.

If you have programming experience and want to see the results after each
step, or need to run many simulations in bulk, use the step-by-step version
below instead.

## What you get back

- The file you actually want: **`CCATT_Trainee_Equipment_Interactions.csv`**
  -- who, what equipment, when.
- A fuller version with extra detail: `combined_segments_with_person_id_v4.csv`.

## Detailed Step by Step Run (Optional if users want to see the results step by step)

If you're processing many videos at once (the ML team's usual workflow), the
same work happens in 4 separate steps, useful when running in bulk or when
someone else already handed you a folder of partial results:

| # | Script | What it does | Speed |
|---|--------|---------------|-------|
| 0 | `Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh` | Video -> frame-by-frame equipment/action guesses | GPU, slow |
| 1 | `V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py` | Those guesses -> interaction start/end times | CPU |
| 2 | `generate_role_assignment_csvs.py` | Figures out who (Nurse/Doctor/RT) is where, per video | GPU, slow |
| 3 | `person_identification_v4.py` | Combines Step 1 + Step 2 into the final "who did what, when" table | CPU, fast |

Run Steps 0-3 in order for a new video. Every script supports `--dry_run` --
always try that first on a new machine.

Replace every `/path/to/...` with your own folder. Add `--dry_run` to check
paths first without running anything.

**Step 0**
```bash
PRETRAINED_MODEL_PATH=/path/to/checkpoint_best.pth \
CDN_REPO_ROOT=/path/to/CDN_Pretrained \
INPUT_DIR=/path/to/videos \
OUTPUT_ROOT=/path/to/hoi_results/OUTPUT_RUN \
bash Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh
```

**Step 1**
```bash
python3 V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py \
  --pred_csvs /path/to/hoi_results/PAN_V1/video1_df_preds.csv /path/to/hoi_results/CAM16_V1/video2_df_preds.csv \
  --video_paths /path/to/videos/video1_PAN_V1.mp4 /path/to/videos/video2_CAM16_V1.mp4 \
  --all_objs --demo_outdir /path/to/hoi_results/OUTPUT_RUN --no_video_demos
```

**Step 2**
```bash
python3 generate_role_assignment_csvs.py \
  --input_root /path/to/hoi_results/OUTPUT_RUN \
  --video_roots /path/to/videos \
  --role_csv_root /path/to/role_assignment_csvs \
  --weights /path/to/role_contrastive/weights/best.pt \
  --demo_role_script /path/to/role_contrastive/demo_role.py \
  --python_bin /path/to/role_yolo_env/bin/python3 \
  --conf 0.10 --device cuda:1
```
Skips any video already done; pass `--force` to redo everything.

**Step 3**
```bash
python3 person_identification_v4.py \
  --input_root /path/to/hoi_results/OUTPUT_RUN \
  --role_csv_root /path/to/role_assignment_csvs \
  --hoi_sampled_fps 4.0
```
Skips videos already done; pass `--inplace` to overwrite them.

## If something goes wrong

- Any **"not found"** error -> re-run with `--dry_run` added; it tells you
  exactly which path is wrong instead of failing partway through.
- **Step 3 can't find a video's role info** -> that video hasn't had Step 2
  run on it yet.
- **Step 2 complains about `--weights` / `--demo_role_script` /
  `--python_bin`** -> ask the ML team for the "role identification" model
  project.
