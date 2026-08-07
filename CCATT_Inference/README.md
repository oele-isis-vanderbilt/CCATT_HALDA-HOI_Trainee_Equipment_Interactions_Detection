# CCATT Inference Pipeline

Turns raw HOI (human-object interaction) model predictions into a simple table of
**trainee-equipment interactions**: which piece of equipment, who used it (Nurse /
Doctor / RT / Additional Team), and the start/end time of the interaction.

You do not need to know anything about the underlying deep-learning models to run
this. You just need paths to a few input/output folders.

## The pipeline

```
 Step 0 (external, GPU)     Step 1                    Step 2 (slow, GPU)              Step 3 (fast, CPU)
 -----------------------    ------                    ------------------               ------------------
 CDN HOI detection    -->   Frame-level HOI      -->   Role-assignment CSVs      -->    Final trainee-
 model produces               confidence scores          per video (who is the           equipment interaction
 *_df_preds.csv per           -> interaction              Nurse/Doctor/RT/etc,            table, with role
 camera view                  start/end times              per tracked person)             attached
```

| # | Script | What it does | Speed | Safe to re-run? |
|---|--------|---------------|-------|------------------|
| 0 | *(external, not in this repo)* | Runs the pretrained CDN HOI detection model over raw video to produce frame-level `*_df_preds.csv` prediction files. | **Slow, needs a GPU** | N/A |
| 1 | `V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py` | Turns frame-level model confidence scores into interaction intervals (start/end time) per piece of equipment. | Fast-moderate, CPU | Yes |
| 2 | `generate_role_assignment_csvs.py` | Runs a role-identification model on each video to figure out who (Nurse/Doctor/RT/Additional Team) is where, on every frame. | **Slow, needs a GPU** | Yes, but only re-processes new videos unless you pass `--force` |
| 3 | `person_identification_v4.py` | Joins step 1's interactions with step 2's role CSVs to attach a person/role to each interaction. | Fast, CPU only | Yes, always |

**Most of the time you will only run Steps 2 and 3.** Steps 0 and 1's output
(`combined_segments.csv` / `*_df_preds.csv` files) is typically produced once per
batch of videos by the ML team and handed to you as a folder. Step 2 only needs to
be re-run when new videos are added. Step 3 is cheap and safe to re-run whenever you
want, e.g. after new role-assignment CSVs show up.

Every script supports `--dry_run`, which checks that all your paths/files exist and
prints what *would* happen, without doing any real work. **Always try `--dry_run`
first** when running on a new machine or a new batch of videos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs everything needed for Steps 1 and 3 (pandas, numpy, matplotlib,
opencv-python, scipy). Step 2 is different -- see below.

### External model dependency for Step 0 (CDN)

Step 0 -- producing the `*_df_preds.csv` frame-level prediction files that Step 1
reads -- is not part of this repository. It uses the CDN (Cascade Disentangling
Network) HOI-detection codebase, kept as its own separate project. On this team's
shared machine it currently lives at:

```
/path/to/CDN
```

(for example, this may be the sibling `CDN/` folder next to this repository). You
only need this if you're regenerating `*_df_preds.csv` from scratch for new videos
-- if the ML team already handed you a folder of `combined_segments.csv` /
`*_df_preds.csv` files, you can skip straight to Step 2.

### External model dependency for Step 2

Step 2 (`generate_role_assignment_csvs.py`) does not contain a role-identification
model itself. It calls a separate, already-trained project ("role_contrastive")
that the CCATT ML team maintains, which needs its own Python environment (with
`torch` and `ultralytics`) and trained weights (`best.pt`). If you don't already
have a copy of that project on your machine, ask the CCATT ML team for it before
running Step 2. Once you have it, tell this script where to find it with
`--weights`, `--demo_role_script`, and `--python_bin` (see example below).

## Expected input folder structure

Step 1 and the pipeline as a whole are organized around a folder of per-simulation
subfolders, each containing a `combined_segments.csv`:

```
<input_root>/
  PAN_V1/
    combined_segments.csv          <- one row per candidate interaction
    2024C_Alpha2_..._df_preds.csv  <- frame-level model predictions (Step 1 input)
  CAM16_V1/
    combined_segments.csv
    ...
```

Videos referenced by `combined_segments.csv` (via a `video_stem`) are looked up in
whatever folder(s) you pass as `--video_roots` -- they do not need to live inside
`<input_root>` itself.

## Expected output files

- **Step 1** writes `combined_segments.csv` (and, if `--save_segments` is given, a
  segments CSV) with one row per predicted interaction: object, start time, end
  time, duration, etc.
- **Step 2** writes one CSV per video under `--role_csv_root`, named after the
  video (e.g. `2024C_Alpha2_high_PAN_V1.csv`), with columns
  `frame_id, time_seconds, track_id, x1, y1, x2, y2, conf, role`.
- **Step 3** writes, next to each `combined_segments.csv`:
  - `combined_segments_with_person_id_v4.csv` -- the full input table plus
    `Subject_Class_ID`, `Trainee_Identity` (Nurse/Doctor/RT/Additional Team), and
    match-diagnostic columns.
  - `CCATT_Trainee_Equipment_Interactions.csv` -- a slimmed-down version with just
    the columns most people care about (who, what equipment, when).

## Example commands

Replace every `/path/to/...` below with your own folders.

### Step 1 -- generate interaction intervals from model predictions

```bash
python3 V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py \
  --pred_csvs /path/to/hoi_results/PAN_V1/video1_df_preds.csv \
              /path/to/hoi_results/CAM16_V1/video2_df_preds.csv \
  --video_paths /path/to/videos/video1_PAN_V1.mp4 \
                /path/to/videos/video2_CAM16_V1.mp4 \
  --all_objs \
  --demo_outdir /path/to/hoi_results/OUTPUT_RUN \
  --no_video_demos
```

This uses built-in, pre-calibrated per-equipment thresholds -- you should not
normally need to change them. Add `--dry_run` to just check the paths first.
(This script also has an advanced `--grid_search` mode for re-tuning those
thresholds against expert ground truth; that is a research tool for the ML team,
not something domain reviewers need to touch -- see `--help` if curious.)

### Step 2 -- generate role-assignment CSVs (slow, GPU)

```bash
python3 generate_role_assignment_csvs.py \
  --input_root  /path/to/hoi_results/OUTPUT_RUN \
  --video_roots /path/to/videos \
  --role_csv_root /path/to/role_assignment_csvs \
  --weights /path/to/role_contrastive/weights/best.pt \
  --demo_role_script /path/to/role_contrastive/demo_role.py \
  --python_bin /path/to/role_yolo_env/bin/python3 \
  --conf 0.10 --device cuda:1
```

Add `--dry_run` first to confirm your paths and see which videos are missing or
already done. Videos that already have a role-assignment CSV are skipped
automatically; pass `--force` to regenerate everything.

### Step 3 -- attach roles to interactions (fast, CPU-only)

```bash
python3 person_identification_v4.py \
  --input_root /path/to/hoi_results/OUTPUT_RUN \
  --role_csv_root /path/to/role_assignment_csvs \
  --hoi_sampled_fps 4.0
```

Safe to re-run any time. By default it skips simulations it has already
processed; pass `--inplace` to force it to overwrite its own previous outputs.

## Troubleshooting

- **"not found" errors**: run the same command with `--dry_run` added -- every
  script checks all its input paths up front and tells you exactly which one is
  missing, instead of failing partway through.
- **Step 3 says a video's role CSV wasn't found**: that video hasn't had Step 2 run
  on it yet (or its role CSV's filename doesn't match its `video_stem`). Re-run
  Step 2 (or Step 2 with `--dry_run`) to check.
- **Step 2 fails immediately with a missing `--weights`/`--demo_role_script`/
  `--python_bin` error**: you need the separate `role_contrastive` project -- see
  "External model dependency" above.

## Repository layout

```
CCATT_Inference/
  V3_Create_Temporal_Predicted_HOI_intervals_version3_Gridsearch_Threshold.py  # Step 1
  generate_role_assignment_csvs.py                                            # Step 2
  person_identification_v4.py                                                 # Step 3
  requirements.txt
  .gitignore
  README.md
```
