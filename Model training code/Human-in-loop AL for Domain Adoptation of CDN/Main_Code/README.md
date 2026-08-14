Semi-automatic HOI annotation pipeline (with active learning)
============================================================

Run order (shell scripts first)
-------------------------------
Use the orchestrator scripts; they string together all Python steps. Iterate Phase 2 until satisfied.

1) Phase 1 (bootstrap) — run first
   - Edit `Main_Code/camera_rois.json` with your ROI coordinates.
   - Defaults in the script point to the original `/home/mereddd/...` environment; if you’re on that machine, just run it. Override env vars only if your paths differ (examples: `CAMERA_VIEWS`, `OUTPUT_ROOT_BASE`, `YOLO_WEIGHTS`, `CDN_REPO`, `CDN_PRETRAINED`, `MMPOSE_ROOT/CONFIG/CHECKPOINT`).
   - From `Main_Code`, run `bash Phase1_SemiAutomtic_Data_Labeling.sh`
   - Produces: phase0 frames/detections, bootstrap CSVs (`bootstrap_candidates/auto/manual`), visualizations.
   - Low confidence output annotations are sent to human annotator. 
   - Post human annotation, From `Main_Code`, run `bash Phase1_Training_Dataprep & Training.sh` for augmenting data and training CDN model.

2) Phase 2 (active learning) — run after Phase 1, then repeat
   - From `Main_Code`, run `bash Phase2_SemiAutomtic_Data_Labeling.sh`
   - Produces: phase1 frames/detections, bootstrap CSVs (`bootstrap_candidates/auto/manual`), visualizations.
   - Low confidence output annotations are sent to human annotator. 
   - Post human annotation, From `Main_Code`, run `bash Phase2_Training_Dataprep & Training.sh` for augmenting data and training CDN model.
   - Each iteration: CDN inference → auto-label confident cases → send top-uncertainty to manual_review → update `active_state/` → train CDN → repeat with new checkpoint.

What’s inside (per-file map)
----------------------------
- Data prep: `1_Frames_Extraction.py`, `1_viz_copy_samples.py`, `2_sample_nonblurry.py`, `2_filter_and_sample_frames.py`, `2_viz_copy_samples.py`
- Detections + ROIs: `3_yolo_person_detect_and_attach_rois.py`, `3_viz_draw_detections.py`
- Bootstrap (Phase 1): `4.1_cdn_predictions.py`, `4.3_MMPOSE_interactions_detection.py`, `4.4_correct_annotations.py`, `4.5_merge_verb_annotations.py`, `4.6_Visulaize_annotations.py`, `4_bootstrap_agree_labels.py`, `4_viz_bootstrap_labels.py`
- Active learning (Phase 2+): `5_active_learning_iteration.py`, `5.3_visualize_phase2_annotations.py`, `5.51_runlocal_NMS.py`, `5.52.1_runlocal_visualize_bboxes_verify.py`, `5.541_Assignverb_based_on_distance.py`, `5.542_correct_annotations_runlocal.py`, `5.5_remove_hoi_duplicates.py`, `5.600merge_annotations.py`, `5.601export_hico_annotations.py`, `5.700_remote_merge.py`
- Training/augmentation utilities: `T1_augment_images.py`, `T1_count_hoi.py`, `T1_select_balance_interactions.py`, `T1_update_trainval_annotations.py`, `export_hico_annotations.py`
- Shell orchestrators: `Phase1_SemiAutomtic_Data_Labeling.sh`, `Phase1_Training_Dataprep & Training.sh`, `Phase2_SemiAutomtic_Data_Labeling.sh`, `Phase2_Training_Dataprep & Training.sh` (and copy variant)
- Config/assets: `camera_rois.json`, `roi_zoom_map_1-1.json`, `object_thresholds.json`, `object_crops.json`, weights placeholders `yolov8n.pt`, `yolov8x-seg.pt`
- Notebooks/analysis/debug: `z999_vaidate_data_phase*.ipynb`, `old_handskeletons_based_onLOgic.ipynb`, `fouronebackup.py`

Core rules (mirrors the request)
--------------------------------
- Phase 0: extract frames -> drop blurry images -> YOLO person detection only -> attach fixed ROIs (no learned object detector).
- Phase 1: run Pose wrist-in-ROI validation and pretrained CDN (HICO) in parallel; auto-accept only when they agree, otherwise send to manual review; use verified labels to fine-tune the first CDN.
- Phase 2+: stop using Pose model; CDN-only active learning loop with high-confidence auto-labels, <<20%> uncertainty budget for manual labels, then retrain and repeat.


