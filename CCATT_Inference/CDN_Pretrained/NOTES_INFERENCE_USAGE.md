# Notes on this copy (not part of the original CDN repo)

This is the CDN codebase actually used for **inference** -- the one
`Advanced_FineTuned_Model_Temporal_Data_Processing_Inference_pipeline_Actv_L.sh`
(and this repo's `run_video_to_temporal_hoi.sh`) runs against, at
`--hoi_path <frames>`.

Source: hyper13, `~/CCAT_Opensource_work/p2_pretrained_Models/CDN/` (the repo
root -- NOT the `CDN_Finetuning/` subfolder, which is a separate, different
codebase used only for training -- see `../CDN_Finetuned/`).

No LoRA/PEFT dependency. `engine.py`'s `save_preds()` here writes every
frame's predictions (no `[:20]` cap, unlike `CDN_Finetuned/engine.py`).

`main.py`, `engine.py`, `models/`, `datasets/`, `util/`, `requirements.txt`,
`LICENSE`, `README.md` (the original CDN paper README, unmodified), and
`convert_parameters.py` are copied verbatim from hyper13, MD5-verified
against the source at copy time.

## trainval_hico.json

The original README's own Evaluation example (see `README.md` above) always
runs `--hoi_path` against the *full prepared* HICO-DET dataset directory --
i.e. one that already has a real `annotations/trainval_hico.json` with real
training samples, because `main.py` builds `dataset_train` (and a
`RandomSampler` over it) unconditionally, even in `--eval` mode, and that
sampler hard-fails if there are zero samples.

For a single new video's freshly-extracted frames folder, there's no such
file generated automatically. **Resolved:** the real `trainval_hico.json`
used for CCATT inference runs is in the same Box folder as the model
weights (see `../README.md`) -- <https://vanderbilt.box.com/s/vc81uk1palnjjh72j3pvsc7awzj56u98>.
Copy it to `<hoi_path>/annotations/trainval_hico.json` alongside each video's
extracted frames before running Step 0.
