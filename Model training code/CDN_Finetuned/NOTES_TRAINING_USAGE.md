# Notes on this copy (not part of the original CDN repo)

This is the CDN codebase used for **training/fine-tuning** the CCATT model
(the `Human-in-loop AL for Domain Adoptation of CDN/Main_Code` shell scripts'
`--adapter_dir`/LoRA training runs). It is **not** the codebase used for
inference -- see `../CDN_Pretrained/` for that.

Source: hyper13, `~/CCAT_Opensource_work/p2_pretrained_Models/CDN/CDN_Finetuning/CDN/`.

Includes LoRA/PEFT support (`from peft import ...` in `models/cdn.py`).
`engine.py`'s `save_preds()` here caps output to the first 20 frames unless
`--adapter_dir save_all` is passed (`save_preds(preds[:20])` otherwise) --
confirmed by reading the code; the original inference driver script does not
pass that flag.

`main.py`, `engine.py`, `models/`, `datasets/`, `util/`, `requirements.txt`,
`LICENSE`, `convert_parameters.py` are copied verbatim from hyper13,
MD5-verified against the source at copy time.
