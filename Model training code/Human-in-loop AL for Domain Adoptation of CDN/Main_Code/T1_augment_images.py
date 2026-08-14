#!/usr/bin/env python3
"""
Lightweight image augmentation to expand Phase 1 training data.

Applies random combinations of:
- motion blur
- brightness/contrast jitter
- saturation jitter
- gaussian blur
- slight rotation
Geometric transforms update bounding boxes in the cloned annotations.

Usage:
  python T1_augment_images.py \
    --input_root /path/to/clear_frames \
    --output_root /path/to/augmented_frames \
    --per_image 3 \
    --seed 42
"""

from __future__ import annotations

import argparse
import random
import os
import json
import copy
from pathlib import Path

import cv2
import numpy as np

# Default annotation paths for Phase 3 (HICO-format).
DEFAULT_ANN_IN = Path("/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase3_Training/annotation_file/annotations/trainval_hico.json")
DEFAULT_ANN_OUT = Path("/home/mereddd/AIED_PAPER/CCATT/SActive_Learning_Based_SemiAutomatic_Data_Annotation/Semiautomaticdata/Phase3_Training/annotation_file/annotations/trainval_hico_with_aug.json")


def list_images(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    # Walk the tree and follow symlinked directories (datasets often expose frames via symlinks).
    image_paths: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if Path(name).suffix.lower() not in exts:
                continue
            stem = Path(name).stem
            # Skip already augmented or masked variants to avoid augmenting augmentations again.
            if "_aug" in stem or "_mask" in stem:
                continue
            image_paths.append(Path(dirpath) / name)
    return sorted(image_paths)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def apply_motion_blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(3, kernel_size | 1)  # ensure odd
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k  # horizontal
    else:
        kernel[:, k // 2] = 1.0 / k  # vertical
    return cv2.filter2D(img, -1, kernel)


def adjust_brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def adjust_saturation(img: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, ksize | 1)  # odd kernel
    return cv2.GaussianBlur(img, (k, k), 0)


def rotate_slight(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def augment_image(img: np.ndarray, *, allow_geometry: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    out = img.copy()
    h, w = img.shape[:2]
    transform = np.eye(3, dtype=np.float32)
    geom_changed = False
    # motion blur
    if random.random() < 0.4:
        out = apply_motion_blur(out, kernel_size=random.choice([3, 5, 7]))
    # brightness/contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)  # contrast
        beta = random.uniform(-12, 12)    # brightness
        out = adjust_brightness_contrast(out, alpha=alpha, beta=beta)
    # saturation
    if random.random() < 0.6:
        sat = random.uniform(0.8, 1.2)
        out = adjust_saturation(out, sat)
    # gaussian blur
    if random.random() < 0.3:
        out = gaussian_blur(out, ksize=random.choice([3, 5]))
    # slight rotation
    if allow_geometry and random.random() < 0.1:
        angle = random.uniform(-0.5, 0.5)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        transform = _compose_affine(transform, m)
        geom_changed = True
    return out, (transform if geom_changed else None)


def _compose_affine(current: np.ndarray, update_2x3: np.ndarray) -> np.ndarray:
    """Left-multiply a new 2x3 affine onto the running 3x3 transform."""
    update = np.eye(3, dtype=np.float32)
    update[:2, :] = update_2x3
    return update @ current


def _transform_bbox(bbox: list[float], transform: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox[:4])
    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]], dtype=np.float32)
    warped = (transform @ corners.T).T
    xs = warped[:, 0]
    ys = warped[:, 1]
    x_min = float(np.clip(xs.min(), 0.0, width - 1))
    x_max = float(np.clip(xs.max(), 0.0, width - 1))
    y_min = float(np.clip(ys.min(), 0.0, height - 1))
    y_max = float(np.clip(ys.max(), 0.0, height - 1))
    return [x_min, y_min, x_max, y_max]


def _apply_transform_to_annotations(entry: dict, transform: np.ndarray, width: int, height: int) -> None:
    anns = entry.get("annotations") or []
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        ann["bbox"] = _transform_bbox(list(bbox), transform, width, height)


def process_all(
    input_root: Path,
    output_root: Path,
    per_image: int,
    seed: int,
    annotations_in: Path | None = None,
    annotations_out: Path | None = None,
) -> None:
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    images = list_images(input_root)
    if not images:
        raise SystemExit(f"No images found under {input_root}")

    def _equiv_keys(key: str) -> list[str]:
        """Return equivalent filename keys across slash and plus-prefixed formats."""
        norm = key.strip().replace("\\", "/")
        keys = {norm}
        p = Path(norm)
        keys.add(p.name)
        keys.add(p.stem)
        if "/" in norm:
            parent = p.parent.as_posix()
            if parent and parent != ".":
                keys.add(f"{parent}+{p.name}")
                keys.add(f"{parent}+{p.stem}")
        if "+" in norm:
            left, right = norm.rsplit("+", 1)
            keys.add(f"{left}/{right}")
            keys.add(Path(right).name)
            keys.add(Path(right).stem)
        return [k for k in keys if k]

    def _entry_lookup(rel_key: str) -> dict | None:
        entry = ann_index.get(rel_key) or ann_index_lower.get(rel_key.lower())
        if entry:
            return entry
        stripped = rel_key
        for prefix in ("images/train2015/", "annotations/images/", "train2015/"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
        for cand in _equiv_keys(stripped) + _equiv_keys(rel_key):
            entry = ann_index.get(cand) or ann_index_lower.get(cand.lower())
            if entry:
                return entry
        return None

    def _is_no_interaction_only(entry: dict) -> bool:
        hois = entry.get("hoi_annotation") or []
        if not hois:
            return False
        no_interaction_ids = {57, 58}
        for hoi in hois:
            if not isinstance(hoi, dict):
                return False
            vid = hoi.get("category_id")
            try:
                vid_int = int(vid)
            except Exception:
                return False
            if vid_int not in no_interaction_ids:
                return False
        return True

    entries: list[dict] = []
    ann_index: dict[str, dict] = {}
    ann_index_lower: dict[str, dict] = {}
    if annotations_in:
        if not annotations_in.exists():
            raise SystemExit(f"Annotations not found: {annotations_in}")
        with annotations_in.open("r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise SystemExit(f"Expected a list of annotations in {annotations_in}")
        # Normalize file_name paths to be relative to images directory
        def _add_index_keys(key: str, entry: dict) -> None:
            key = key.strip().replace("\\", "/")
            for k in _equiv_keys(key):
                ann_index[k] = entry
                ann_index_lower[k.lower()] = entry
                # Alternate forms to catch path drift between annotations and disk.
                if k.startswith("images/train2015/"):
                    alt = k[len("images/train2015/") :]
                    ann_index[alt] = entry
                    ann_index_lower[alt.lower()] = entry
                if k.startswith("annotations/images/"):
                    alt = k[len("annotations/images/") :]
                    ann_index[alt] = entry
                    ann_index_lower[alt.lower()] = entry
                if k.startswith("train2015/"):
                    alt = k[len("train2015/") :]
                    ann_index[alt] = entry
                    ann_index_lower[alt.lower()] = entry

        for e in entries:
            fname = e.get("file_name")
            if not fname:
                continue
            fname = str(fname).strip().replace("\\", "/")
            for prefix in ("images/train2015/", "annotations/images/"):
                if fname.startswith(prefix):
                    fname = fname[len(prefix) :]
                    break
            e["file_name"] = fname
            _add_index_keys(fname, e)
            # Also index absolute paths if the file exists on disk (helps when JSON stores abs paths).
            abs_candidate = (input_root / fname).resolve()
            if abs_candidate.exists():
                _add_index_keys(str(abs_candidate), e)
            parent_candidate = (input_root.parent / fname).resolve()
            if parent_candidate.exists():
                _add_index_keys(str(parent_candidate), e)
        if annotations_out is None:
            annotations_out = annotations_in.parent / f"{annotations_in.stem}_with_aug.json"

    clones: list[dict] = []
    missing_ann = 0

    for img_path in images:
        rel = img_path.relative_to(input_root)
        rel_str = str(rel).replace("\\", "/")
        out_dir = (output_root / rel.parent)
        ensure_dir(out_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read {img_path}")
            continue
        stem = img_path.stem
        height, width = img.shape[:2]
        entry = _entry_lookup(rel_str) if ann_index else None

        # If frame contains only no-interaction verbs (57/58), apply a single non-geometric
        # random augmentation and overwrite the original image (do not create _aug clones).
        if entry and _is_no_interaction_only(entry):
            random.seed(seed + hash((rel, "replace")))
            np.random.seed(seed + (abs(hash((rel, "replace"))) % 100000))
            aug, _ = augment_image(img, allow_geometry=False)
            cv2.imwrite(str(img_path), aug)
            continue

        for i in range(per_image):
            random.seed(seed + hash((rel, i)))
            np.random.seed(seed + i)
            aug, transform = augment_image(img)
            out_path = out_dir / f"{stem}_aug{i+1}{img_path.suffix}"
            cv2.imwrite(str(out_path), aug)
            if ann_index:
                if not entry:
                    entry = _entry_lookup(rel_str)
                if entry:
                    clone = copy.deepcopy(entry)
                    if transform is not None:
                        _apply_transform_to_annotations(clone, transform, width, height)
                    new_fname = str(out_path.relative_to(output_root)).replace("\\", "/")
                    for prefix in ("images/train2015/", "annotations/images/", "images/", "train2015/"):
                        if new_fname.startswith(prefix):
                            new_fname = new_fname[len(prefix) :]
                            break
                    clone["file_name"] = new_fname
                    clones.append(clone)
                else:
                    missing_ann += 1
        # Optionally copy original? Skip to avoid duplication.
    print(f"Augmented {len(images)} images -> {output_root} (x{per_image} each)")
    if ann_index and annotations_out:
        print(f"[INFO] Cloned {len(clones)} annotations for augmented images; missing annotations for {missing_ann} files")
        annotations_out.parent.mkdir(parents=True, exist_ok=True)
        with annotations_out.open("w") as f:
            json.dump(entries + clones, f, indent=2)
        print(f"[INFO] Wrote augmented annotations to {annotations_out} (total entries: {len(entries) + len(clones)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply lightweight augmentations to images to expand training data.")
    ap.add_argument("--input_root", type=Path, required=True, help="Root directory of input images (recursively scanned).")
    ap.add_argument("--output_root", type=Path, required=True, help="Where to write augmented images (mirrors folder structure).")
    ap.add_argument("--per_image", type=int, default=2, help="Number of augmented variants per image (default: 2).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    ap.add_argument(
        "--annotations_in",
        type=Path,
        default=DEFAULT_ANN_IN,
        help="HICO-style annotations JSON; defaults to the Phase 3 merged annotations.",
    )
    ap.add_argument(
        "--annotations_out",
        type=Path,
        default=DEFAULT_ANN_OUT,
        help="Where to write augmented annotations JSON (default: Phase 3 merged_with_aug).",
    )
    args = ap.parse_args()

    process_all(
        args.input_root,
        args.output_root,
        per_image=args.per_image,
        seed=args.seed,
        annotations_in=args.annotations_in,
        annotations_out=args.annotations_out,
    )


if __name__ == "__main__":
    main()
