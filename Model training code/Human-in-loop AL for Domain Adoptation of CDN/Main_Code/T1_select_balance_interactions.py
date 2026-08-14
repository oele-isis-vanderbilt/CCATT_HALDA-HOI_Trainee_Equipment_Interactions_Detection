#!/usr/bin/env python3
"""
Balance valid HOI samples across multiple object categories (e.g., 91–102) by masking the majority class in duplicated images.

- Works with an arbitrary list of object ids (`--object_ids ...`); legacy 3-class args are kept for compatibility.
- Keeps all interactions unless per-class caps are provided; defines the majority class by kept counts.
- Writes a combined CSV with per-object subsets and optionally copies images into output_dir/images/.
- Optionally writes an updated annotations JSON (trainval) that removes masked interactions for every trimmed class and
  appends masked/augmented-image annotations when an --augment_root is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def load_annotations(path: Path) -> list[dict]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of annotations in {path}, got {type(data)}")
    return data


def sample_entries(matches: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count <= 0 or count >= len(matches):
        return list(matches)
    pool = list(matches)
    rng.shuffle(pool)
    return pool[:count]


def find_entries(
    entries: list[dict],
    object_category: int,
    verb_id: Optional[int],
) -> list[dict]:
    """Return one row per unique frame containing the given object category (and verb, if provided)."""
    matches = []
    for entry in entries:
        anns = entry.get("annotations", [])
        hois = entry.get("hoi_annotation", [])
        for hoi in hois:
            obj_id = hoi.get("object_id")
            verb = hoi.get("category_id")
            if obj_id is None or obj_id >= len(anns):
                continue
            obj_ann = anns[obj_id]
            if obj_ann.get("category_id") != object_category:
                continue
            if verb_id is not None and verb != verb_id:
                continue
            matches.append(
                {
                    "file_name": entry.get("file_name"),
                    "verb_id": verb,
                    "object_category": obj_ann.get("category_id"),
                }
            )
    # De-duplicate by file_name to avoid repeated frames.
    seen = set()
    unique = []
    for m in matches:
        fname = m.get("file_name")
        if fname in seen:
            continue
        seen.add(fname)
        unique.append(m)
    return unique


def build_aug_index(augment_root: Path) -> Dict[str, List[str]]:
    """
    Map original relative file names to augmented file names (expects *_aug*.ext).
    Uses relative paths so nested directories match annotations that include subfolders.
    """
    def key_variants(key: str) -> List[str]:
        k = key.replace("\\", "/")
        out = {k, Path(k).name}
        if "/" in k:
            p = Path(k)
            parent = p.parent.as_posix()
            if parent and parent != ".":
                out.add(f"{parent}+{p.name}")
        if "+" in k:
            left, right = k.rsplit("+", 1)
            out.add(f"{left}/{right}")
            out.add(Path(right).name)
        return [x for x in out if x]

    index: Dict[str, List[str]] = {}
    exts = {".jpg", ".jpeg", ".png"}
    for dirpath, _, filenames in os.walk(augment_root, followlinks=True):
        for name in filenames:
            if Path(name).suffix.lower() not in exts:
                continue
            stem = Path(name).stem
            if "_aug" not in stem:
                continue
            rel = (Path(dirpath) / name).relative_to(augment_root)
            rel_s = str(rel).replace("\\", "/")
            base_name = stem.split("_aug")[0] + Path(name).suffix
            # Map using full relative context and aliases so annotations keyed as
            # subdir/frame.jpg or subdir+frame.jpg both resolve.
            full_base = str(rel.parent / base_name).replace("\\", "/") if str(rel.parent) != "." else base_name
            for k in key_variants(full_base):
                index.setdefault(k, []).append(rel_s)
            for k in key_variants(base_name):
                index.setdefault(k, []).append(rel_s)
    return index


def mask_objects(
    src_path: Path,
    dest_path: Path,
    annotations: List[dict],
    mask_cats: set[int],
) -> bool:
    """
    Mask specified object categories (by bbox) in the image and save to dest_path.
    If no masks are applied, the original image is copied so callers can still use the clone.
    Returns True when a file is written.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    masked = False
    h, w = img.shape[:2]
    for ann in annotations:
        try:
            ann_cat = int(ann.get("category_id"))
        except Exception:
            continue
        if ann_cat not in mask_cats:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        try:
            x1, y1, x3, y3 = map(float, bbox)
        except Exception:
            continue
        # Support both xyxy and xywh-style boxes.
        if x3 <= x1 or y3 <= y1:
            x2 = x1 + max(0.0, x3)
            y2 = y1 + max(0.0, y3)
        else:
            x2, y2 = x3, y3
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        masked = True
    if not masked:
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), img)
    return True

def _prune_entry_annotations_and_hois(
    entry: dict,
    drop_object_categories: set[int],
    keep_only_object_category: Optional[int] = None,
    keep_masked_annotations: bool = False,
) -> tuple[dict, int]:
    """
    Remove dropped object categories from `annotations` and keep only valid HOIs.
    Reindex subject_id/object_id after annotation compaction.
    Returns (new_entry, removed_hoi_count).
    """
    anns = entry.get("annotations", []) or []
    hois = entry.get("hoi_annotation", []) or []

    if keep_masked_annotations:
        keep_ann_old_indices: set[int] = set(range(len(anns)))
    else:
        keep_ann_old_indices = set()
        for i, ann in enumerate(anns):
            cat = ann.get("category_id")
            if cat in drop_object_categories:
                continue
            keep_ann_old_indices.add(i)

    kept_hois = []
    removed_hois = 0
    for hoi in hois:
        s = hoi.get("subject_id")
        o = hoi.get("object_id")
        if s is None or o is None:
            removed_hois += 1
            continue
        if s not in keep_ann_old_indices or o not in keep_ann_old_indices:
            removed_hois += 1
            continue
        try:
            obj_cat = anns[o].get("category_id")
        except Exception:
            removed_hois += 1
            continue
        if keep_only_object_category is not None and obj_cat != keep_only_object_category:
            removed_hois += 1
            continue
        kept_hois.append(dict(hoi))

    # Keep all non-dropped annotations (not only HOI-referenced ones), so
    # visualization still shows the full scene context after masking.
    new_annotations = []
    old_to_new: Dict[int, int] = {}
    for old_idx, ann in enumerate(anns):
        if old_idx not in keep_ann_old_indices:
            continue
        old_to_new[old_idx] = len(new_annotations)
        new_annotations.append(dict(ann))

    new_hois = []
    for hoi in kept_hois:
        old_s = int(hoi["subject_id"])
        old_o = int(hoi["object_id"])
        if old_s not in old_to_new or old_o not in old_to_new:
            removed_hois += 1
            continue
        new_hoi = dict(hoi)
        new_hoi["subject_id"] = old_to_new[old_s]
        new_hoi["object_id"] = old_to_new[old_o]
        new_hois.append(new_hoi)

    new_entry = dict(entry)
    new_entry["annotations"] = new_annotations
    new_entry["hoi_annotation"] = new_hois
    return new_entry, removed_hois


def index_frames(frames_root: Path) -> Dict[str, List[Path]]:
    def key_variants(s: str) -> List[str]:
        s = s.replace("\\", "/")
        out = {s, Path(s).name}
        if "/" in s:
            p = Path(s)
            parent = p.parent.as_posix()
            if parent and parent != ".":
                out.add(f"{parent}+{p.name}")
        if "+" in s:
            left, right = s.rsplit("+", 1)
            out.add(f"{left}/{right}")
            out.add(Path(right).name)
        return [k for k in out if k]

    idx: Dict[str, List[Path]] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for path in frames_root.rglob(ext):
            for k in key_variants(path.name):
                idx.setdefault(k, []).append(path)
            try:
                rel = path.relative_to(frames_root)
                rel_s = str(rel).replace("\\", "/")
                for k in key_variants(rel_s):
                    idx.setdefault(k, []).append(path)
            except Exception:
                pass
    return idx


def copy_matches(
    matches: list[dict],
    frame_index: Dict[str, List[Path]],
    frames_root: Path,
    output_dir: Path,
    copied_set: set[Path],
    subset: str,
) -> list[dict]:
    images_dir = output_dir / "images"
    copied_rows = []
    for m in matches:
        fname = m.get("file_name")
        base_name = Path(fname).name if fname else ""
        candidates = frame_index.get(base_name, [])
        copied_path = ""
        source_path = ""
        if candidates:
            src = candidates[0]
            source_path = str(src)
            try:
                rel = src.relative_to(frames_root)
            except Exception:
                rel = Path(src.name)
            dest = images_dir / rel
            if dest not in copied_set:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                copied_set.add(dest)
            copied_path = str(dest)
        copied_rows.append({**m, "subset": subset, "source_path": source_path, "copied_path": copied_path})
    return copied_rows


def tag_subset(rows: list[dict], subset: str) -> list[dict]:
    return [{**r, "subset": subset, "source_path": r.get("source_path", ""), "copied_path": r.get("copied_path", "")} for r in rows]


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["file_name", "verb_id", "object_category", "subset", "source_path", "copied_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select balanced HOI interactions across one or more object categories.")
    ap.add_argument("--annotations", type=Path, required=True, help="Merged HICO-style annotations JSON.")
    ap.add_argument("--frames_root", type=Path, required=True, help="Root containing clear frames (searched recursively).")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory for CSVs and copied images.")
    ap.add_argument("--propac_count", "--count", dest="propac_count", type=int, default=0, help="Target Propac interaction frames (0 => keep all).")
    ap.add_argument("--mv_count", type=int, default=None, help="Target MV interaction frames (default: keep all).")
    ap.add_argument("--object_propac", type=int, default=92, help="Object category id for Propac (default 92).")
    ap.add_argument("--object_mv", type=int, default=93, help="Object category id for MV (default 93).")
    ap.add_argument("--object_iv", type=int, default=91, help="Object category id for IV (default 91).")
    ap.add_argument(
        "--object_ids",
        type=int,
        nargs="+",
        help="Object category ids to balance (e.g., 91 92 93 94 ... 102). Overrides object_propac/object_mv/object_iv when provided.",
    )
    ap.add_argument(
        "--object_caps",
        type=int,
        nargs="+",
        help="Optional per-object caps aligned with --object_ids (<=0 keeps all). If omitted, all kept unless legacy caps are used.",
    )
    ap.add_argument("--verb_id", type=int, default=118, help="Verb id to treat as positive interaction (set -1 to accept any verb).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    ap.add_argument("--copy_images", action="store_true", help="Copy matched images into output_dir/images/.")
    ap.add_argument("--annotations_out", type=Path, help="If set, write updated annotations JSON with IV interactions removed (and augmented clones if provided).")
    ap.add_argument("--augment_root", type=Path, help="Root of augmented images; when set, clone annotations for any *_aug* files found here.")
    ap.add_argument(
        "--keep_masked_boxes",
        action="store_true",
        help="Keep masked object boxes in annotations (pixel masking still applied); default removes masked object annotations.",
    )
    args = ap.parse_args()

    if not args.annotations.exists():
        raise SystemExit(f"Annotations file not found: {args.annotations}")
    if not args.frames_root.exists():
        raise SystemExit(f"Frames root not found: {args.frames_root}")

    frame_index: Dict[str, List[Path]] = index_frames(args.frames_root)
    total_indexed = sum(len(v) for v in frame_index.values())
    sample_keys = list(frame_index.keys())[:5]
    print(f"[debug] indexed {total_indexed} image entries from {args.frames_root}; sample keys={sample_keys}")

    def resolve_image_path(fname: str) -> Optional[Path]:
        """Try to locate an image for fname under frames_root (and augment_root), handling duplicated subpaths."""
        if not fname:
            return None
        fname_s = str(fname).replace("\\", "/")
        variants_raw = {fname_s}
        if "+" in fname_s:
            left, right = fname_s.rsplit("+", 1)
            variants_raw.add(f"{left}/{right}")
            variants_raw.add(Path(right).name)
        rel_candidates = [Path(v) for v in variants_raw]

        def variants(rel_path: Path, root: Path) -> list[Path]:
            vars: list[Path] = []
            # As-is
            vars.append(root / rel_path)
            # Strip leading annotations/
            if str(rel_path).startswith("annotations/"):
                try:
                    trimmed = rel_path.relative_to("annotations")
                    vars.append(root / trimmed)
                except Exception:
                    pass
            # Strip leading images/train2015 if frames_root already endswith it
            if str(rel_path).startswith("images/train2015"):
                try:
                    trimmed = rel_path.relative_to("images/train2015")
                    vars.append(root / trimmed)
                except Exception:
                    pass
            # One level up (handles double images/train2015/images/train2015)
            vars.append(root.parent / rel_path)
            # Basename
            vars.append(root / rel_path.name)
            return vars

        candidates = []
        for rel in rel_candidates:
            candidates.extend(variants(rel, args.frames_root))
        if args.augment_root:
            for rel in rel_candidates:
                candidates.extend(variants(rel, args.augment_root))

        for cand in candidates:
            if cand.exists():
                return cand

        # Fallback: lookup by basename/relative string in the indexed frames.
        for rel in rel_candidates:
            for key in (str(rel).replace("\\", "/"), rel.name):
                paths = frame_index.get(key)
                if paths:
                    return paths[0]
        # Final fallback: raw provided name and slash/plus alternate.
        for key in variants_raw:
            paths = frame_index.get(key)
            if paths:
                return paths[0]
        return None

    entries = load_annotations(args.annotations)
    verb_filter = None if args.verb_id == -1 else args.verb_id

    rng = random.Random(args.seed)

    # Determine which object ids to balance.
    if args.object_ids:
        object_ids = list(dict.fromkeys(args.object_ids))  # dedupe while preserving order
    else:
        # Default to the full IV/Propac/MV range the project uses (91–102).
        object_ids = list(range(91, 103))

    # Build per-object caps (<=0 means keep all).
    per_object_caps: Dict[int, int] = {oid: 0 for oid in object_ids}
    if args.object_ids:
        if args.object_caps:
            if len(args.object_caps) != len(object_ids):
                raise SystemExit("--object_caps length must match --object_ids")
            per_object_caps = {oid: cap for oid, cap in zip(object_ids, args.object_caps)}
    else:
        # Legacy caps still honored for Propac/MV when using default object_ids.
        mv_cap = args.mv_count if args.mv_count is not None else 0
        per_object_caps[args.object_propac] = args.propac_count
        per_object_caps[args.object_mv] = mv_cap

    # Collect matches and apply caps.
    matches_by_obj: Dict[int, list[dict]] = {}
    selected_by_obj: Dict[int, list[dict]] = {}
    masked_by_obj: Dict[int, list[dict]] = {}
    for oid in object_ids:
        matches = find_entries(entries, object_category=oid, verb_id=verb_filter)
        matches_by_obj[oid] = matches
        cap = per_object_caps.get(oid, 0)
        selected = list(matches) if cap <= 0 else sample_entries(matches, cap, rng)
        selected_by_obj[oid] = selected
        keep_names = {m.get("file_name") for m in selected}
        masked_by_obj[oid] = [m for m in matches if m.get("file_name") not in keep_names]

    # Determine majority class and count (after caps).
    pos_counts = {oid: len(rows) for oid, rows in selected_by_obj.items()}
    majority_cat = max(pos_counts, key=pos_counts.get)
    majority_count = pos_counts[majority_cat]

    copied_paths: set[Path] = set()

    def maybe_copy(rows: list[dict], subset: str) -> list[dict]:
        if args.copy_images and rows:
            return copy_matches(rows, frame_index, args.frames_root, args.output_dir, copied_paths, subset)
        return tag_subset(rows, subset)

    # Flatten per-object selections/masks with subset tags.
    combined_rows: list[dict] = []
    for oid, rows in selected_by_obj.items():
        combined_rows.extend(maybe_copy(rows, f"obj{oid}_keep"))
    for oid, rows in masked_by_obj.items():
        if rows:
            combined_rows.extend(maybe_copy(rows, f"obj{oid}_mask"))

    augmented_rows: list[dict] = []

    save_csv(combined_rows, args.output_dir / "samples.csv")
    print(f"Wrote combined samples to {args.output_dir}/samples.csv")

    for oid in object_ids:
        print(
            f"Obj {oid}: selected {len(selected_by_obj[oid])} of {len(matches_by_obj[oid])} (verb={args.verb_id}), subset=keep/mask"
        )
    print(f"Majority category: {majority_cat} with {majority_count} samples")
    if args.copy_images:
        print(f"Copied images under {args.output_dir}/images")

    # Optional: write updated annotations with IV masked and augmented/masked clones appended.
    if args.annotations_out:
        with args.annotations.open("r") as f:
            all_entries = json.load(f)
        if not isinstance(all_entries, list):
            raise SystemExit(f"Expected list in {args.annotations}")

        # Map filename -> set(object categories to remove from that frame)
        mask_lookup: Dict[str, set[int]] = {}
        for oid, rows in masked_by_obj.items():
            for m in rows:
                fname = m.get("file_name")
                if not fname:
                    continue
                mask_lookup.setdefault(fname, set()).add(oid)

        filtered_entries = []
        removed_interactions = 0
        for entry in all_entries:
            fname = entry.get("file_name")
            cats_to_remove = mask_lookup.get(fname, set())
            if cats_to_remove:
                new_entry, removed_here = _prune_entry_annotations_and_hois(
                    entry,
                    drop_object_categories=cats_to_remove,
                    keep_only_object_category=None,
                    keep_masked_annotations=args.keep_masked_boxes,
                )
                removed_interactions += removed_here
                filtered_entries.append(new_entry)
            else:
                filtered_entries.append(entry)

        aug_index: Dict[str, List[str]] = {}
        if args.augment_root:
            if args.augment_root.exists():
                aug_index = build_aug_index(args.augment_root)
            else:
                print(f"[WARN] augment_root not found: {args.augment_root} (skipping augmentation annotations)")

        augmented_entries = []
        if aug_index:
            for entry in filtered_entries:
                fname = entry.get("file_name")
                if not fname:
                    continue
                aug_list = aug_index.get(fname, [])
                for aug_name in aug_list:
                    clone = dict(entry)
                    clone["file_name"] = aug_name
                    augmented_entries.append(clone)

        # Create masked clones to upsample every class up to the top-count class,
        # masking any higher-frequency classes in those clones.
        masked_entries = []
        clone_stats: Dict[int, Dict[str, int]] = {}
        clone_counters: Dict[int, int] = {}
        entry_map = {e.get("file_name"): e for e in filtered_entries}

        def make_masked_clone(match: dict, target_cat: int, mask_cats: set[int]) -> Optional[dict]:
            stat = clone_stats.setdefault(target_cat, {"made": 0, "no_image": 0, "empty": 0, "no_mask_applied": 0})
            fname = match.get("file_name")
            if not fname or fname not in entry_map:
                stat["no_image"] += 1
                return None
            entry = entry_map[fname]
            anns = entry.get("annotations", []) or []
            pruned_entry, _removed = _prune_entry_annotations_and_hois(
                entry,
                drop_object_categories=mask_cats,
                keep_only_object_category=target_cat,
                keep_masked_annotations=args.keep_masked_boxes,
            )
            kept_hois = pruned_entry.get("hoi_annotation", []) or []
            if not kept_hois:
                stat["empty"] += 1
                return None
            orig_path = Path(fname)
            clone_counters[target_cat] = clone_counters.get(target_cat, 0) + 1
            suffix = f"_mask{clone_counters[target_cat]}"
            src_img = resolve_image_path(fname)
            if not src_img:
                stat["no_image"] += 1
                return None
            # Save next to original (same folder) with _mask suffix to avoid overwriting.
            try:
                rel_to_root = src_img.relative_to(args.frames_root)
            except Exception:
                rel_to_root = Path(src_img.name)
            masked_rel = rel_to_root.parent / (rel_to_root.stem + suffix + rel_to_root.suffix)
            dest_img = args.frames_root / masked_rel
            if not mask_objects(src_img, dest_img, anns, mask_cats):
                stat["no_mask_applied"] += 1
                return None
            if args.copy_images:
                dbg_dest = args.output_dir / "images" / masked_rel
                dbg_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest_img, dbg_dest)
            clone_entry = dict(pruned_entry)
            clone_entry["file_name"] = str(masked_rel)
            stat["made"] += 1
            return clone_entry

        # Rank classes by current kept count.
        ranked = sorted(object_ids, key=lambda oid: len(selected_by_obj.get(oid, [])), reverse=True)
        target_count = len(selected_by_obj.get(ranked[0], [])) if ranked else 0

        for idx, obj_cat in enumerate(ranked[1:], start=1):
            current = len(selected_by_obj.get(obj_cat, []))
            if current >= target_count:
                continue
            # Mask all higher-frequency categories in clones.
            mask_cats = set(ranked[:idx])
            src_pool = selected_by_obj.get(obj_cat, []) or matches_by_obj.get(obj_cat, [])
            if not src_pool:
                continue
            attempts = 0
            # Try a bounded number of attempts to reach target_count.
            while len(selected_by_obj[obj_cat]) < target_count and attempts < target_count * 3:
                m = rng.choice(src_pool)
                clone = make_masked_clone(m, obj_cat, mask_cats)
                attempts += 1
                if clone:
                    masked_entries.append(clone)
                    # Track the new sample for counts (verb_id reused if available).
                    selected_by_obj[obj_cat].append(
                        {
                            "file_name": clone["file_name"],
                            "verb_id": m.get("verb_id"),
                            "object_category": obj_cat,
                        }
                    )

        final_entries = filtered_entries + augmented_entries + masked_entries
        args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
        with args.annotations_out.open("w") as f:
            json.dump(final_entries, f, indent=2)
        print(
            f"Wrote {len(final_entries)} entries to {args.annotations_out} "
            f"(removed {removed_interactions} IV interactions, added {len(augmented_entries)} augmented clones, "
            f"added {len(masked_entries)} masked minority clones)"
        )
        if clone_stats:
            for oid in sorted(clone_stats):
                stat = clone_stats[oid]
                print(
                    f"[clone-stats] obj {oid}: made={stat['made']} no_image={stat['no_image']} "
                    f"empty={stat['empty']} no_mask_applied={stat['no_mask_applied']}"
                )


if __name__ == "__main__":
    main()
