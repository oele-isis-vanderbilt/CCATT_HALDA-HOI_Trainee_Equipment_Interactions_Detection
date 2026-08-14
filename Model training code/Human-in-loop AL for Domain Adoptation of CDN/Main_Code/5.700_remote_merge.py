#!/usr/bin/env python3
import json
import argparse
from copy import deepcopy

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def merge_by_filename(list1, list2):
    merged = {}
    
    # index first file by filename
    for item in list1:
        merged[item["file_name"]] = deepcopy(item)

    # add items from second file
    for item in list2:
        fname = item["file_name"]
        if fname not in merged:
            merged[fname] = deepcopy(item)
        else:
            # OPTIONAL: merge HOI annotations if same frame appears twice
            merged[fname]["hoi_annotation"].extend(
                deepcopy(item["hoi_annotation"])
            )

    return list(merged.values())

def parse_args():
    ap = argparse.ArgumentParser(description="Merge Phase-1 and Phase-2 HICO JSON annotations by file_name.")
    ap.add_argument("--phase1-json", default="merged_hico_annotations.json", help="Phase-1 JSON path.")
    ap.add_argument("--phase2-json", default="merged_hico_annotations_merged_ph2_with_aug.json", help="Phase-2 JSON path.")
    ap.add_argument("--output-json", default="trainval_hico.json", help="Merged output JSON path.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file1 = args.phase1_json
    file2 = args.phase2_json
    output = args.output_json

    data1 = load_json(file1)
    data2 = load_json(file2)

    assert isinstance(data1, list)
    assert isinstance(data2, list)

    merged = merge_by_filename(data1, data2)
    save_json(merged, output)

    print(f"Merged {len(data1)} + {len(data2)} → {len(merged)} unique frames")
    print(f"Saved -> {output}")
