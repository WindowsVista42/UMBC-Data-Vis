"""
Encode recipes as one-hot vectors based on tag presence.

Unlike assign.py (which uses KNN to infer categories), this does a direct
deterministic lookup: each recipe is assigned the first matching tag from
the list (most specific first). Recipes with no matching tag get the
--other label.

Output filename is derived from the input .txt filename:
  cook_time.txt -> recipes_cook_time_proba.npy
                   recipes_cook_time_classes.json

Outputs a one-hot (N, n_classes) float32 matrix in the same format as
assign.py, so project.py picks it up automatically.

Usage:
    uv run pipeline/tag_encode.py pipeline/cook_time.txt
    uv run pipeline/tag_encode.py pipeline/cook_time.txt --other longer
    uv run pipeline/tag_encode.py pipeline/cook_time.txt --other none
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX  = os.path.join(SCRIPT_DIR, "recipes_index.json")
JSONL_PATH     = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags_file", help="Path to .txt file with tags in priority order (most specific first)")
    parser.add_argument("--index",   default=DEFAULT_INDEX, help="Path to recipes_index.json")
    parser.add_argument("--other",   default="other",
                        help="Label for recipes with no matching tag (default: 'other')")
    return parser.parse_args()


def load_tags(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def output_paths(tags_file):
    base = os.path.splitext(os.path.basename(tags_file))[0]
    return (
        os.path.join(SCRIPT_DIR, f"recipes_{base}_proba.npy"),
        os.path.join(SCRIPT_DIR, f"recipes_{base}_classes.json"),
    )


def main():
    args = parse_args()

    if not os.path.exists(args.tags_file):
        print(f"ERROR: File not found: {args.tags_file}")
        sys.exit(1)

    tags     = load_tags(args.tags_file)
    classes  = tags + [args.other]
    tag_set  = set(tags)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"Tags (priority order): {', '.join(tags)}")
    print(f"Catch-all label:       {args.other}")

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    id_to_idx = {entry["id"]: entry["index"] for entry in index}

    # Scan JSONL and assign each recipe to first matching tag
    assignments = {}
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r       = json.loads(line)
            rid     = r.get("id")
            recipe_tags = set(r.get("tags") or [])
            label   = args.other
            for tag in tags:
                if tag in recipe_tags:
                    label = tag
                    break
            assignments[rid] = label

    # Build one-hot matrix in embedding index order
    n_classes  = len(classes)
    n_recipes  = len(index)
    proba      = np.zeros((n_recipes, n_classes), dtype=np.float32)

    for entry in index:
        rid = entry["id"]
        row = entry["index"]
        label = assignments.get(rid, args.other)
        proba[row, class_to_idx[label]] = 1.0

    proba_out, classes_out = output_paths(args.tags_file)

    np.save(proba_out, proba)
    print(f"\nSaved {proba_out}  shape={proba.shape}")

    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print(f"Saved {classes_out}")

    dist = Counter(assignments.values())
    print(f"\nDistribution:")
    for label in classes:
        count = dist.get(label, 0)
        print(f"  {label:<30} {count:>7,}  ({count * 100 / len(assignments):.1f}%)")


if __name__ == "__main__":
    main()
