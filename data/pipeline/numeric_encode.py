"""
Encode a single numeric recipe field as a feature vector for UMAP augmentation.

Two encoding types are supported:

  bins      - Soft binning with linear interpolation between bin centers.
              A value between two centers gets split weight between them.
              Output: one column per bin.

  normalize - Min-max normalization to [0, 1] as a single column.
              Good for continuous values like dates where bin placement
              would be arbitrary.

Output filename is derived from the config filename:
  n_steps.json   -> recipes_n_steps_features.npy
  submitted.json -> recipes_submitted_features.npy

Date fields (strings containing "-") are automatically converted to year.
Missing values get a uniform/midpoint assignment.

Config format:

  Bins:
    {
      "field": "n_steps",
      "type": "bins",
      "centers": [3, 8, 15, 25],
      "labels": ["1-5 steps", "6-10 steps", "11-20 steps", "21+ steps"]
    }

  Normalize:
    {
      "field": "submitted",
      "type": "normalize",
      "label": "date"
    }

Usage:
    uv run pipeline/numeric_encode.py pipeline/n_steps.json
    uv run pipeline/numeric_encode.py pipeline/submitted.json
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(SCRIPT_DIR, "recipes_index.json")
JSONL_PATH    = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Path to recipes_index.json")
    return parser.parse_args()


def output_paths(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return (
        os.path.join(SCRIPT_DIR, f"recipes_{base}_features.npy"),
        os.path.join(SCRIPT_DIR, f"recipes_{base}_classes.json"),
    )


def extract_value(recipe, field):
    """Extract a numeric value from a recipe record. Converts date strings to year."""
    val = recipe.get(field)
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return float(val[:4])
        except (ValueError, TypeError):
            return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def soft_encode(value, centers):
    """Linear interpolation between adjacent bin centers. Returns weights summing to 1."""
    weights = np.zeros(len(centers), dtype=np.float32)
    if value <= centers[0]:
        weights[0] = 1.0
    elif value >= centers[-1]:
        weights[-1] = 1.0
    else:
        for i in range(len(centers) - 1):
            if centers[i] <= value <= centers[i + 1]:
                t = (value - centers[i]) / (centers[i + 1] - centers[i])
                weights[i]     = 1.0 - t
                weights[i + 1] = t
                break
    return weights


def main():
    args = parse_args()

    for path in [args.config, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    field   = config["field"]
    kind    = config.get("type", "bins")

    if kind == "bins":
        centers = config["centers"]
        labels  = config["labels"]
        assert len(centers) == len(labels), "centers and labels must have the same length"
        n_cols  = len(centers)
        print(f"Field '{field}': {n_cols} soft bins")
        for label, center in zip(labels, centers):
            print(f"  center={center:<6}  '{label}'")
    else:
        labels = [config["label"]]
        n_cols = 1
        print(f"Field '{field}': min-max normalize -> '{labels[0]}'")

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    id_to_idx = {entry["id"]: entry["index"] for entry in index}

    features = np.zeros((len(index), n_cols), dtype=np.float32)

    # For normalize: compute min/max from a first pass
    if kind == "normalize":
        vals = []
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                val = extract_value(json.loads(line), field)
                if val is not None:
                    vals.append(val)
        vmin, vmax = min(vals), max(vals)
        print(f"  Range: {vmin} - {vmax}")

    # Encode
    missing = 0
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            rid = r.get("id")
            if rid not in id_to_idx:
                continue
            row = id_to_idx[rid]
            val = extract_value(r, field)

            if kind == "normalize":
                if val is not None and vmax > vmin:
                    features[row, 0] = (val - vmin) / (vmax - vmin)
                else:
                    features[row, 0] = 0.5
                    if val is None:
                        missing += 1
            else:
                if val is not None:
                    features[row] = soft_encode(val, centers)
                else:
                    features[row] = 1.0 / n_cols
                    missing += 1

    features_out, classes_out = output_paths(args.config)

    np.save(features_out, features)
    print(f"\nSaved {features_out}  shape={features.shape}")

    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Saved {classes_out}")

    if missing:
        print(f"  {missing:,} recipes had missing values")


if __name__ == "__main__":
    main()
