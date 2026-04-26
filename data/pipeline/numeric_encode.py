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

By default reads field values from RAW_recipes.jsonl. Use --contrib to read
from a recipe_contrib_*.json.gz file instead (e.g. for avg_rating, n_ratings).

Usage:
    uv run pipeline/numeric_encode.py pipeline/n_steps.json
    uv run pipeline/numeric_encode.py pipeline/avg_rating.json --contrib pipeline/recipe_contrib_ratings.json.gz
"""

import argparse
import gzip
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
    parser.add_argument("--index",  default=DEFAULT_INDEX, help="Path to recipes_index.json")
    parser.add_argument("--contrib", default=None,
                        help="Path to a recipe_contrib_*.json.gz to read field values from "
                             "instead of RAW_recipes.jsonl (e.g. pipeline/recipe_contrib_ratings.json.gz)")
    return parser.parse_args()


def output_paths(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return (
        os.path.join(SCRIPT_DIR, f"recipes_{base}_features.npy"),
        os.path.join(SCRIPT_DIR, f"recipes_{base}_classes.json"),
        os.path.join(SCRIPT_DIR, f"recipes_{base}.json"),
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

    # Load contrib data if provided
    contrib_data = None
    if args.contrib:
        print(f"Reading from contrib file: {args.contrib}")
        with gzip.open(args.contrib, "rt", encoding="utf-8") as f:
            contrib_data = json.load(f)  # {recipe_id_str: {field: value, ...}}

    def get_record_value(rid):
        if contrib_data is not None:
            rec = contrib_data.get(str(rid))
            return extract_value(rec, field) if rec else None
        return None  # caller handles JSONL path

    # For normalize: compute min/max
    if kind == "normalize":
        vals = []
        if contrib_data is not None:
            for rec in contrib_data.values():
                val = extract_value(rec, field)
                if val is not None:
                    vals.append(val)
        else:
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

    def encode_row(row, val):
        nonlocal missing
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

    if contrib_data is not None:
        for entry in index:
            rid = entry["id"]
            row = entry["index"]
            encode_row(row, get_record_value(rid))
    else:
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r   = json.loads(line)
                rid = r.get("id")
                if rid not in id_to_idx:
                    continue
                encode_row(id_to_idx[rid], extract_value(r, field))

    features_out, classes_out, json_out = output_paths(args.config)

    np.save(features_out, features)
    print(f"\nSaved {features_out}  shape={features.shape}")

    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Saved {classes_out}")

    if missing:
        print(f"  {missing:,} recipes had missing values")

    if kind == "bins":
        results = []
        for entry in index:
            rid    = entry["id"]
            row    = entry["index"]
            scores = features[row]
            order  = np.argsort(scores)[::-1]
            results.append({
                "id":       rid,
                "category": labels[order[0]],
                "score":    round(float(scores[order[0]]), 4),
                "runners_up": [
                    {"category": labels[order[i]], "score": round(float(scores[order[i]]), 4)}
                    for i in range(1, min(3, len(labels)))
                    if scores[order[i]] > 0
                ],
            })
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, separators=(",", ":"))
        print(f"Saved {json_out}  ({len(results):,} entries)")
    else:
        print(f"Skipping {json_out} (normalize type has no categorical assignment)")


if __name__ == "__main__":
    main()
