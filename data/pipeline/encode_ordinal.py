"""
Encode a single numeric recipe field as an ordinal feature for UMAP augmentation.

Three encoding types are supported:

  bins      - Hard binning using explicit edge boundaries. A value falls
              entirely into one bin. Output: single float in [0, 1] where
              0 = first bin and 1 = last bin. Config requires "edges"
              (N-1 split points) and "labels" (N bin names). Values below
              the first edge go to bin 0; above the last edge to the last bin.

  soft_bins - Soft binning with linear interpolation between bin centers.
              Output: single float in [0, 1] computed as the weighted average
              of normalized bin positions. A value halfway between bin 2 and
              bin 3 of 5 gets 0.5 * (2/4) + 0.5 * (3/4) = 0.625. Config
              requires "centers" and "labels" of equal length.

  normalize - Min-max normalization to [0, 1] using the actual data range.
              Good for continuous values like dates where explicit bin
              placement would be arbitrary.

All types output a single float column per recipe. bins and soft_bins also
output a per-recipe category JSON with the winning label.

Output filenames are derived from the config filename:
  n_steps.json   -> artifacts/recipes_n_steps_features.npy
  submitted.json -> artifacts/recipes_submitted_features.npy

By default reads field values from RAW_recipes.jsonl. Use --contrib to read
from a recipe_contrib_*.json.gz file instead (e.g. for avg_rating, n_ratings).

Usage:
    uv run pipeline/encode_ordinal.py pipeline/configs/n_steps.json
    uv run pipeline/encode_ordinal.py pipeline/configs/avg_rating.json --contrib artifacts/recipe_contrib_ratings.json.gz
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "..", "artifacts")
DEFAULT_INDEX = os.path.join(ARTIFACTS_DIR, "recipes_index.json")
JSONL_PATH    = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to JSON config file")
    parser.add_argument("--index",   default=DEFAULT_INDEX, help="Path to recipes_index.json")
    parser.add_argument("--contrib", default=None,
                        help="Path to a recipe_contrib_*.json.gz to read field values from "
                             "instead of RAW_recipes.jsonl")
    return parser.parse_args()


def output_paths(config_file):
    base = os.path.splitext(os.path.basename(config_file))[0]
    return (
        os.path.join(ARTIFACTS_DIR, f"recipes_{base}_features.npy"),
        os.path.join(ARTIFACTS_DIR, f"recipes_{base}_classes.json"),
        os.path.join(ARTIFACTS_DIR, f"recipes_{base}.json"),
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


def hard_bin_position(value, edges):
    """Return normalized bin position in [0, 1] using hard boundaries."""
    n = len(edges) + 1
    idx = n - 1

    if value <= edges[0]:
        return 0.0
    if value >= edges[-1]:
        return 1.0
    for i in range(n - 1):
        if edges[i] < value <= edges[i + 1]:
            t = (value - edges[i]) / (edges[i + 1] - edges[i])
            pos_lo = i / (n - 1)
            pos_hi = (i + 1) / (n - 1)
            return pos_lo + t * (pos_hi - pos_lo)
    return 0.5


def soft_bin_position(value, centers):
    """Return normalized bin position in [0, 1] as a soft-interpolated weighted average."""
    n = len(centers)
    if value <= centers[0]:
        return 0.0
    if value >= centers[-1]:
        return 1.0
    for i in range(n - 1):
        if centers[i] <= value <= centers[i + 1]:
            t = (value - centers[i]) / (centers[i + 1] - centers[i])
            pos_lo = i / (n - 1)
            pos_hi = (i + 1) / (n - 1)
            return pos_lo + t * (pos_hi - pos_lo)
    return 0.5


def main():
    args = parse_args()

    for path in [args.config, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    field      = config["field"]
    kind       = config.get("type", "bins")
    missing_cfg = config.get("missing_value", None)

    if missing_cfg is None:
        one_hot_missing = False
        missing_val     = None
        missing_label   = None
    elif isinstance(missing_cfg, dict) and missing_cfg.get("type") == "one_hot":
        one_hot_missing = True
        missing_val     = None
        missing_label   = missing_cfg["label"]
    elif isinstance(missing_cfg, dict) and missing_cfg.get("type") == "value":
        one_hot_missing = False
        missing_val     = missing_cfg["value"]
        missing_label   = None
    else:
        print(f"ERROR: missing_value must be {{\"type\": \"one_hot\", \"label\": \"...\"}} or {{\"type\": \"value\", \"value\": N}}")
        sys.exit(1)

    if kind == "bins":
        edges  = config["edges"]
        labels = config["labels"]
        assert len(labels) == len(edges) + 1, "labels must have exactly len(edges)+1 entries"
        if one_hot_missing:
            labels = [missing_label] + labels
        print(f"Field '{field}': {len(labels)} hard bins -> single normalized value")
        if one_hot_missing:
            print(f"  [missing]         '{missing_label}'")
        for i, label in enumerate(config["labels"]):
            lo = f"{edges[i-1]}" if i > 0 else "-inf"
            hi = f"{edges[i]}"   if i < len(edges) else "+inf"
            print(f"  [{lo}, {hi})  '{label}'")
    elif kind == "soft_bins":
        centers = config["centers"]
        labels  = config["labels"]
        assert len(centers) == len(labels), "centers and labels must have the same length"
        if one_hot_missing:
            labels = [missing_label] + labels
        print(f"Field '{field}': {len(labels)} soft bins -> single normalized value")
        if one_hot_missing:
            print(f"  [missing]         '{missing_label}'")
        for label, center in zip(config["labels"], centers):
            print(f"  center={center:<6}  '{label}'")
    elif kind == "normalize":
        labels = [config["label"]]
        print(f"Field '{field}': min-max normalize -> '{labels[0]}'")
    else:
        print(f"ERROR: Unknown type '{kind}'. Expected: bins, soft_bins, normalize")
        sys.exit(1)

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    id_to_idx = {entry["id"]: entry["index"] for entry in index}

    n_cols      = 2 if one_hot_missing else 1
    features    = np.zeros((len(index), n_cols), dtype=np.float32)
    bin_indices = np.zeros(len(index), dtype=np.int32)

    contrib_data = None
    if args.contrib:
        print(f"Reading from contrib file: {args.contrib}")
        with gzip.open(args.contrib, "rt", encoding="utf-8") as f:
            contrib_data = json.load(f)

    def get_record_value(rid):
        if contrib_data is not None:
            rec = contrib_data.get(str(rid))
            return extract_value(rec, field) if rec else None
        return None

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

    missing = 0

    def encode_row(row, val):
        nonlocal missing
        if val is None:
            if one_hot_missing:
                features[row, 0] = 0.0
                features[row, 1] = 1.0
                # bin_indices[row] stays 0 = missing_label (prepended)
                missing += 1
                return
            elif missing_val is not None:
                val = missing_val
            else:
                features[row, 0] = 0.5
                missing += 1
                return
        offset = 1 if one_hot_missing else 0
        if kind == "bins":
            idx = next((i for i, e in enumerate(edges) if val < e), len(edges))
            bin_indices[row]  = idx + offset
            features[row, 0]  = hard_bin_position(val, edges)
        elif kind == "soft_bins":
            idx = min(range(len(centers)), key=lambda i: abs(centers[i] - val))
            bin_indices[row]  = idx + offset
            features[row, 0]  = soft_bin_position(val, centers)
        else:
            features[row, 0] = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    if contrib_data is not None:
        for entry in index:
            encode_row(entry["index"], get_record_value(entry["id"]))
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

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    features_out, classes_out, json_out = output_paths(args.config)

    np.save(features_out, features)
    print(f"\nSaved {features_out}  shape={features.shape}")

    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"Saved {classes_out}")

    if missing:
        print(f"  {missing:,} recipes had missing values")

    if kind in ("bins", "soft_bins"):
        n_bins = len(labels)
        results = []
        for entry in index:
            row     = entry["index"]
            val     = float(features[row, 0])
            bin_idx = int(bin_indices[row])
            results.append({
                "id":       entry["id"],
                "category": labels[bin_idx],
                "score":    round(val, 4),
                "runners_up": [],
            })
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, separators=(",", ":"))
        print(f"Saved {json_out}  ({len(results):,} entries)")
    else:
        print(f"Skipping {json_out} (normalize type has no categorical assignment)")


if __name__ == "__main__":
    main()
