"""
Encode a single numeric recipe field as an ordinal feature for UMAP augmentation.

Encoding types (controls how value maps to a bin):

  bins      - Hard binning using explicit edge boundaries. Config requires
              "edges" (N-1 split points) and "labels" (N bin names).
              Values below the first edge go to bin 0; above the last to last bin.

  normalize - Min-max normalization to [0, 1] using actual data range.
              Good for continuous values like dates.

Output formats (controls shape of feature vector):

  rank_hot  - N columns. Cumulative encoding: col i = 1 if value >= bin i,
              0 if below, fractional if at the boundary. Preserves ordinality
              — adjacent bins are always closer than distant bins.
              Default for bins.

  scalar    - 1 column. Single float in [0, 1] representing bin position.

Missing value handling:

  {"type": "value",   "value": N}          - treat missing as N
  {"type": "one_hot", "label": "..."}      - add orthogonal missing column,
                                             prepend label at index 0

Output filenames are derived from the config filename:
  n_steps.json   -> artifacts/recipes_n_steps_features.npy
  submitted.json -> artifacts/recipes_submitted_features.npy

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


def rank_hot_encode(value, edges):
    """Cumulative rank-hot encoding. Returns N-length vector where col i is 1 if value
    >= bin i, 0 if below, and fractional at the boundary between bins."""
    n = len(edges) + 1
    vec = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i == 0:
            lo = -np.inf
            hi = edges[0]
        elif i == n - 1:
            lo = edges[-1]
            hi = np.inf
        else:
            lo = edges[i - 1]
            hi = edges[i]

        if value >= hi:
            vec[i] = 1.0
        elif value < lo:
            vec[i] = 0.0
        else:
            # fractional position within this bin
            if np.isinf(lo):
                vec[i] = 1.0
            elif np.isinf(hi):
                vec[i] = 1.0
            else:
                vec[i] = (value - lo) / (hi - lo)
    return vec


def hard_bin_index(value, edges):
    return next((i for i, e in enumerate(edges) if value < e), len(edges))


def hard_bin_position(value, edges):
    """Single float [0,1] for scalar output."""
    n = len(edges) + 1
    idx = hard_bin_index(value, edges)
    return idx / (n - 1)



def main():
    args = parse_args()

    for path in [args.config, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    field       = config["field"]
    kind        = config.get("type", "bins")
    output_fmt  = config.get("output", "rank_hot" if kind == "bins" else "scalar")
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
        print("ERROR: missing_value must be {\"type\": \"one_hot\", \"label\": \"...\"} or {\"type\": \"value\", \"value\": N}")
        sys.exit(1)

    if kind == "bins":
        edges  = config["edges"]
        labels = config["labels"]
        assert len(labels) == len(edges) + 1, "labels must have exactly len(edges)+1 entries"
        if one_hot_missing:
            labels = [missing_label] + labels
        print(f"Field '{field}': {len(config['labels'])} hard bins  output={output_fmt}")
        if one_hot_missing:
            print(f"  [missing]  '{missing_label}'")
        for i, label in enumerate(config["labels"]):
            lo = f"{edges[i-1]}" if i > 0 else "-inf"
            hi = f"{edges[i]}"   if i < len(edges) else "+inf"
            print(f"  [{lo}, {hi})  '{label}'")
    elif kind == "normalize":
        output_fmt = "scalar"
        labels = [config["label"]]
        print(f"Field '{field}': normalize -> '{labels[0]}'")
    else:
        print(f"ERROR: Unknown type '{kind}'. Expected: bins, normalize")
        sys.exit(1)

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    id_to_idx = {entry["id"]: entry["index"] for entry in index}

    # Determine feature columns
    n_data_bins = len(config.get("labels", [config.get("label")]))
    if output_fmt == "rank_hot":
        n_cols = n_data_bins + (1 if one_hot_missing else 0)
    else:
        n_cols = 1 + (1 if one_hot_missing else 0)

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
        missing_col = n_data_bins if output_fmt == "rank_hot" else 1
        if val is None:
            if one_hot_missing:
                features[row, missing_col] = 1.0
                # bin_indices[row] = 0 = missing_label
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
            idx = hard_bin_index(val, edges)
            bin_indices[row] = idx + offset
            if output_fmt == "rank_hot":
                features[row, :n_data_bins] = rank_hot_encode(val, edges)
            else:
                features[row, 0] = hard_bin_position(val, edges)
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

    if kind == "bins":
        results = []
        for entry in index:
            row     = entry["index"]
            bin_idx = int(bin_indices[row])
            results.append({
                "id":       entry["id"],
                "category": labels[bin_idx],
                "score":    round(float(features[row, 0]), 4),
                "runners_up": [],
            })
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, separators=(",", ":"))
        print(f"Saved {json_out}  ({len(results):,} entries)")
    else:
        print(f"Skipping {json_out} (normalize type has no categorical assignment)")


if __name__ == "__main__":
    main()
