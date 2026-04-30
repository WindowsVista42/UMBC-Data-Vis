"""
Export the data pipeline outputs to a format suitable for the Three.js web app.

Reads:
  - pipeline/recipes_umap3d.npy + recipes_umap3d_index.json  (coordinates + IDs)
  - pipeline/*_proba.npy and *_features.npy + matching *_classes.json  (categories)
  - pipeline/recipe_contrib_*.json.gz  (extra per-recipe fields, e.g. ratings)
  - raw/RAW_recipes.jsonl  (base recipe metadata)

Writes to data/export/:
  geometry.drc          Draco point cloud: positions + per-point attributes
  meta.json             Manifest: totals, categories, chunk info, coord bounds
  chunks/chunk_XXXXXX.json.gz  Per-recipe metadata dicts, spatially organized

Points are sorted by Z-order (Morton code) so nearby UMAP points land in the
same metadata chunk, giving good cache locality for hover interactions.

Usage:
    uv run pipeline/export.py
    uv run pipeline/export.py --chunk-size 500 --output ../export
"""

import argparse
import glob
import gzip
import json
import math
import os
import sys

import DracoPy
import numpy as np

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "..", "artifacts")
DEFAULT_OUT   = os.path.join(SCRIPT_DIR, "..", "export")
JSONL_PATH    = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")


DEFAULT_SITE = os.path.join(SCRIPT_DIR, "..", "..", "site", "data")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output",     default=DEFAULT_OUT, help="Output directory (default: data/export)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Recipes per metadata chunk (default: 500)")
    parser.add_argument("--no-copy",    action="store_true",  help="Skip copying output to site/data/")
    return parser.parse_args()


# --- Morton encoding (vectorized numpy, no Python loop) ---

def morton_encode(xi, yi, zi):
    """Encode integer coordinates as 63-bit Morton codes (Z-order curve)."""
    def expand(v):
        v = v.astype(np.int64) & 0x1fffff
        v = (v | (v << 32)) & 0x1f00000000ffff
        v = (v | (v << 16)) & 0x1f0000ff0000ff
        v = (v | (v << 8))  & 0x100f00f00f00f00f
        v = (v | (v << 4))  & 0x10c30c30c30c30c3
        v = (v | (v << 2))  & 0x1249249249249249
        return v
    return expand(xi) | (expand(yi) << 1) | (expand(zi) << 2)


# --- Feature file discovery ---

def find_feature_files():
    """
    Return list of (stem, npy_path, classes) for all *_proba.npy / *_features.npy
    files that have a matching *_classes.json. Skips the embeddings file.
    """
    results = []
    for npy_path in sorted(
        glob.glob(os.path.join(ARTIFACTS_DIR, "*_proba.npy")) +
        glob.glob(os.path.join(ARTIFACTS_DIR, "*_features.npy"))
    ):
        classes_path = (npy_path
                        .replace("_proba.npy",    "_classes.json")
                        .replace("_features.npy", "_classes.json"))
        if not os.path.exists(classes_path):
            continue
        with open(classes_path, encoding="utf-8") as f:
            classes = json.load(f)
        stem = (os.path.basename(npy_path)
                .replace("recipes_", "")
                .replace("_proba.npy",    "")
                .replace("_features.npy", ""))
        results.append((stem, npy_path, classes))
    return results


def find_contrib_files():
    """Return all artifacts/recipe_contrib_*.json.gz paths."""
    return sorted(glob.glob(os.path.join(ARTIFACTS_DIR, "recipe_contrib_*.json.gz")))


# --- Main ---

def main():
    args = parse_args()
    out_dir    = os.path.abspath(args.output)
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Load coordinates and index
    coords_path = os.path.join(ARTIFACTS_DIR, "recipes_umap3d.npy")
    index_path  = os.path.join(ARTIFACTS_DIR, "recipes_umap3d_index.json")
    for p in [coords_path, index_path, JSONL_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print("Loading UMAP coordinates...")
    coords = np.load(coords_path).astype(np.float32)  # (N, 3)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    N = len(index)
    assert coords.shape[0] == N, f"Coords rows {coords.shape[0]} != index entries {N}"
    recipe_ids = np.array([entry["id"] for entry in index], dtype=np.int64)
    print(f"  {N:,} points")

    # Load feature files (categories + scalars)
    feature_files = find_feature_files()
    print(f"\nFeature files ({len(feature_files)}):")
    categories    = []  # bin-type: {stem, labels, data (N,) uint8}
    scalars       = []  # normalize-type: {stem, label, data (N,) float32}

    for stem, npy_path, classes in feature_files:
        arr = np.load(npy_path)  # (N, k)
        if len(classes) > 1:
            if arr.shape[1] == 1:
                # Ordinal: single float [0,1] -> bin index
                n = len(classes)
                cat_ids = np.round(arr[:, 0] * (n - 1)).clip(0, n - 1).astype(np.uint8)
                print(f"  [ordinal]   {stem}  ({n} labels)")
            else:
                # Multi-column proba: argmax -> category index
                cat_ids = np.argmax(arr, axis=1).astype(np.uint8)
                print(f"  [bin]       {stem}  ({len(classes)} labels)")
            categories.append({"stem": stem, "labels": classes, "data": cat_ids})
        else:
            scalars.append({"stem": stem, "label": classes[0], "data": arr[:, 0].astype(np.float32)})
            print(f"  [normalize] {stem}  label='{classes[0]}'")

    # Load contrib files
    contrib_files = find_contrib_files()
    print(f"\nContrib files ({len(contrib_files)}):")
    contrib = {}  # recipe_id_str -> merged dict of extra fields
    for path in contrib_files:
        print(f"  {os.path.basename(path)}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for rid_str, fields in data.items():
            contrib.setdefault(rid_str, {}).update(fields)
    print(f"  {len(contrib):,} recipes have contrib data")

    # Z-order sort
    print("\nComputing Z-order sort...")
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    crange = np.maximum(cmax - cmin, 1e-9)
    INT_MAX = 2**21
    xi = ((coords[:, 0] - cmin[0]) / crange[0] * (INT_MAX - 1)).astype(np.int64)
    yi = ((coords[:, 1] - cmin[1]) / crange[1] * (INT_MAX - 1)).astype(np.int64)
    zi = ((coords[:, 2] - cmin[2]) / crange[2] * (INT_MAX - 1)).astype(np.int64)
    morton_codes = morton_encode(xi, yi, zi)
    order        = np.argsort(morton_codes)

    # Apply sort order
    coords_s     = coords[order]
    recipe_ids_s = recipe_ids[order]
    chunk_ids    = (np.arange(N) // args.chunk_size).astype(np.int32)
    n_chunks     = math.ceil(N / args.chunk_size)

    # Encode Draco (positions only — generic attributes stored separately as
    # attributes.bin.gz using proper integer types, giving much better compression)
    print(f"Encoding Draco ({N:,} points, positions only)...")
    drc_bytes = DracoPy.encode(
        coords_s,
        preserve_order=True,
        quantization_bits=16,
        compression_level=10,
    )
    drc_path = os.path.join(out_dir, "geometry.drc")
    with open(drc_path, "wb") as f:
        f.write(drc_bytes)
    print(f"  Saved {drc_path}  ({len(drc_bytes)/1e6:.2f} MB)")

    # Write attributes.bin.gz — all per-point attributes in Z-order, packed as
    # their native integer types for good gzip compression.
    # Layout (N entries each, same Z-order as geometry.drc):
    #   recipe_id  : uint32 x N
    #   chunk_id   : uint16 x N
    #   <cat stem> : uint8  x N  (one block per bin-type category family)
    #   <sc stem>  : float32 x N (one block per normalize-type scalar)
    attribute_layout = [{"name": "recipe_id", "dtype": "uint32"},
                        {"name": "chunk_id",  "dtype": "uint16"}]
    raw_parts = [recipe_ids_s.astype(np.uint32).tobytes(),
                 chunk_ids.astype(np.uint16).tobytes()]
    for cat in categories:
        raw_parts.append(cat["data"][order].astype(np.uint8).tobytes())
        attribute_layout.append({"name": cat["stem"], "dtype": "uint8"})
    for sc in scalars:
        raw_parts.append(sc["data"][order].astype(np.float32).tobytes())
        attribute_layout.append({"name": sc["stem"], "dtype": "float32"})

    attr_path = os.path.join(out_dir, "attributes.bin.gz")
    with gzip.open(attr_path, "wb", compresslevel=9) as f:
        for part in raw_parts:
            f.write(part)
    print(f"  Saved {attr_path}  ({os.path.getsize(attr_path)/1e6:.2f} MB)")

    # Load recipe metadata from JSONL
    print("\nLoading recipe metadata...")
    base_meta = {}
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r["id"]
            base_meta[rid] = {
                "name":         r.get("name", ""),
                "description":  (r.get("description") or "").strip(),
                "ingredients":  r.get("ingredients") or [],
                "minutes":      r.get("minutes"),
                "n_steps":      r.get("n_steps"),
                "n_ingredients": r.get("n_ingredients"),
                "submitted":    r.get("submitted"),
            }
    print(f"  {len(base_meta):,} recipes loaded")

    # Write metadata chunks (Z-order sorted, keyed by recipe_id string)
    print(f"Writing {n_chunks} metadata chunks (chunk_size={args.chunk_size})...")
    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        end   = min(start + args.chunk_size, N)
        chunk_data = {}
        for i in range(start, end):
            rid     = int(recipe_ids_s[i])
            rid_str = str(rid)
            entry   = dict(base_meta.get(rid, {}))
            entry.update(contrib.get(rid_str, {}))
            chunk_data[rid_str] = entry

        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:06d}.json.gz")
        with gzip.open(chunk_path, "wt", encoding="utf-8") as f:
            json.dump(chunk_data, f, separators=(",", ":"), ensure_ascii=False)

        if chunk_idx % 100 == 0 or chunk_idx == n_chunks - 1:
            print(f"  {chunk_idx + 1} / {n_chunks}", flush=True)

    # Write meta.json
    meta = {
        "total":      N,
        "n_chunks":   n_chunks,
        "chunk_size": args.chunk_size,
        "coord_bounds": {
            "min": cmin.tolist(),
            "max": cmax.tolist(),
        },
        # Layout of attributes.bin.gz — N entries per block, packed contiguously
        "attribute_layout": attribute_layout,
        "categories": [
            {"name": cat["stem"], "attribute": cat["stem"], "labels": cat["labels"]}
            for cat in categories
        ],
        "scalar_attributes": [
            {"name": sc["label"], "attribute": sc["stem"]}
            for sc in scalars
        ],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {meta_path}")

    # Summary
    chunk_sizes_mb = sum(
        os.path.getsize(os.path.join(chunks_dir, f))
        for f in os.listdir(chunks_dir)
    ) / 1e6
    attr_mb = os.path.getsize(attr_path) / 1e6
    print(f"\n{'='*50}")
    print(f"  Points          : {N:,}")
    print(f"  Chunks          : {n_chunks}")
    print(f"  geometry.drc    : {len(drc_bytes)/1e6:.2f} MB")
    print(f"  attributes.bin.gz: {attr_mb:.2f} MB")
    print(f"  Chunks total    : {chunk_sizes_mb:.1f} MB")
    print(f"  Total           : {len(drc_bytes)/1e6 + attr_mb + chunk_sizes_mb:.1f} MB")
    print(f"  Output          : {out_dir}")
    print(f"{'='*50}")

    # Copy to site/data/ unless --no-copy
    if not args.no_copy:
        import shutil
        site_data = os.path.abspath(DEFAULT_SITE)
        print(f"\nCopying to {site_data}...")
        if os.path.exists(site_data):
            shutil.rmtree(site_data)
        shutil.copytree(out_dir, site_data)
        print(f"  Done.")


if __name__ == "__main__":
    main()
