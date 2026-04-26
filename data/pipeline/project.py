"""
Project recipe embeddings to 2D or 3D using UMAP for visualization.

Reads the embedding matrix from embed.py and runs a single UMAP pass to
2D or 3D coordinates suitable for a point cloud visualization.

All *_proba.npy files in the pipeline directory (output of assign.py) are
auto-detected and concatenated to the embeddings before projection:

  concat(embedding, category_weight * proba_cuisines, category_weight * proba_meal_types, ...)

This pulls same-category recipes together in the projection. Set
--category-weight 0 to disable, or pass explicit --proba paths to override
auto-detection.

Outputs:
  recipes_umap{N}d.npy         (N, 2|3) float32
  recipes_umap{N}d_index.json  [{"id": <recipe_id>, "index": <int>}, ...]

Usage:
    uv run pipeline/project.py
    uv run pipeline/project.py --dims 2
    uv run pipeline/project.py --category-weight 0.5
    uv run pipeline/project.py --category-weight 0
    uv run pipeline/project.py --proba pipeline/recipes_cuisines_proba.npy
    uv run pipeline/project.py --random-state 42
    uv run pipeline/project.py --max-rows 10000
    uv run pipeline/project.py --neighbors 30 --min-dist 0.05
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import umap

SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMBEDDINGS    = os.path.join(SCRIPT_DIR, "recipes_embeddings.npy")
DEFAULT_INDEX         = os.path.join(SCRIPT_DIR, "recipes_index.json")
DEFAULT_OUTPUT_PREFIX = os.path.join(SCRIPT_DIR, "recipes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings",       default=DEFAULT_EMBEDDINGS, help="Path to embeddings .npy file")
    parser.add_argument("--index",            default=DEFAULT_INDEX,      help="Path to index .json file")
    parser.add_argument("--output-prefix",    default=DEFAULT_OUTPUT_PREFIX, help="Prefix for output files")
    parser.add_argument("--dims",             type=int, default=3, choices=[2, 3], help="Output dimensions (default: 3)")
    parser.add_argument("--neighbors",        type=int, default=15,   help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--min-dist",         type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--metric",           default="cosine",        help="Distance metric (default: cosine)")
    parser.add_argument("--n-jobs",           type=int, default=-1,    help="Parallel jobs, -1 = all cores (default: -1)")
    parser.add_argument("--random-state",     type=int, default=None,  help="Random seed for reproducibility")
    parser.add_argument("--max-rows",         type=int, default=None,  help="Use only first N rows (for testing)")
    parser.add_argument("--proba",            nargs="*", default=None,
                        help="Explicit proba .npy paths to use. Defaults to auto-detecting all "
                             "*_proba.npy files in the pipeline directory.")
    parser.add_argument("--category-weight",  type=float, default=0.5,
                        help="Weight applied to each proba vector before concatenation. "
                             "0 disables augmentation. (default: 0.5)")
    return parser.parse_args()


def find_proba_files():
    """Auto-detect all *_proba.npy and *_features.npy files in the pipeline directory."""
    return sorted(
        glob.glob(os.path.join(SCRIPT_DIR, "*_proba.npy")) +
        glob.glob(os.path.join(SCRIPT_DIR, "*_features.npy"))
    )


def main():
    args = parse_args()

    for path in [args.embeddings, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"  Shape: {embeddings.shape}")

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)

    if args.max_rows:
        n = min(args.max_rows, len(index))
        embeddings = embeddings[:n]
        index = index[:n]
        print(f"  Using first {n:,} rows (--max-rows)")

    if args.category_weight > 0:
        proba_paths = args.proba if args.proba is not None else find_proba_files()
        proba_paths = [p for p in proba_paths if os.path.exists(p)]

        if proba_paths:
            parts = [embeddings]
            for path in proba_paths:
                proba = np.load(path)
                if args.max_rows:
                    proba = proba[:n]
                parts.append(args.category_weight * proba)
                name = os.path.basename(path)
                print(f"  + {name}  ({proba.shape[1]} classes, weight={args.category_weight})")
            embeddings = np.concatenate(parts, axis=1)
            print(f"  Augmented shape: {embeddings.shape}")
        else:
            print("No proba files found - running without augmentation")

    print(f"\nRunning UMAP: {embeddings.shape[1]}D -> {args.dims}D")
    print(f"  neighbors={args.neighbors}, min_dist={args.min_dist}, metric={args.metric}, "
          f"n_jobs={args.n_jobs}, random_state={args.random_state}")
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=args.dims,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")

    coords_path = args.output_prefix + f"_umap{args.dims}d.npy"
    index_path  = args.output_prefix + f"_umap{args.dims}d_index.json"

    np.save(coords_path, coords)
    print(f"Saved {coords_path}  shape={coords.shape}")

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"Saved {index_path}  ({len(index):,} entries)")

    print(f"\n{'='*50}")
    print(f"  Points     : {len(index):,}")
    print(f"  Dims       : {args.dims}")
    print(f"  Coords     : {coords_path}")
    print(f"  Index      : {index_path}")
    print(f"  Time       : {elapsed/60:.1f} min")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
