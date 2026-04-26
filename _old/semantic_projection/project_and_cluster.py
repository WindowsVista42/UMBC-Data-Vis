# -*- coding: utf-8 -*-
"""
Project embedding vectors to 3D using UMAP, then cluster with HDBSCAN.

Following best practices from the UMAP documentation on clustering:
  - A SEPARATE UMAP embedding is used for clustering (higher n_neighbors,
    min_dist=0.0, more components) vs. the 3D visualization embedding.
  - HDBSCAN runs on the clustering embedding, NOT the 3D coords.
  - This avoids false cluster splits from the 3D projection and gives
    HDBSCAN denser, cleaner input to work with.

Pipeline:
  Clustering: embeddings (768) -> UMAP -> N dims -> HDBSCAN
  Visualization: embeddings (768) -> UMAP -> 3 dims

Reads the outputs from embed_abstracts.py:
  - <base>_embeddings.npy   (N, D) float32 embeddings
  - <base>_index.json       list of {"id": ..., "index": ...}

Outputs:
  - <base>_umap_cluster.npy    (N, C) float32 clustering embedding (cached for --recluster)
  - <base>_umap3d.npy          (N, 3) float32 array of x, y, z coordinates
  - <base>_umap3d_index.json   list of {"id": ..., "index": ..., "cluster": ...}
                               cluster is an int >= 0, or -1 for noise/unclustered

Usage:
    python project_and_cluster.py arxiv-metadata-oai-snapshot
    python project_and_cluster.py arxiv-metadata-oai-snapshot --max-rows 50000
    python project_and_cluster.py arxiv-metadata-oai-snapshot --cluster-components 20
    python project_and_cluster.py arxiv-metadata-oai-snapshot --hdbscan-min 100

    # Re-run only HDBSCAN using the saved clustering embedding (fast):
    python project_and_cluster.py arxiv-metadata-oai-snapshot --recluster --hdbscan-min 100
    python project_and_cluster.py arxiv-metadata-oai-snapshot --recluster --hdbscan-min 50

Requirements:
    uv add numpy umap-learn hdbscan
"""

import os
import sys
import json
import argparse
import time
import numpy as np

# -- Args ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="UMAP 3D projection + HDBSCAN clustering of paper embeddings"
)
parser.add_argument("prefix",
                    help="File prefix (e.g. 'arxiv-metadata-oai-snapshot' to read "
                         "*_embeddings.npy and *_index.json)")

# Visualization UMAP params
parser.add_argument("--viz-neighbors", type=int, default=15,
                    help="UMAP n_neighbors for 3D visualization (default: 15)")
parser.add_argument("--viz-min-dist", type=float, default=0.1,
                    help="UMAP min_dist for 3D visualization (default: 0.1)")

# Clustering UMAP params (separate embedding, per UMAP docs)
parser.add_argument("--cluster-neighbors", type=int, default=30,
                    help="UMAP n_neighbors for clustering embedding (default: 30). "
                         "Larger values capture broader structure.")
parser.add_argument("--cluster-components", type=int, default=40,
                    help="Number of dimensions UMAP outputs for HDBSCAN (default: 40).")

# HDBSCAN params
parser.add_argument("--hdbscan-min", type=int, default=None,
                    help="HDBSCAN min_cluster_size (default: auto)")
parser.add_argument("--hdbscan-samples", type=int, default=None,
                    help="HDBSCAN min_samples (default: auto)")
parser.add_argument("--hdbscan-method", default="eom",
                    choices=["eom", "leaf"],
                    help="HDBSCAN cluster_selection_method (default: eom). "
                         "'leaf' finds smaller clusters and assigns less noise.")

# General
parser.add_argument("--metric", default="cosine",
                    help="Distance metric for both UMAP passes (default: cosine)")
parser.add_argument("--max-rows", type=int, default=None,
                    help="Use only the first N rows (for testing)")
parser.add_argument("--output-prefix", default=None,
                    help="Output prefix (default: same as input prefix)")
parser.add_argument("--recluster", action="store_true",
                    help="Skip UMAP — load the saved clustering embedding and re-run "
                         "HDBSCAN only. Use this to tune --hdbscan-min quickly after "
                         "an initial full run. Requires <prefix>_umap_cluster.npy to exist.")
args = parser.parse_args()

out_prefix         = args.output_prefix or args.prefix
embed_path         = args.prefix + "_embeddings.npy"
index_path         = args.prefix + "_index.json"
cluster_embed_path = out_prefix + "_umap_cluster.npy"
coords_path        = out_prefix + "_umap3d.npy"
coords_index_path  = out_prefix + "_umap3d_index.json"

# -- Validate inputs -----------------------------------------------------------

if args.recluster:
    for path in [index_path, cluster_embed_path, coords_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            if path == cluster_embed_path:
                print(f"  Run without --recluster first to generate the cached embedding.")
            sys.exit(1)
else:
    for path in [embed_path, index_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

# -- Load index ----------------------------------------------------------------

print(f"Loading index from {index_path}...")
with open(index_path, "r") as f:
    index_data = json.load(f)
n_total = len(index_data)

if args.max_rows and args.max_rows < n_total:
    n_use = args.max_rows
    print(f"  Using first {n_use:,} rows (--max-rows)")
    index_data = index_data[:n_use]
else:
    n_use = n_total

ids = [entry["id"] for entry in index_data]

# -- UMAP pass 1 OR load cached clustering embedding --------------------------

cluster_umap_elapsed = 0.0

if args.recluster:
    print(f"\nLoading cached clustering embedding from {cluster_embed_path}...")
    t0 = time.time()
    cluster_embedding = np.load(cluster_embed_path)
    if len(cluster_embedding) != n_use:
        print(f"ERROR: Cached embedding has {len(cluster_embedding):,} rows but "
              f"index has {n_use:,}. Re-run without --recluster.")
        sys.exit(1)
    print(f"  Loaded {cluster_embedding.shape} in {time.time()-t0:.1f}s")

else:
    # Load full embeddings
    print(f"\nLoading embeddings from {embed_path}...")
    t0 = time.time()
    embeddings = np.load(embed_path, mmap_mode="r")
    n_embed, n_dim = embeddings.shape
    print(f"  Shape: {n_embed:,} x {n_dim}")
    assert n_embed == n_total, f"Embedding count ({n_embed}) != index count ({n_total})"

    if n_use < n_embed:
        embeddings = np.array(embeddings[:n_use])
    else:
        print(f"  Loading full array into memory...")
        embeddings = np.array(embeddings)

    print(f"  Loaded in {time.time()-t0:.1f}s")

    if n_use > 500_000:
        print(f"\n  NOTE: {n_use:,} points on CPU — this will take a while.")

    # UMAP pass 1: clustering embedding
    # Per the UMAP docs on clustering:
    #   - Use LARGER n_neighbors (broader structure, less noise-driven clusters)
    #   - Set min_dist=0.0 (pack points densely, which is what HDBSCAN wants)
    #   - Use MORE components so HDBSCAN has richer input than just 3D
    import umap as umap_learn

    print(f"\n--- Pass 1: Clustering embedding ---")
    print(f"  {n_dim}D -> UMAP -> {args.cluster_components}D "
          f"(neighbors={args.cluster_neighbors}, min_dist=0.0, metric={args.metric})")
    t0 = time.time()

    cluster_reducer = umap_learn.UMAP(
        n_components=args.cluster_components,
        n_neighbors=min(args.cluster_neighbors, n_use - 1),
        min_dist=0.0,
        metric=args.metric,
        low_memory=True,
        verbose=True,
        n_jobs=-1,
    )
    cluster_embedding = cluster_reducer.fit_transform(embeddings)
    cluster_umap_elapsed = time.time() - t0
    print(f"  Clustering embedding complete in {cluster_umap_elapsed/60:.1f} min")

    # Save clustering embedding so --recluster can skip this step later
    print(f"  Saving clustering embedding to {cluster_embed_path}...")
    np.save(cluster_embed_path, cluster_embedding.astype(np.float32))
    print(f"  Saved ({os.path.getsize(cluster_embed_path)/1e6:.1f} MB)")

# -- HDBSCAN on the clustering embedding --------------------------------------

import hdbscan

if args.hdbscan_min:
    min_cluster = args.hdbscan_min
else:
    min_cluster = max(10, min(500, n_use // 500))

if args.hdbscan_samples:
    min_samples = args.hdbscan_samples
else:
    min_samples = max(5, min(50, min_cluster // 5))

print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster}, "
      f"min_samples={min_samples}, method={args.hdbscan_method})...")
t0 = time.time()

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster,
    min_samples=min_samples,
    metric="euclidean",
    cluster_selection_method=args.hdbscan_method,
    core_dist_n_jobs=-1,
)
cluster_labels = clusterer.fit_predict(cluster_embedding)

n_clusters  = len(set(cluster_labels) - {-1})
n_noise     = int((cluster_labels == -1).sum())
n_clustered = n_use - n_noise
cluster_elapsed = time.time() - t0

print(f"  Found {n_clusters:,} clusters")
print(f"  Clustered: {n_clustered:,} ({n_clustered*100/n_use:.1f}%)")
print(f"  Noise:     {n_noise:,} ({n_noise*100/n_use:.1f}%)")
print(f"  Clustering complete in {cluster_elapsed:.1f}s")

# -- UMAP pass 2: 3D visualization embedding ----------------------------------

viz_umap_elapsed = 0.0

if args.recluster:
    print(f"\nSkipping visualization UMAP (--recluster) — reusing {coords_path}")
else:
    import umap as umap_learn  # already imported above, but guard for clarity

    print(f"\n--- Pass 2: Visualization embedding ---")
    print(f"  {n_dim}D -> UMAP -> 3D "
          f"(neighbors={args.viz_neighbors}, min_dist={args.viz_min_dist}, metric={args.metric})")
    t0 = time.time()

    viz_reducer = umap_learn.UMAP(
        n_components=3,
        n_neighbors=min(args.viz_neighbors, n_use - 1),
        min_dist=args.viz_min_dist,
        metric=args.metric,
        low_memory=True,
        verbose=True,
        n_jobs=-1,
    )
    coords = viz_reducer.fit_transform(embeddings)
    coords = np.asarray(coords, dtype=np.float32)
    viz_umap_elapsed = time.time() - t0
    print(f"  Visualization embedding complete in {viz_umap_elapsed/60:.1f} min")

    print(f"\nWriting coords to {coords_path}...")
    np.save(coords_path, coords)
    print(f"  Saved -> {coords_path}  ({os.path.getsize(coords_path)/1e6:.1f} MB)")

# -- Write index ---------------------------------------------------------------

print(f"Writing index to {coords_index_path}...")
coords_index = [
    {"id": aid, "index": i, "cluster": int(cluster_labels[i])}
    for i, aid in enumerate(ids)
]
with open(coords_index_path, "w", encoding="utf-8") as f:
    json.dump(coords_index, f, separators=(",", ":"))
print(f"  Saved -> {coords_index_path}  ({os.path.getsize(coords_index_path)/1e6:.1f} MB)")

# -- Summary -------------------------------------------------------------------

total_elapsed = cluster_umap_elapsed + cluster_elapsed + viz_umap_elapsed

print(f"\n{'='*60}")
print(f"  Points          : {n_use:,}")
print(f"  Clusters        : {n_clusters:,} ({n_noise:,} noise)")
if args.recluster:
    print(f"  Cluster UMAP    : skipped (--recluster)")
else:
    print(f"  Cluster UMAP    : {cluster_umap_elapsed/60:.1f} min "
          f"({n_dim}D -> {args.cluster_components}D)")
print(f"  HDBSCAN         : {cluster_elapsed:.1f}s "
      f"(min_cluster={min_cluster}, min_samples={min_samples}, method={args.hdbscan_method})")
if args.recluster:
    print(f"  Viz UMAP        : skipped (--recluster)")
else:
    print(f"  Viz UMAP        : {viz_umap_elapsed/60:.1f} min "
          f"({n_dim}D -> 3D)")
print(f"  Total time      : {total_elapsed/60:.1f} min")
print(f"  Cluster embed   : {cluster_embed_path}")
print(f"  Coords          : {coords_path}")
print(f"  Index           : {coords_index_path}")
print(f"{'='*60}")

if not args.recluster:
    print(f"\nTo tune HDBSCAN without re-running UMAP:")
    print(f"  uv run project_and_cluster.py {args.prefix} --recluster --hdbscan-min <N>")
