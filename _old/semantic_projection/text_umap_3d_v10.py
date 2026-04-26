# -*- coding: utf-8 -*-
"""
Text -> Embeddings -> UMAP 3D -> Interactive Plotly visualization
with automatic cluster discovery via HDBSCAN, LLM-generated cluster titles and
meta-group titles via Ollama, and a collapsible sidebar legend.

Designed for large CSVs (~100MB-1GB+). All expensive outputs are cached to disk
so re-runs skip straight to plotting.

Usage:
    python text_umap_3d.py data.csv --text-col review_text
    python text_umap_3d.py data.csv --text-col body --label-col category
    python text_umap_3d.py data.csv --text-col content --max-rows 50000
    python text_umap_3d.py data.csv --text-col content --no-cache

Cache files produced (all named after your CSV):
    data_embeddings.npy      - sentence embeddings
    data_coords.npy          - UMAP 3D coordinates
    data_cluster_names.json  - cluster id -> title
    data_groups.json         - meta-group id -> {name, cluster_ids}

Requirements:
    pip install sentence-transformers hdbscan umap-learn plotly scikit-learn pandas requests

Ollama setup:
    1. Install Ollama: https://ollama.com
    2. ollama pull llama3.2
    3. Ollama runs automatically in the background
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import requests
import json
import random
import textwrap
from collections import defaultdict


# -- Config --------------------------------------------------------------------

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
PREVIEW_CHARS    = 300
HOVER_LINE_WIDTH = 60
UMAP_NEIGHBORS   = 15
UMAP_MIN_DIST    = 0.1
HDBSCAN_MIN      = 5
BATCH_SIZE       = 256
CHUNK_SIZE       = 10_000

OLLAMA_MODEL  = "llama3.2"
OLLAMA_URL    = "http://localhost:11434/api/generate"
LABEL_SAMPLES = 10


# -- Args ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("csv",             help="Path to CSV file")
parser.add_argument("--text-col",      required=True)
parser.add_argument("--label-col",     default=None)
parser.add_argument("--rating-col",    default=None, help="Numeric column for rating (e.g. 0-5 stars)")
parser.add_argument("--max-rows",      type=int, default=None)
parser.add_argument("--no-cache",      action="store_true")
parser.add_argument("--output-mode",   default="chunked", choices=["chunked", "single"])
args = parser.parse_args()

base         = args.csv.replace(".csv", "")
embed_cache  = base + "_embeddings.npy"
coords_cache = base + "_coords.npy"
names_cache  = base + "_cluster_names.json"
groups_cache = base + "_groups.json"

if args.no_cache:
    for f in [embed_cache, coords_cache, names_cache, groups_cache]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed cache: {f}")


# -- Load CSV ------------------------------------------------------------------

print(f"Reading '{args.text_col}' from {args.csv}...")
cols = [args.text_col]
if args.label_col:  cols.append(args.label_col)
if args.rating_col: cols.append(args.rating_col)

chunks, total = [], 0
for chunk in pd.read_csv(args.csv, usecols=cols, chunksize=CHUNK_SIZE, dtype=str):
    chunk = chunk.dropna(subset=[args.text_col])
    chunk[args.text_col] = chunk[args.text_col].str.strip()
    chunk = chunk[chunk[args.text_col] != ""]
    chunks.append(chunk)
    total += len(chunk)
    print(f"  Loaded {total:,} rows...", end="\r")
    if args.max_rows and total >= args.max_rows:
        break

df = pd.concat(chunks, ignore_index=True)
if args.max_rows:
    df = df.head(args.max_rows)

texts  = df[args.text_col].tolist()
labels = df[args.label_col].tolist() if args.label_col else None

# Ratings: normalise to [0,1]; missing values -> -1 (treated as "no rating")
if args.rating_col and args.rating_col in df.columns:
    raw = pd.to_numeric(df[args.rating_col], errors='coerce')
    rmin, rmax = raw.min(), raw.max()
    span = (rmax - rmin) or 1.0
    ratings = [(float((v - rmin) / span) if not pd.isna(v) else -1.0) for v in raw]
    print(f"\nLoaded {len(texts):,} rows. Ratings: {rmin:.1f}-{rmax:.1f} ({raw.isna().sum():,} missing)")
else:
    ratings = [-1.0] * len(texts)  # no rating data
    print(f"\nLoaded {len(texts):,} rows.")


# -- Embed ---------------------------------------------------------------------

if os.path.exists(embed_cache):
    print(f"\nLoading cached embeddings from {embed_cache}")
    embeddings = np.load(embed_cache)
    if len(embeddings) != len(texts):
        print("  Cache size mismatch - re-embedding.")
        os.remove(embed_cache)

if not os.path.exists(embed_cache):
    print(f"\nEmbedding {len(texts):,} texts with '{EMBEDDING_MODEL}'...")
    model      = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    print(f"  Using device: {model.device}")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    np.save(embed_cache, embeddings)
    print(f"  Saved -> {embed_cache}")

print(f"Embeddings: {embeddings.shape}")


# -- UMAP ----------------------------------------------------------------------

if os.path.exists(coords_cache):
    print(f"\nLoading cached UMAP coords from {coords_cache}")
    coords = np.load(coords_cache)
    if len(coords) != len(texts):
        print("  Cache size mismatch - re-running UMAP.")
        os.remove(coords_cache)

if not os.path.exists(coords_cache):
    print("\nRunning UMAP...")
    n = len(texts)
    reducer = umap.UMAP(
        n_components=3, n_neighbors=min(UMAP_NEIGHBORS, n - 1),
        min_dist=UMAP_MIN_DIST, metric="cosine",
        random_state=42, low_memory=True, verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    np.save(coords_cache, coords)
    print(f"  Saved -> {coords_cache}")


# -- Cluster -------------------------------------------------------------------

print("\nClustering with HDBSCAN...")
n = len(texts)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min(HDBSCAN_MIN, max(2, n // 50)),
    metric="euclidean", core_dist_n_jobs=-1,
)
cluster_labels  = clusterer.fit_predict(coords)
unique_clusters = sorted(c for c in set(cluster_labels) if c >= 0)
n_clusters      = len(unique_clusters)
n_noise         = (cluster_labels == -1).sum()
print(f"Found {n_clusters} clusters, {n_noise:,} unclustered points.")


# -- Ollama helpers ------------------------------------------------------------

def ask_ollama(prompt, fallback):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip().strip('"').strip("'")
        return result or fallback
    except requests.exceptions.ConnectionError:
        print(f"\n  Warning: Could not connect to Ollama. Using fallback labels.")
        return None
    except Exception as e:
        print(f"\n  Warning: Ollama error: {e}")
        return fallback


# -- Cluster titles ------------------------------------------------------------

def load_names_cache():
    try:
        with open(names_cache) as f:
            return {int(k): v for k, v in json.load(f).items()}
    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        if os.path.exists(names_cache):
            print("  Cluster names cache is malformed - deleting.")
            os.remove(names_cache)
        return None

cluster_names = load_names_cache()

if cluster_names is None:
    print("\nGenerating cluster titles with Ollama...")
    cluster_names = {}
    ollama_ok = True
    for lbl in unique_clusters:
        idxs    = [i for i, c in enumerate(cluster_labels) if c == lbl]
        samples = [texts[i] for i in random.sample(idxs, min(LABEL_SAMPLES, len(idxs)))]
        numbered = "\n".join(f"{i+1}. {t[:400]}" for i, t in enumerate(samples))
        prompt = (
            f"Below are {len(samples)} text samples that belong to the same cluster.\n\n"
            f"{numbered}\n\n"
            "Based only on these texts, respond with a single short category title (3-6 words) "
            "that best describes what this group is about. "
            "Reply with ONLY the title, no explanation, no punctuation, no quotes."
        )
        print(f"  Cluster {lbl} ({len(idxs):,} pts)...", end=" ", flush=True)
        title = ask_ollama(prompt, f"Cluster {lbl}")
        if title is None:
            ollama_ok = False
            break
        cluster_names[int(lbl)] = title
        print(title)

    if not ollama_ok:
        cluster_names = {int(lbl): f"Cluster {lbl}" for lbl in unique_clusters}

    with open(names_cache, "w") as f:
        json.dump({int(k): v for k, v in cluster_names.items()}, f, indent=2)
    print(f"  Saved -> {names_cache}")


# -- Meta-groups ---------------------------------------------------------------

def load_groups_cache():
    try:
        with open(groups_cache) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError, FileNotFoundError):
        if os.path.exists(groups_cache):
            print("  Groups cache is malformed - deleting.")
            os.remove(groups_cache)
        return None

meta_groups = load_groups_cache()

if meta_groups is None:
    print("\nEmbedding cluster titles for meta-grouping...")
    title_list   = [cluster_names.get(c, f"Cluster {c}") for c in unique_clusters]
    title_model  = SentenceTransformer(EMBEDDING_MODEL)
    title_embeds = title_model.encode(title_list, show_progress_bar=False,
                                      convert_to_numpy=True, normalize_embeddings=True)

    n_titles = len(title_list)
    meta_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min(5, n_titles // 10)),
        metric="euclidean",
    )
    meta_labels = meta_clusterer.fit_predict(title_embeds)

    group_to_clusters = defaultdict(list)
    for cluster_id, group_id in zip(unique_clusters, meta_labels):
        group_to_clusters[int(group_id)].append(int(cluster_id))

    n_meta = len([g for g in group_to_clusters if g >= 0])
    print(f"Found {n_meta} meta-groups, "
          f"{len(group_to_clusters.get(-1, []))} ungrouped clusters.")

    print("\nGenerating group titles with Ollama...")
    meta_groups  = {}
    ollama_ok    = True

    for gid, cids in sorted(group_to_clusters.items()):
        if gid == -1:
            meta_groups[-1] = {"name": "Ungrouped", "cluster_ids": cids}
            continue
        ctitles  = [cluster_names.get(c, f"Cluster {c}") for c in cids]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ctitles))
        prompt = (
            f"Below are {len(ctitles)} category titles that belong to the same group.\n\n"
            f"{numbered}\n\n"
            "Based only on these titles, respond with a single short umbrella title (2-5 words) "
            "that best describes what this group has in common. "
            "Reply with ONLY the title, no explanation, no punctuation, no quotes."
        )
        print(f"  Group {gid} ({len(cids)} clusters)...", end=" ", flush=True)
        name = ask_ollama(prompt, f"Group {gid}")
        if name is None:
            ollama_ok = False
            break
        meta_groups[gid] = {"name": name, "cluster_ids": cids}
        print(name)

    if not ollama_ok:
        for gid, cids in group_to_clusters.items():
            meta_groups[gid] = {
                "name": "Ungrouped" if gid == -1 else f"Group {gid}",
                "cluster_ids": cids,
            }

    # -- Merge groups whose Ollama-generated names are too similar --------------
    # Embed the group names, find pairs with cosine similarity > threshold,
    # then union-find merge them and ask Ollama for a new combined name.
    GROUP_MERGE_THRESHOLD = 0.82   # tweak: lower = more merging

    named_gids = [gid for gid in meta_groups if gid >= 0]
    if len(named_gids) > 1:
        print("\nChecking for redundant groups by name similarity...")
        gname_list   = [meta_groups[gid]["name"] for gid in named_gids]
        gname_embeds = title_model.encode(gname_list, show_progress_bar=False,
                                          convert_to_numpy=True, normalize_embeddings=True)
        # Cosine similarity matrix (embeddings are already normalised -> dot product)
        sim_matrix = gname_embeds @ gname_embeds.T

        # Union-Find
        parent = list(range(len(named_gids)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            parent[find(a)] = find(b)

        merges = 0
        for i in range(len(named_gids)):
            for j in range(i + 1, len(named_gids)):
                if sim_matrix[i, j] >= GROUP_MERGE_THRESHOLD:
                    union(i, j)
                    merges += 1

        if merges:
            # Collect merged groups
            merged_map = defaultdict(list)   # root_idx -> list of original gids
            for idx, gid in enumerate(named_gids):
                merged_map[find(idx)].append(gid)

            new_meta_groups = {}
            new_gid = 0
            for root_idx, gids_to_merge in sorted(merged_map.items()):
                all_cids = []
                all_ctitles = []
                for gid in gids_to_merge:
                    all_cids.extend(meta_groups[gid]["cluster_ids"])
                    all_ctitles.extend(
                        cluster_names.get(c, f"Cluster {c}")
                        for c in meta_groups[gid]["cluster_ids"]
                    )
                if len(gids_to_merge) == 1:
                    # No merge happened -- keep original name
                    new_meta_groups[new_gid] = {
                        "name": meta_groups[gids_to_merge[0]]["name"],
                        "cluster_ids": all_cids,
                    }
                else:
                    # Re-ask Ollama for a combined name
                    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(all_ctitles))
                    prompt = (
                        f"Below are {len(all_ctitles)} category titles that belong to the same group.\n\n"
                        f"{numbered}\n\n"
                        "Based only on these titles, respond with a single short umbrella title (2-5 words) "
                        "that best describes what this group has in common. "
                        "Reply with ONLY the title, no explanation, no punctuation, no quotes."
                    )
                    merged_names = " / ".join(meta_groups[g]["name"] for g in gids_to_merge)
                    print(f"  Merging [{merged_names}]...", end=" ", flush=True)
                    name = ask_ollama(prompt, merged_names.split(" / ")[0])
                    print(name)
                    new_meta_groups[new_gid] = {"name": name or merged_names.split(" / ")[0],
                                                "cluster_ids": all_cids}
                new_gid += 1

            # Preserve ungrouped
            if -1 in meta_groups:
                new_meta_groups[-1] = meta_groups[-1]

            n_before = len(named_gids)
            n_after  = len(new_meta_groups) - (1 if -1 in new_meta_groups else 0)
            print(f"  Merged {n_before} groups -> {n_after} groups.")
            meta_groups = new_meta_groups
        else:
            print("  No redundant groups found.")

    with open(groups_cache, "w") as f:
        json.dump({int(k): v for k, v in meta_groups.items()}, f, indent=2)
    print(f"  Saved -> {groups_cache}")


# -- Plot ----------------------------------------------------------------------

print("\nBuilding plot...")

PALETTE = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
    "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    "#17becf","#bcbd22","#7f7f7f","#d62728","#2ca02c",
    "#1f77b4","#ff7f0e","#aec7e8","#ffbb78","#98df8a",
]

def cluster_color(lbl):
    return "rgba(180,180,180,0.25)" if lbl == -1 else PALETTE[lbl % len(PALETTE)]

def format_hover(text):
    truncated = text[:PREVIEW_CHARS]
    ellipsis  = "..." if len(text) > PREVIEW_CHARS else ""
    return "<br>".join(textwrap.wrap(truncated, width=HOVER_LINE_WIDTH)) + ellipsis

point_groups = defaultdict(list)
for i, lbl in enumerate(cluster_labels):
    point_groups[lbl].append(i)

fig = go.Figure()

for lbl, idxs in sorted(point_groups.items()):
    idxs        = np.array(idxs)
    color       = cluster_color(lbl)
    legend_name = cluster_names.get(lbl, f"Cluster {lbl}") if lbl >= 0 else f"Unclustered ({len(idxs):,})"

    hover_parts = []
    for i in idxs:
        header  = f"<b>Row {i}</b>"
        header += f" . {labels[i]}" if labels else ""
        header += f" . <i>{cluster_names.get(lbl, '')}</i>" if lbl >= 0 else ""
        hover_parts.append(f"{header}<br>{format_hover(texts[i])}")

    fig.add_trace(go.Scatter3d(
        x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
        mode="markers", name=legend_name,
        marker=dict(size=3, color=color, opacity=0.75, line=dict(width=0)),
        text=hover_parts,
        hovertemplate="%{text}<extra></extra>",
    ))

fig.update_layout(
    title      = f"3D UMAP - {len(texts):,} rows . {n_clusters} clusters",
    scene      = dict(
        xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3",
        bgcolor="rgb(15,15,25)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showbackground=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showbackground=False),
        zaxis=dict(gridcolor="rgba(255,255,255,0.08)", showbackground=False),
    ),
    paper_bgcolor = "rgb(15,15,25)",
    font_color    = "white",
    showlegend    = False,
    width         = None,
    height        = None,
    margin        = dict(l=0, r=280, t=40, b=0),
)


# -- Build sidebar data --------------------------------------------------------

trace_order = [lbl for lbl, _ in sorted(point_groups.items())]

# Flat cluster list for sidebar — no meta-group nesting
# Unclustered entry first (hidden by default), then all named clusters sorted by trace order
sidebar_clusters = []

if -1 in trace_order:
    tidx = trace_order.index(-1)
    sidebar_clusters.append({
        "traceIdx":    tidx,
        "name":        "Unclustered",
        "color":       "rgba(180,180,180,0.35)",
        "size":        len(point_groups[-1]),
        "unclustered": True,
    })

for lbl, _ in sorted(point_groups.items()):
    if lbl < 0:
        continue
    tidx = trace_order.index(lbl)
    sidebar_clusters.append({
        "traceIdx":    tidx,
        "name":        cluster_names.get(lbl, f"Cluster {lbl}"),
        "color":       cluster_color(lbl),
        "size":        len(point_groups[lbl]),
        "unclustered": False,
    })

sidebar_js     = json.dumps(sidebar_clusters)
trace_names_js = json.dumps([
    cluster_names.get(lbl, f"Cluster {lbl}") if lbl >= 0 else "Unclustered"
    for lbl, _ in sorted(point_groups.items())
])



# -- Colour helpers ------------------------------------------------------------

import re as _re

def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def color_to_rgb(c):
    if c.startswith('#'):
        return hex_to_rgb(c)
    nums = [float(x) for x in _re.findall(r'[\d.]+', c)]
    return [int(nums[0]), int(nums[1]), int(nums[2])]

# -- Build per-point arrays ----------------------------------------------------

print("Building point arrays...")

all_x, all_y, all_z = [], [], []
all_r, all_g, all_b = [], [], []
all_trace, all_hover, all_rating = [], [], []

for lbl, idxs in sorted(point_groups.items()):
    idxs  = list(idxs)
    tidx  = trace_order.index(lbl)
    color = cluster_color(lbl)
    rgb   = color_to_rgb(color)

    for i in idxs:
        all_x.append(float(coords[i, 0]))
        all_y.append(float(coords[i, 1]))
        all_z.append(float(coords[i, 2]))
        all_r.append(rgb[0])
        all_g.append(rgb[1])
        all_b.append(rgb[2])
        all_trace.append(tidx)
        all_rating.append(float(ratings[i]))

        cluster_label = cluster_names.get(lbl, '') if lbl >= 0 else 'Unclustered'
        label_part    = f"<span class='ht-label'>{labels[i]}</span>" if labels else ""
        rating_part   = ""
        if ratings[i] >= 0 and args.rating_col and args.rating_col in df.columns:
            raw_val  = df[args.rating_col].iloc[i]
            try:
                stars_val = float(raw_val)
                full      = int(round(stars_val))
                full      = max(0, min(5, full))
                stars_str = "★" * full + "☆" * (5 - full)
                rating_part = f"<span class='ht-rating'>{stars_str}</span>"
            except (ValueError, TypeError):
                rating_part = f"<span class='ht-rating'>{raw_val}</span>"

        preview = format_hover(texts[i])

        hover_html = (
            f"<div class='ht-head'>"
            f"<span class='ht-row'>Row {i}</span>"
            f"<span class='ht-sep'></span>"
            f"<span class='ht-cluster'>{cluster_label}</span>"
            f"{label_part}"
            f"{rating_part}"
            f"</div>"
            f"<div class='ht-body'>{preview}</div>"
        )
        all_hover.append(hover_html)

N_POINTS = len(all_x)
print(f"  {N_POINTS:,} points ready.")

# Normalise to [-1, 1]
cx = (max(all_x) + min(all_x)) / 2
cy = (max(all_y) + min(all_y)) / 2
cz = (max(all_z) + min(all_z)) / 2
scale = max(
    max(all_x) - min(all_x),
    max(all_y) - min(all_y),
    max(all_z) - min(all_z),
) / 2 or 1.0

all_x = [(v - cx) / scale for v in all_x]
all_y = [(v - cy) / scale for v in all_y]
all_z = [(v - cz) / scale for v in all_z]


# -- UI: sidebar CSS + HTML ----------------------------------------------------

UI_STYLE = """
<style>
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%;
    overflow: hidden; background: rgb(15,15,25); font-family: sans-serif;
  }
  #three-canvas {
    position: fixed; left: 0; top: 0; display: block;
  }
  #page-title {
    position: fixed; bottom: 18px;
    /* centre within the plot area (full width minus 270px sidebar) */
    left: calc((100vw - 270px) / 2);
    transform: translateX(-50%);
    font-size: 15px; font-weight: 700; color: rgba(255,255,255,0.92);
    white-space: nowrap; z-index: 495; font-family: sans-serif;
    letter-spacing: 0.02em; pointer-events: none;
    background: rgba(15,15,28,0.82); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 6px 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.5);
  }
  #sidebar-search-row {
    position: relative; display: flex; align-items: center; margin-bottom: 6px;
  }
  #sidebar-search {
    flex: 1; padding: 6px 28px 6px 10px; margin-bottom: 0;
  }
  #sidebar-search-clear {
    position: absolute; right: 6px; background: none; border: none;
    color: rgba(255,255,255,0.35); font-size: 12px; cursor: pointer;
    padding: 0; line-height: 1; display: none;
  }
  #sidebar-search-clear:hover { color: rgba(255,255,255,0.8); }
  #sort-btns {
    display: flex; gap: 4px; padding: 6px 12px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06); flex-shrink: 0;
  }
  .sort-btn {
    flex: 1; padding: 4px 4px; border-radius: 4px; font-size: 10px; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.12); background: rgba(30,30,50,0.7);
    color: rgba(255,255,255,0.5); text-align: center; user-select: none;
    transition: all 0.12s;
  }
  .sort-btn:hover { background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.8); }
  .sort-btn.active { background: rgba(78,121,167,0.35); border-color: rgba(78,121,167,0.7); color: rgba(255,255,255,0.9); }
  #sidebar {
    position: fixed; top: 0; right: 0; bottom: 0; width: 270px;
    background: rgba(15,15,28,0.97); border-left: 1px solid rgba(255,255,255,0.08);
    display: flex; flex-direction: column; z-index: 1000; overflow: hidden;
  }
  #sidebar-search-wrap {
    padding: 10px 12px 8px; border-bottom: 1px solid rgba(255,255,255,0.08); flex-shrink: 0;
  }
  #sidebar-search {
    width: 100%; box-sizing: border-box; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.2); background: rgba(30,30,50,0.9);
    color: white; font-size: 12px; outline: none;
  }
  #sidebar-search::placeholder { color: rgba(255,255,255,0.3); }
  #sidebar-search:focus { border-color: rgba(255,255,255,0.45); }
  #sidebar-btns { display: flex; gap: 5px; }
  .sb-btn {
    flex: 1; padding: 5px 2px; border-radius: 5px;
    border: 1px solid rgba(255,255,255,0.18); background: rgba(30,30,50,0.9);
    color: rgba(255,255,255,0.75); font-size: 11px; cursor: pointer;
    text-align: center; user-select: none;
  }
  .sb-btn:hover { background: rgba(255,255,255,0.1); }
  #sidebar-stats {
    padding: 4px 12px 3px; font-size: 10px; color: rgba(255,255,255,0.35);
    border-bottom: 1px solid rgba(255,255,255,0.05); flex-shrink: 0;
    display: flex; gap: 10px; flex-wrap: wrap;
  }
  #sidebar-stats span { white-space: nowrap; }
  #sidebar-tree { flex: 1; overflow-y: auto; padding: 4px 0; }
  #sidebar-tree::-webkit-scrollbar { width: 4px; }
  #sidebar-tree::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }
  .cluster-row {
    display: flex; align-items: center; gap: 6px;
    padding: 3px 10px; cursor: pointer; user-select: none;
  }
  .cluster-row:hover { background: rgba(255,255,255,0.04); }
  .cluster-row.selected { background: rgba(78,121,167,0.18); }
  .cluster-eye {
    font-size: 13px; cursor: pointer; flex-shrink: 0;
    color: rgba(255,255,255,0.75); transition: color 0.15s; line-height: 1;
  }
  .cluster-eye:hover { color: white; }
  .cluster-eye.hidden { color: rgba(255,255,255,0.18); }
  .cluster-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; opacity: 0.85; }
  .cluster-label {
    flex: 1; min-width: 0; display: flex; align-items: baseline;
    gap: 6px; overflow: hidden;
  }
  .cluster-name {
    font-size: 13px; color: rgba(255,255,255,0.72);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;
  }
  .cluster-name.highlighted { color: #f28e2b; font-weight: 600; }
  .cluster-size { font-size: 11px; color: rgba(255,255,255,0.28); flex-shrink: 0; }
  .match-hl { color: #f28e2b; font-weight: 700; }
  #hover-tip {
    position: fixed; pointer-events: none; z-index: 490;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 6px; font-size: 12px;
    max-width: 380px; display: none;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    overflow: visible; font-family: sans-serif;
  }
  #hover-tip-arrow {
    position: absolute; top: 50%; z-index: 0;
    width: 11px; height: 11px;
    transform: translateY(-50%) rotate(45deg);
    border: 1px solid rgba(255,255,255,0.15);
  }
  #hover-tip-inner {
    border-radius: 6px;
    overflow: hidden;
    position: relative; z-index: 1;
  }
  .ht-head {
    display: flex; align-items: center; gap: 0;
    padding: 6px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    flex-wrap: nowrap; overflow: hidden;
    background: rgba(40,40,70,0.95);
  }
  .ht-row {
    font-weight: 700; font-size: 13px;
    white-space: nowrap; flex-shrink: 0;
  }
  .ht-sep {
    width: 1px; height: 12px; background: rgba(255,255,255,0.25);
    margin: 0 8px; flex-shrink: 0;
  }
  .ht-cluster {
    font-size: 11px; font-style: italic;
    opacity: 0.75; flex-shrink: 1; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .ht-label {
    font-size: 11px; opacity: 0.6;
    white-space: nowrap; flex-shrink: 0;
    margin-left: 8px;
  }
  .ht-rating {
    font-size: 11px; font-weight: 600;
    white-space: nowrap; flex-shrink: 0;
    margin-left: auto; padding-left: 10px;
  }
  .ht-body {
    padding: 7px 10px; font-size: 12px;
    line-height: 1.55; word-wrap: break-word; white-space: normal;
    background: rgba(10,10,20,0.97); color: rgba(255,255,255,0.88);
  }
  #progress-bar-wrap {
    position: fixed; bottom: 0; left: 0; right: 270px;
    height: 3px; background: rgba(255,255,255,0.08); z-index: 9999; display: none;
  }
  #progress-bar { height: 100%; width: 0%; background: #4e79a7; transition: width 0.15s; }
  #progress-label {
    position: fixed; bottom: 8px; left: 12px; font-size: 11px;
    color: rgba(255,255,255,0.4); z-index: 9999; display: none;
  }
  /* -- Floating controls -- */
  #float-controls {
    position: fixed; top: 12px; left: 12px; z-index: 500;
    display: flex; flex-direction: column; gap: 6px;
    user-select: none;
  }
  .fc-group {
    background: rgba(15,15,28,0.88); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 7px 9px;
    display: flex; flex-direction: column; gap: 5px;
    backdrop-filter: blur(6px);
  }
  .fc-row { display: flex; gap: 5px; align-items: center; }
  .fc-btn {
    padding: 4px 9px; border-radius: 5px; font-size: 10px; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.8); white-space: nowrap; transition: background 0.12s;
  }
  .fc-btn:hover { background: rgba(255,255,255,0.14); }
  .fc-btn.active { background: rgba(78,121,167,0.5); border-color: rgba(78,121,167,0.8); color: white; }
  .fc-label { font-size: 10px; color: rgba(255,255,255,0.4); min-width: 58px; }
  .fc-slider {
    -webkit-appearance: none; appearance: none;
    width: 90px; height: 3px; border-radius: 2px;
    background: rgba(255,255,255,0.18); outline: none; cursor: pointer;
  }
  .fc-slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 11px; height: 11px;
    border-radius: 50%; background: rgba(255,255,255,0.85); cursor: pointer;
  }
  .rslider {
    position: relative; height: 28px; margin: 2px 0;
    user-select: none; overflow: visible;
  }
  .rslider-track {
    position: absolute; left: 0; right: 0;
    top: calc(50% - 1.5px); height: 3px;
    background: rgba(255,255,255,0.15); border-radius: 2px;
    pointer-events: none;
  }
  .rslider-fill {
    /* Tall transparent hit zone; visible bar via ::after */
    position: absolute;
    top: calc(50% - 11px); height: 22px;
    background: rgba(255,255,255,0.001); /* near-zero so pointer events fire */
    cursor: grab; z-index: 1; pointer-events: all;
  }
  .rslider-fill:active { cursor: grabbing; }
  .rslider-fill::after {
    content: ''; position: absolute;
    left: 0; right: 0; top: calc(50% - 1.5px); height: 3px;
    background: rgba(78,121,167,0.9); border-radius: 2px;
    pointer-events: none;
  }
  .rslider-thumb {
    position: absolute; top: calc(50% - 9px);
    width: 9px; height: 18px; box-sizing: border-box;
    background: rgba(255,255,255,0.88);
    border: 2px solid rgba(78,121,167,0.9);
    cursor: grab; z-index: 3;
  }
  .rslider-thumb:active { cursor: grabbing; }
  .rslider-thumb-lo {
    border-radius: 10px 0 0 10px; border-right: none;
  }
  .rslider-thumb-hi {
    border-radius: 0 10px 10px 0; border-left: none;
  }
  .fc-toggle-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .fc-sep { height: 1px; background: rgba(255,255,255,0.08); margin: 3px 0; }
  .fc-val { font-size: 10px; color: rgba(255,255,255,0.55); min-width: 24px; text-align: right; font-family: sans-serif; }
</style>
"""

UI_HTML = """
<div id="hover-tip"><div id="hover-tip-arrow"></div><div id="hover-tip-inner"></div></div>
<div id="float-controls">
  <div class="fc-group">
    <div class="fc-row" style="margin-bottom:3px;">
      <span class="fc-label" style="min-width:unset;color:rgba(255,255,255,0.55);font-size:10px;letter-spacing:0.05em;">CAMERA</span>
    </div>
    <div class="fc-row" style="gap:4px;">
      <div class="fc-btn" id="fc-reset"  title="Reset camera">&#8635; Reset</div>
      <div class="fc-btn" id="fc-top"    title="Top view (X/Y)">Top</div>
      <div class="fc-btn" id="fc-front"  title="Front view (X/Z)">Front</div>
      <div class="fc-btn" id="fc-side"   title="Side view (Y/Z)">Side</div>
    </div>
    <div class="fc-row" style="font-size:9px;color:rgba(255,255,255,0.28);padding-top:2px;flex-direction:column;gap:1px;align-items:flex-start;">
      <span>Drag to rotate &nbsp;&middot;&nbsp; Scroll to zoom</span>
      <span>Shift+drag to pan</span>
    </div>
  </div>
  <div class="fc-group">
    <div class="fc-row" style="margin-bottom:3px;">
      <span class="fc-label" style="min-width:unset;color:rgba(255,255,255,0.55);font-size:10px;letter-spacing:0.05em;">DISPLAY</span>
    </div>
    <div class="fc-row">
      <span class="fc-label">Point size</span>
      <input class="fc-slider" id="fc-size" type="range" min="1" max="8" value="4" step="0.5" style="flex:1;">
      <span class="fc-val" id="fc-size-val">4</span>
    </div>
    <div class="fc-sep"></div>
    <div class="fc-row"><span class="fc-label">Shading</span></div>
    <div class="fc-toggle-row">
      <div class="fc-btn active" id="fc-outline">Outline</div>
      <div class="fc-btn" id="fc-rating">Rating &#9733;</div>
    </div>
  </div>
  <div class="fc-group" id="fc-rating-panel">
    <div class="fc-row" style="margin-bottom:5px;">
      <span class="fc-label" style="min-width:unset;color:rgba(255,255,255,0.55);font-size:10px;letter-spacing:0.05em;">RATING FILTER</span>
      <span style="margin-left:auto;font-size:10px;color:rgba(255,255,255,0.5);"><span id="fc-rmin-val">0</span> &ndash; <span id="fc-rmax-val">5</span> &#9733;</span>
    </div>
    <div class="rslider" id="rslider">
      <div class="rslider-track">
        <div class="rslider-fill" id="rslider-fill"></div>
      </div>
      <div class="rslider-thumb rslider-thumb-lo" id="rslider-lo"></div>
      <div class="rslider-thumb rslider-thumb-hi" id="rslider-hi"></div>
    </div>
  </div>
</div>
<div id="sidebar">
  <div id="sidebar-search-wrap">
    <div id="sidebar-search-row">
      <input id="sidebar-search" type="text" placeholder="Search clusters..." autocomplete="off" />
      <button id="sidebar-search-clear" title="Clear search">&#10005;</button>
    </div>
    <div id="sidebar-btns">
      <div class="sb-btn" id="sb-show-all">Show all</div>
      <div class="sb-btn" id="sb-focus">Isolate</div>
      <div class="sb-btn" id="sb-clear-hl">Clear</div>
    </div>
  </div>
  <div id="sort-btns">
    <div class="sort-btn active" id="sort-default" title="Default order">Default</div>
    <div class="sort-btn" id="sort-alpha"   title="Sort A to Z">A&#8594;Z</div>
    <div class="sort-btn" id="sort-size"    title="Sort by point count">&#8595; Size</div>
  </div>
  <div id="sidebar-stats"></div>
  <div id="sidebar-tree"></div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar"></div></div>
<div id="progress-label"></div>
"""


def ui_script(trace_names_js, sidebar_js="[]", auto_init=True):
    init_call = "initSidebar();" if auto_init else ""
    return f"""
<script>
function initSidebar() {{
  const TRACE_NAMES   = {trace_names_js};
  // SIDEBAR_DATA is a flat array of cluster objects:
  // {{ traceIdx, name, color, size, unclustered }}
  const SIDEBAR_DATA  = {sidebar_js};

  // -- Visibility / highlight state ------------------------------------------
  const traceVisible = new Array(TRACE_NAMES.length).fill(true);
  let highlightedSet = null;
  // Unclustered hidden by default -- set before buildTree so eye icons render correctly
  SIDEBAR_DATA.forEach(c => {{ if (c.unclustered) traceVisible[c.traceIdx] = false; }});

  function redraw() {{
    if (window.threeRedraw) window.threeRedraw(traceVisible, highlightedSet);
    updateStats();
  }}

  function updateStats(ratingLo, ratingHi) {{
    const el = document.getElementById('sidebar-stats');
    if (!el) return;
    const hasRatingFilter = (ratingLo !== undefined && (ratingLo > 0 || ratingHi < 1));
    const totalClusters = SIDEBAR_DATA.filter(c => !c.unclustered).length;
    const totalPts      = SIDEBAR_DATA.filter(c => !c.unclustered).reduce((s,c) => s+c.size, 0);
    const visClusters   = SIDEBAR_DATA.filter(c => !c.unclustered && traceVisible[c.traceIdx]).length;
    const visPts        = SIDEBAR_DATA.filter(c => !c.unclustered && traceVisible[c.traceIdx]).reduce((s,c) => s+c.size, 0);
    const selCount      = highlightedSet ? highlightedSet.size : 0;
    const fmt = n => n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n);
    let ptStr, clStr;
    if (hasRatingFilter && window._getRatingFilteredCount && window._getFilteredClusterCount) {{
      const filteredPts  = window._getRatingFilteredCount(ratingLo, ratingHi, traceVisible);
      const filteredClus = window._getFilteredClusterCount(ratingLo, ratingHi, traceVisible);
      ptStr = fmt(filteredPts)+' / '+fmt(visPts)+' pts';
      clStr = filteredClus+' / '+visClusters+' clusters';
    }} else {{
      ptStr = fmt(visPts)+' / '+fmt(totalPts)+' pts';
      clStr = visClusters+' / '+totalClusters+' clusters';
    }}
    let html = '<span>'+ptStr+'</span><span>'+clStr+'</span>';
    if (selCount > 0) html += '<span style="color:rgba(242,142,43,0.85);">'+selCount+' selected</span>';
    el.innerHTML = html;
  }}
  window._updateStats = updateStats;

  function applyHighlight(tidxSet) {{
    highlightedSet = tidxSet;
    document.querySelectorAll('.cl-name').forEach(el => {{
      el.classList.toggle('highlighted', tidxSet && tidxSet.has(parseInt(el.dataset.tidx)));
    }});
    document.querySelectorAll('.cluster-row').forEach(row => {{
      const _cn = row.querySelector('.cl-name'); const tidx = _cn ? parseInt(_cn.dataset.tidx) : NaN;
      row.classList.toggle('selected', tidxSet && tidxSet.has(tidx));
    }});
    redraw();
  }}

  function clearHighlight() {{
    highlightedSet = null;
    document.querySelectorAll('.cl-name').forEach(el => el.classList.remove('highlighted'));
    document.querySelectorAll('.cluster-row').forEach(r => r.classList.remove('selected'));
    redraw();
  }}

  // -- Helpers ---------------------------------------------------------------
  function hlText(text, query) {{
    if (!query) return text;
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) return text;
    return text.slice(0,idx)
      + '<span class="match-hl">' + text.slice(idx, idx+query.length) + '</span>'
      + text.slice(idx+query.length);
  }}

  function fmtSize(n) {{
    return n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n);
  }}

  // -- Sort ------------------------------------------------------------------
  let sortMode = 'default';

  function sorted(data, mode) {{
    if (mode === 'default') return data;
    const copy = data.slice();
    if (mode === 'alpha') copy.sort((a,b) => a.name.localeCompare(b.name));
    if (mode === 'size')  copy.sort((a,b) => b.size - a.size);
    return copy;
  }}

  // -- Build flat cluster list -----------------------------------------------
  const tree = document.getElementById('sidebar-tree');
  let lastClickedIdx = -1;  // index in current sorted/filtered list for range selection

  function buildTree(filter) {{
    tree.innerHTML = '';
    const data = sorted(SIDEBAR_DATA, sortMode);
    let rowIndex = 0;
    data.forEach(c => {{
      if (filter && !c.name.toLowerCase().includes(filter.toLowerCase())) return;

      const row = document.createElement('div');
      const isSelected = highlightedSet && highlightedSet.has(c.traceIdx);
      row.className = 'cluster-row' + (isSelected ? ' selected' : '');
      row.dataset.rowIndex = rowIndex++;
      row.dataset.traceIdx = c.traceIdx;

      const dot = document.createElement('div');
      dot.className = 'cluster-dot';
      dot.style.background = c.color;

      const eye = document.createElement('span');
      eye.className = 'cluster-eye' + (traceVisible[c.traceIdx] ? '' : ' hidden');
      eye.innerHTML = '&#128065;';
      eye.addEventListener('click', e => {{
        e.stopPropagation();
        traceVisible[c.traceIdx] = !traceVisible[c.traceIdx];
        eye.classList.toggle('hidden', !traceVisible[c.traceIdx]);
        redraw();
      }});

      const label = document.createElement('div');
      label.className = 'cluster-label';

      const cname = document.createElement('div');
      cname.className = 'cl-name cluster-name' + (isSelected ? ' highlighted' : '');
      cname.dataset.tidx = c.traceIdx;
      cname.innerHTML = hlText(c.name, filter);
      cname.title = c.name;

      const csize = document.createElement('div');
      csize.className = 'cluster-size';
      csize.textContent = fmtSize(c.size);

      label.appendChild(cname); label.appendChild(csize);

      row.addEventListener('click', e => {{
        if (e.target === eye || eye.contains(e.target)) return;
        e.stopPropagation();
        const ri = parseInt(row.dataset.rowIndex);
        const allRows = [...tree.querySelectorAll('.cluster-row')];

        if (e.shiftKey && lastClickedIdx >= 0) {{
          // Shift+click: select contiguous range
          const lo = Math.min(lastClickedIdx, ri);
          const hi = Math.max(lastClickedIdx, ri);
          const next = new Set(highlightedSet || []);
          allRows.slice(lo, hi+1).forEach(r => next.add(parseInt(r.dataset.traceIdx)));
          applyHighlight(next);
        }} else if (e.ctrlKey || e.metaKey) {{
          // Ctrl/Cmd+click: toggle single without clearing others
          const next = new Set(highlightedSet || []);
          if (next.has(c.traceIdx)) next.delete(c.traceIdx);
          else next.add(c.traceIdx);
          lastClickedIdx = ri;
          if (next.size === 0) clearHighlight();
          else applyHighlight(next);
        }} else {{
          // Normal click: select single (toggle if already sole selection)
          const already = highlightedSet && highlightedSet.size === 1
            && highlightedSet.has(c.traceIdx);
          lastClickedIdx = ri;
          if (already) clearHighlight();
          else applyHighlight(new Set([c.traceIdx]));
        }}
      }});

      row.appendChild(dot); row.appendChild(eye); row.appendChild(label);
      tree.appendChild(row);
    }});
  }}

  // -- Arrow key navigation --------------------------------------------------
  document.getElementById('sidebar-tree').addEventListener('keydown', e => {{
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const rows = [...tree.querySelectorAll('.cluster-row')];
    if (rows.length === 0) return;
    // Find current focused row (sole selection or last clicked)
    let cur = rows.findIndex(r => r.classList.contains('selected'));
    if (cur === -1) cur = (e.key === 'ArrowDown') ? -1 : rows.length;
    const next = e.key === 'ArrowDown'
      ? Math.min(cur + 1, rows.length - 1)
      : Math.max(cur - 1, 0);
    const tidx = parseInt(rows[next].dataset.traceIdx);
    lastClickedIdx = next;
    applyHighlight(new Set([tidx]));
    rows[next].scrollIntoView({{ block: 'nearest' }});
  }});
  document.getElementById('sidebar-tree').setAttribute('tabindex', '0');

  // Expose for use by 3D double-click handler (null = clear)
  window._sidebarHighlight = (tidxSet) => {{
    if (tidxSet === null) clearHighlight();
    else applyHighlight(tidxSet);
  }};

  // Initial render + initial redraw to hide Unclustered
  buildTree('');
  redraw();
  updateStats();

  // -- Sort buttons ----------------------------------------------------------
  ['sort-default','sort-alpha','sort-size'].forEach(id => {{
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.addEventListener('click', () => {{
      sortMode = id.replace('sort-','');
      document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      doSearch(searchInput ? searchInput.value.trim() : '');
    }});
  }});

  // -- Search (filter list + highlight 3D) -----------------------------------
  const searchInput = document.getElementById('sidebar-search');
  const searchClear = document.getElementById('sidebar-search-clear');

  function doSearch(q) {{
    buildTree(q);
    if (searchClear) searchClear.style.display = q ? 'block' : 'none';
    if (q) {{
      const matchSet = new Set();
      SIDEBAR_DATA.forEach(c => {{
        if (c.name.toLowerCase().includes(q.toLowerCase())) matchSet.add(c.traceIdx);
      }});
      if (matchSet.size > 0) applyHighlight(matchSet);
      else clearHighlight();
    }} else {{
      clearHighlight();
    }}
  }}

  if (searchInput) searchInput.addEventListener('input', function() {{
    doSearch(this.value.trim());
  }});
  if (searchClear) searchClear.addEventListener('click', () => {{
    searchInput.value = '';
    doSearch('');
    searchInput.focus();
  }});

  // -- Toolbar buttons -------------------------------------------------------
  document.getElementById('sb-show-all').addEventListener('click', () => {{
    traceVisible.fill(true);
    // Keep Unclustered hidden
    SIDEBAR_DATA.forEach(c => {{ if (c.unclustered) traceVisible[c.traceIdx] = false; }});
    clearHighlight();
    buildTree(document.getElementById('sidebar-search').value.trim());
    redraw();
  }});
  const focusBtn = document.getElementById('sb-focus');
  if (focusBtn) focusBtn.addEventListener('click', () => {{
    if (!highlightedSet || highlightedSet.size === 0) return;
    // Hide everything, show only selected
    traceVisible.fill(false);
    highlightedSet.forEach(t => {{ traceVisible[t] = true; }});
    buildTree(document.getElementById('sidebar-search').value.trim());
    redraw();
  }});
  document.getElementById('sb-clear-hl').addEventListener('click', clearHighlight);
}}
{init_call}
</script>
"""



# -- Three.js shared renderer code ---------------------------------------------

THREE_CDN     = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
ORBIT_CDN     = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"

# Shared JS body used by both single + chunked modes.
# Expects these to already be in scope:
#   posArr, colArr, baseR, baseG, baseB, traceArr, hoverArr, N (total points)
#   loaded (mutable, starts 0 in chunked; equals N in single)
THREEJS_BODY = """
  const W = () => window.innerWidth - 270;
  const H = () => window.innerHeight;

  const canvas = document.getElementById('three-canvas');
  canvas.style.width  = W() + 'px';
  canvas.style.height = H() + 'px';

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W(), H());
  renderer.setClearColor(0x0f0f19, 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, W()/H(), 0.0001, 5000);
  const DEFAULT_POS    = new THREE.Vector3(1.8, 1.4, 2.8);
  const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);
  camera.position.copy(DEFAULT_POS);
  camera.lookAt(DEFAULT_TARGET);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping      = true;
  controls.dampingFactor      = 0.08;
  controls.minDistance        = 0.0;
  controls.maxDistance        = 50;
  controls.screenSpacePanning = true;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN,
  };
  renderer.domElement.addEventListener('mousedown', e => {
    controls.mouseButtons.LEFT = e.shiftKey ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
  });
  controls.target.copy(DEFAULT_TARGET);
  controls.update();

  window.addEventListener('resize', () => {
    canvas.style.width = W()+'px'; canvas.style.height = H()+'px';
    renderer.setSize(W(), H());
    camera.aspect = W()/H(); camera.updateProjectionMatrix();
  });

  // -- Custom shader -----------------------------------------------------------
  const VERT = [
    'attribute float aRating;',
    'varying vec3  vColor;',
    'varying float vRating;',
    'uniform float uPointSize;',
    'uniform float uNear;',
    'uniform float uFar;',
    'void main() {',
    '  vec4 mvPos = modelViewMatrix * vec4(position, 1.0);',
    '  gl_Position  = projectionMatrix * mvPos;',
    '  float rawDepth = -mvPos.z;',
    '  vColor   = color;',
    '  vRating  = aRating;',
    '  gl_PointSize = uPointSize / max(rawDepth * 0.4, 0.001);',
    '}',
  ].join(' ');
  const FRAG = [
    'varying vec3  vColor;',
    'varying float vRating;',
    'uniform float uOutline;',
    'uniform float uRatingShade;',
    'uniform float uRatingMin;',
    'uniform float uRatingMax;',
    'void main() {',
    '  if (vRating >= 0.0 && (vRating < uRatingMin || vRating > uRatingMax)) discard;',
    '  vec2  uv   = gl_PointCoord - vec2(0.5);',
    '  float dist = length(uv);',
    '  if (dist > 0.5) discard;',
    '  vec3 col = vColor;',
    '  if (uRatingShade > 0.5 && vRating >= 0.0) {',
    '    float b = 0.25 + vRating * 0.75;',
    '    col *= b;',
    '  }',
    '  if (uOutline > 0.5 && dist > 0.38) {',
    '    float t = smoothstep(0.38, 0.48, dist);',
    '    col = mix(col, vec3(0.0), t * 0.85);',
    '  }',
    '  gl_FragColor = vec4(col, 1.0);',
    '}',
  ].join(' ');

  const uniforms = {
    uPointSize:   { value: 4.0   },
    uNear:        { value: 0.1   },
    uFar:         { value: 8.0   },
    uOutline:     { value: 1.0   },  // float not bool (GLSL ES compat)
    uRatingShade: { value: 0.0   },
    uRatingMin:   { value: 0.0   },
    uRatingMax:   { value: 1.0   },
  };

  // -- Point cloud -------------------------------------------------------------
  const geometry = new THREE.BufferGeometry();
  const origPos  = posArr.slice();
  const posAttr  = new THREE.BufferAttribute(posArr, 3);
  const colAttr  = new THREE.BufferAttribute(colArr, 3);
  const ratAttr  = new THREE.BufferAttribute(ratArr, 1);
  geometry.setAttribute('position', posAttr);
  geometry.setAttribute('color',    colAttr);
  geometry.setAttribute('aRating',  ratAttr);
  geometry.setDrawRange(0, loaded);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0,0,0), 1000);
  geometry.boundingBox    = new THREE.Box3(
    new THREE.Vector3(-1000,-1000,-1000), new THREE.Vector3(1000,1000,1000));

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, uniforms,
    vertexColors: true, transparent: false, depthWrite: true, depthTest: true,
  });
  const pointsMesh = new THREE.Points(geometry, material);
  scene.add(pointsMesh);

  // -- Axes --------------------------------------------------------------------
  const AXIS_LEN = 1.35;
  function makeAxisLine(dir, color) {
    const buf = new Float32Array(6);
    if (dir===0){buf[0]=-AXIS_LEN;buf[3]=AXIS_LEN;}
    if (dir===1){buf[1]=-AXIS_LEN;buf[4]=AXIS_LEN;}
    if (dir===2){buf[2]=-AXIS_LEN;buf[5]=AXIS_LEN;}
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(buf,3));
    return new THREE.Line(geo, new THREE.LineBasicMaterial({color, opacity:0.45, transparent:true}));
  }
  const axisGroup = new THREE.Group();
  [0xe15759,0x59a14f,0x4e79a7].forEach((c,i)=>axisGroup.add(makeAxisLine(i,c)));
  scene.add(axisGroup);

  const LABEL_DATA = [
    {name:'UMAP 1',pos:new THREE.Vector3(AXIS_LEN+0.08,0,0),color:'#e15759'},
    {name:'UMAP 2',pos:new THREE.Vector3(0,AXIS_LEN+0.08,0),color:'#59a14f'},
    {name:'UMAP 3',pos:new THREE.Vector3(0,0,AXIS_LEN+0.08),color:'#4e79a7'},
  ];
  const labelDivs = LABEL_DATA.map(({name,color}) => {
    const d = document.createElement('div');
    d.textContent = name;
    d.style.cssText = 'position:fixed;font-size:11px;font-family:sans-serif;color:'+color+';opacity:0.75;pointer-events:none;z-index:10;';
    document.body.appendChild(d); return d;
  });
  let axesVisible = true;
  function setAxesVisible(v){axesVisible=v;axisGroup.visible=v;labelDivs.forEach(d=>{d.style.display=v?'block':'none';});}
  function updateLabels(){
    if(!axesVisible)return;
    LABEL_DATA.forEach(({pos},i)=>{
      const v=pos.clone().project(camera);
      labelDivs[i].style.left=((v.x*0.5+0.5)*W())+'px';
      labelDivs[i].style.top=((-v.y*0.5+0.5)*H())+'px';
      labelDivs[i].style.display=v.z>1?'none':'block';
    });
  }

  // -- Camera animation (spherical arc interpolation) --------------------------
  let camAnim=null;
  const MS_PER_RAD = 350;
  const MS_MIN     = 200;
  const MS_MAX     = 800;

  function animateCameraTo(toQ, toDist, toTarget){
    const t0    = performance.now();
    const fromQ = camera.quaternion.clone();
    const fromDist = camera.position.distanceTo(controls.target);
    const fromTarget = controls.target.clone();

    // Duration proportional to rotation angle
    const cosHalf = Math.abs(fromQ.dot(toQ));
    const angle   = 2 * Math.acos(Math.min(1, cosHalf));
    const ms      = Math.max(MS_MIN, Math.min(MS_MAX, angle * MS_PER_RAD));

    // Camera local +Z is the "back" direction (camera looks down -Z in THREE.js)
    const CAM_BACK = new THREE.Vector3(0, 0, 1);

    camAnim = now => {
      const t = Math.min((now - t0) / ms, 1);
      const e = t * t * (3 - 2 * t);  // smoothstep

      // Slerp the full camera quaternion — single operation, no gimbal issues
      const q = fromQ.clone().slerp(toQ, e);

      // Lerp distance and target
      const dist   = fromDist + (toDist - fromDist) * e;
      const target = fromTarget.clone().lerp(toTarget, e);

      // Reconstruct position: target + (camera back direction * dist)
      const back = CAM_BACK.clone().applyQuaternion(q);
      camera.position.copy(target).addScaledVector(back, dist);

      // Apply quaternion directly — bypass OrbitControls during animation
      camera.quaternion.copy(q);

      // Sync controls.target but DON'T call controls.update() mid-animation
      // as it would re-derive camera orientation from position and fight the slerp
      controls.target.copy(target);

      if (t >= 1) {
        const finalDir = new THREE.Vector3(0,0,1).applyQuaternion(toQ);
        camera.position.copy(toTarget).addScaledVector(finalDir, toDist);
        camera.quaternion.copy(toQ);
        camera.up.set(0,1,0);
        controls.target.copy(toTarget);
        controls.update();
      }

      return t < 1;
    };
  }

  function quatFromLookUp(lookDir, up) {
    const m = new THREE.Matrix4().lookAt(
      new THREE.Vector3(0,0,0),
      lookDir.clone().normalize(),
      up
    );
    return new THREE.Quaternion().setFromRotationMatrix(m);
  }

  // lockedIdx: point whose tooltip is pinned; -1 = none
  let lockedIdx = -1;
  let lastClickTime = 0, lastClickPointIdx = -1;
  const DBL_CLICK_MS = 400;

  // -- Render loop --------------------------------------------------------------
  (function animate(){
    requestAnimationFrame(animate);
    if(camAnim&&!camAnim(performance.now()))camAnim=null;
    // Only let OrbitControls update when not animating — during animation
    // we set camera.position and camera.quaternion directly, and controls.update()
    // would re-derive orientation from position and fight the slerp
    if(!camAnim) controls.update();
    uniforms.uNear.value = camera.near;
    uniforms.uFar.value  = camera.position.distanceTo(controls.target)*2.5;
    renderer.render(scene,camera);
    updateLabels();
    // Reproject locked tooltip every frame so it tracks the point as camera moves
    if (lockedIdx !== -1) showTip(lockedIdx);
  })();


  // -- Highlight / visibility
  // -- Highlight / visibility ---------------------------------------------------
  let traceVisible=null,highlightedSet=null;
  function applyColors(){
    const n=loaded;
    for(let i=0;i<n;i++){
      const t=traceArr[i],vis=!traceVisible||traceVisible[t];
      if(!vis){
        posAttr.array[i*3]=posAttr.array[i*3+1]=posAttr.array[i*3+2]=NaN;
        colArr[i*3]=colArr[i*3+1]=colArr[i*3+2]=0;
      } else {
        posAttr.array[i*3]=origPos[i*3];posAttr.array[i*3+1]=origPos[i*3+1];posAttr.array[i*3+2]=origPos[i*3+2];
        if(!highlightedSet||highlightedSet.has(t)){
          colArr[i*3]=baseR[i];colArr[i*3+1]=baseG[i];colArr[i*3+2]=baseB[i];
        } else {
          colArr[i*3]=baseR[i]*0.12+0.03;colArr[i*3+1]=baseG[i]*0.12+0.03;colArr[i*3+2]=baseB[i]*0.12+0.03;
        }
      }
    }
    posAttr.needsUpdate=true; colAttr.needsUpdate=true;
  }
  window.threeRedraw=(vis,hl)=>{traceVisible=vis;highlightedSet=hl;applyColors();};
  // Expose function to update rating filter bounds from slider JS
  window._setRatingFilter = function(lo, hi) {
    ratingFilterLo = lo; ratingFilterHi = hi;
  };
  // Count points passing both visibility and rating filter (for stats bar)
  window._getRatingFilteredCount = function(ratingLo, ratingHi, visArr) {
    let count = 0;
    for (let i = 0; i < loaded; i++) {
      const t = traceArr[i];
      if (visArr && !visArr[t]) continue;
      if (isNaN(posAttr.array[i*3])) continue;
      const r = ratArr[i];
      if (r < 0) continue; // no rating data
      if (r >= ratingLo && r <= ratingHi) count++;
    }
    return count;
  };
  // Count visible clusters where at least one point passes the rating filter
  window._getFilteredClusterCount = function(ratingLo, ratingHi, visArr) {
    const passing = new Set();
    for (let i = 0; i < loaded; i++) {
      const t = traceArr[i];
      if (visArr && !visArr[t]) continue;
      if (isNaN(posAttr.array[i*3])) continue;
      const r = ratArr[i];
      if (r < 0 || (r >= ratingLo && r <= ratingHi)) passing.add(t);
    }
    return passing.size;
  };

  // -- Hover --------------------------------------------------------------------
  const raycaster=new THREE.Raycaster();
  const mouse=new THREE.Vector2();
  const tip=document.getElementById('hover-tip');
  const tipInner=document.getElementById('hover-tip-inner');
  const tipArrow=document.getElementById('hover-tip-arrow');
  let lastIdx=-1,hoverThrottled=false;

  function relativeLuminance(r,g,b){
    const lin=v=>v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4);
    return 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b);
  }
  function showTip(idx){
    const r=baseR[idx],g=baseG[idx],b=baseB[idx];
    const br=r*0.75, bg_=g*0.75, bb=b*0.75;
    const bgHead = 'rgb('+Math.round(br*255)+','+Math.round(bg_*255)+','+Math.round(bb*255)+')';
    const bgBody = 'rgb('+Math.round(r*30)+','+Math.round(g*30)+','+Math.round(b*30)+')';
    const border = 'rgba('+Math.round(r*255)+','+Math.round(g*255)+','+Math.round(b*255)+',0.6)';
    const headLum = relativeLuminance(br, bg_, bb);
    const headTxt = headLum > 0.18 ? 'rgba(0,0,0,0.88)' : 'rgba(255,255,255,0.92)';
    const sepCol  = headLum > 0.18 ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.25)';

    tip.style.borderColor = border;
    tipInner.innerHTML = hoverArr[idx] || '(no data)';

    const head = tipInner.querySelector('.ht-head');
    const body = tipInner.querySelector('.ht-body');
    const sep  = tipInner.querySelector('.ht-sep');
    if (head) { head.style.background = bgHead; head.style.color = headTxt; }
    if (body) { body.style.background = bgBody; body.style.color = 'rgba(255,255,255,0.88)'; }
    if (sep)  { sep.style.background  = sepCol; }

    tip.style.display = 'block';
    // Project the 3D point into screen space and snap tooltip to it
    const pt3 = new THREE.Vector3(posAttr.array[idx*3], posAttr.array[idx*3+1], posAttr.array[idx*3+2]);
    const projected = pt3.clone().project(camera);
    const W = window.innerWidth - 270;
    const H = window.innerHeight;
    const sx = ( projected.x * 0.5 + 0.5) * W;
    const sy = (-projected.y * 0.5 + 0.5) * H;
    const tipW=tip.offsetWidth||300, tipH=tip.offsetHeight||80;
    const headH = head ? head.offsetHeight : 32;
    const MARGIN = 14;
    // Flip to left if not enough room on the right
    const fitsRight = sx + MARGIN + tipW + 8 < W;
    let tx, arrowOnLeft;
    if (fitsRight) {
      tx = sx + MARGIN;
      arrowOnLeft = true;
    } else {
      tx = sx - MARGIN - tipW;
      arrowOnLeft = false;
    }
    tx = Math.max(8, Math.min(tx, W - tipW - 8));
    // If point is behind camera (projected.z > 1), hide tip
    if (projected.z > 1) { tip.style.display='none'; return; }
    const ty = Math.min(Math.max(sy - headH/2, 8), window.innerHeight - tipH - 8);
    tip.style.left=tx+'px'; tip.style.top=ty+'px';
    // Arrow: rotated square, half-hidden behind tooltip edge
    // The exposed half shows border+fill; the hidden half is behind the tip box
    if (arrowOnLeft) {
      tipArrow.style.left = '-6px';
      tipArrow.style.right = '';
      tipArrow.style.background = bgHead;
      tipArrow.style.borderColor = border;
      tipArrow.style.borderRightColor = bgHead;  // mask inner edges
      tipArrow.style.borderTopColor = bgHead;
    } else {
      tipArrow.style.right = '-6px';
      tipArrow.style.left = '';
      tipArrow.style.background = bgHead;
      tipArrow.style.borderColor = border;
      tipArrow.style.borderLeftColor = bgHead;   // mask inner edges
      tipArrow.style.borderBottomColor = bgHead;
    }
    const arrowY = Math.min(Math.max(sy - ty, headH/2), tipH - 10);
    tipArrow.style.top=arrowY+'px'; tipArrow.style.transform='translateY(-50%) rotate(45deg)';
  }

  // Current rating filter bounds (normalised 0-1), kept in sync with slider
  let ratingFilterLo = 0.0, ratingFilterHi = 1.0;

  function pointPassesRatingFilter(i) {
    const r = ratArr[i];
    if (r < 0) return true;  // no rating data = always visible
    return r >= ratingFilterLo && r <= ratingFilterHi;
  }

  function raycastBest(e) {
    const rc = renderer.domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX-rc.left)/rc.width)*2-1;
    mouse.y = -((e.clientY-rc.top)/rc.height)*2+1;
    raycaster.setFromCamera(mouse, camera);
    const dist = camera.position.distanceTo(controls.target);
    raycaster.params.Points.threshold = Math.max(0.01, dist*0.02);
    const hits = raycaster.intersectObject(pointsMesh);
    if (!hits.length) return -1;
    let best=-1, bestD=Infinity;
    for (const hit of hits) {
      const i = hit.index;
      if (isNaN(posAttr.array[i*3])) continue;
      if (!pointPassesRatingFilter(i)) continue;  // skip filtered-out points
      const pt = new THREE.Vector3(posAttr.array[i*3],posAttr.array[i*3+1],posAttr.array[i*3+2]);
      const d = raycaster.ray.distanceToPoint(pt);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // Track pointer-down position to distinguish click from drag
  let mouseDownX = 0, mouseDownY = 0;
  const DRAG_THRESHOLD = 4; // pixels

  renderer.domElement.addEventListener('pointerdown', e => {
    mouseDownX = e.clientX; mouseDownY = e.clientY;
    // Hide hover tooltip when a drag/pan starts
    if (lockedIdx === -1) { tip.style.display = 'none'; lastIdx = -1; }
  });

  // Hover: only show tooltip when not locked and not dragging
  renderer.domElement.addEventListener('pointermove', e => {
    if (lockedIdx !== -1) return;
    if (e.buttons !== 0) return;  // any button held = dragging/panning, skip
    if (hoverThrottled) return; hoverThrottled = true;
    requestAnimationFrame(() => { hoverThrottled = false; });
    const best = raycastBest(e);
    if (best !== -1) { lastIdx = best; showTip(best); }
    else { tip.style.display='none'; lastIdx=-1; }
  });

  // Pointerup: lock/unlock + double-tap group highlight
  renderer.domElement.addEventListener('pointerup', e => {
    if (e.button !== 0) return;
    const dx = e.clientX - mouseDownX, dy = e.clientY - mouseDownY;
    if (Math.sqrt(dx*dx + dy*dy) > DRAG_THRESHOLD) return;

    const now = Date.now();
    const best = raycastBest(e);

    // Double-tap: check FIRST before any lock logic
    if (now - lastClickTime < DBL_CLICK_MS && lastClickTime !== 0) {
      lockedIdx = -1;
      tip.style.display = 'none';
      if (best !== -1) {
        const t = traceArr[best];
        const prevT = traceArr[lastClickPointIdx];
        if (prevT === t) {
          const tidxSet = new Set([t]);
          const already = highlightedSet && highlightedSet.size === 1 && highlightedSet.has(t);
          if (window._sidebarHighlight) window._sidebarHighlight(already ? null : tidxSet);
        } else {
          if (window._sidebarHighlight) window._sidebarHighlight(new Set([t]));
        }
      } else {
        if (window._sidebarHighlight) window._sidebarHighlight(null);
      }
      lastClickTime = 0;
      lastClickPointIdx = -1;
      return;
    }

    // If locked: unlock, immediately show hover at current position
    if (lockedIdx !== -1) {
      lockedIdx = -1;
      lastClickTime = now;
      lastClickPointIdx = best !== -1 ? best : -1;
      // Immediately reflect hover state without waiting for pointermove
      if (best !== -1) { lastIdx = best; showTip(best); }
      else { tip.style.display = 'none'; lastIdx = -1; }
      return;
    }

    // Single tap on a point — lock tooltip
    if (best !== -1) {
      lockedIdx = best;
      lastIdx = best;
      showTip(best);
    } else {
      tip.style.display = 'none';
      lastIdx = -1;
    }
    lastClickTime = now;
    lastClickPointIdx = best !== -1 ? best : -1;
  });

  // Pointer leave: hide hover tooltip only if not locked
  renderer.domElement.addEventListener('pointerleave', () => {
    if (lockedIdx !== -1) return;
    tip.style.display='none'; lastIdx=-1;
  });

  // -- Floating controls wiring -------------------------------------------------
  const sizeSlider=document.getElementById('fc-size');
  const sizeVal=document.getElementById('fc-size-val');
  if(sizeSlider){
    sizeSlider.addEventListener('input',()=>{
      const v=parseFloat(sizeSlider.value);
      sizeVal.textContent=v; uniforms.uPointSize.value=v;
    });
  }
  function wireToggle(id,key){
    const btn=document.getElementById(id); if(!btn)return;
    btn.addEventListener('click',()=>{
      const nowOn = uniforms[key].value < 0.5;
      uniforms[key].value = nowOn ? 1.0 : 0.0;
      btn.classList.toggle('active', nowOn);
    });
  }
  wireToggle('fc-outline','uOutline');
  wireToggle('fc-rating','uRatingShade');

  // Custom dual-handle rating slider
  (function(){
    const STEPS = 5;  // 0..5 inclusive
    let lo = 0, hi = 5;
    const rslider  = document.getElementById('rslider');
    const fill     = document.getElementById('rslider-fill');
    const thumbLo  = document.getElementById('rslider-lo');
    const thumbHi  = document.getElementById('rslider-hi');
    const rminVal  = document.getElementById('fc-rmin-val');
    const rmaxVal  = document.getElementById('fc-rmax-val');
    if (!rslider) return;

    function pct(v){ return v/STEPS*100; }

    function applyState(){
      fill.style.left  = pct(lo)+'%';
      fill.style.width = pct(hi-lo)+'%';
      // thumbs: offset by half their width (4.5px) to centre on the step position
      thumbLo.style.left = 'calc('+pct(lo)+'% - 4.5px)';
      thumbHi.style.left = 'calc('+pct(hi)+'% - 4.5px)';
      if(rminVal) rminVal.textContent = lo;
      if(rmaxVal) rmaxVal.textContent = hi;
      uniforms.uRatingMin.value = lo/STEPS;
      uniforms.uRatingMax.value = hi/STEPS;
      if(window._setRatingFilter) window._setRatingFilter(lo/STEPS, hi/STEPS);
      if(window._updateStats) window._updateStats(lo/STEPS, hi/STEPS);
    }

    function stepFromX(clientX){
      const rect = rslider.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return Math.round(ratio * STEPS);
    }

    function attachDrag(el, e, onMove){
      el.setPointerCapture(e.pointerId);
      el.addEventListener('pointermove', onMove);
      function up(){ el.removeEventListener('pointermove',onMove); el.removeEventListener('pointerup',up); }
      el.addEventListener('pointerup', up);
    }

    thumbLo.addEventListener('pointerdown', e => {
      e.stopPropagation();
      attachDrag(thumbLo, e, e2 => { lo = Math.min(stepFromX(e2.clientX), hi-1); if(lo<0)lo=0; applyState(); });
    });

    thumbHi.addEventListener('pointerdown', e => {
      e.stopPropagation();
      attachDrag(thumbHi, e, e2 => { hi = Math.max(stepFromX(e2.clientX), lo+1); if(hi>STEPS)hi=STEPS; applyState(); });
    });

    // Drag on fill bar: move both handles together (height handled by CSS)
    fill.addEventListener('pointerdown', e => {
      e.stopPropagation();
      const startX = e.clientX;
      const startLo = lo, startHi = hi;
      const rect = rslider.getBoundingClientRect();
      const pxPerStep = rect.width / STEPS;
      attachDrag(fill, e, e2 => {
        const dx = e2.clientX - startX;
        const ds = Math.round(dx / pxPerStep);
        let nlo = startLo + ds, nhi = startHi + ds;
        if(nlo < 0){ nlo=0; nhi=startHi-startLo; }
        if(nhi > STEPS){ nhi=STEPS; nlo=STEPS-(startHi-startLo); }
        lo=nlo; hi=nhi; applyState();
      });
    });

    // Click on track outside fill: snap nearest handle
    rslider.addEventListener('pointerdown', e => {
      if(e.target !== rslider && e.target !== rslider.querySelector('.rslider-track')) return;
      const s = stepFromX(e.clientX);
      if(Math.abs(s-lo) <= Math.abs(s-hi)) lo = Math.min(s, hi-1);
      else hi = Math.max(s, lo+1);
      if(lo<0)lo=0; if(hi>STEPS)hi=STEPS;
      applyState();
    });

    applyState();
  })();

  const DIST=3.5, Z=new THREE.Vector3(0,0,0);

  // Presets: each is [quaternion, distance, target] — no position hacks
  const camPresets = {
    'fc-reset': [
      quatFromLookUp(new THREE.Vector3(-1.8,-1.4,-2.8).normalize(), new THREE.Vector3(0,1,0)),
      DEFAULT_POS.distanceTo(DEFAULT_TARGET), Z.clone()
    ],
    'fc-top': [
      quatFromLookUp(new THREE.Vector3(0,-1,-0.00001).normalize(), new THREE.Vector3(0,0,-1)),
      DIST, Z.clone()
    ],
    'fc-front': [
      quatFromLookUp(new THREE.Vector3(0,0,-1), new THREE.Vector3(0,1,0)),
      DIST, Z.clone()
    ],
    'fc-side': [
      quatFromLookUp(new THREE.Vector3(-1,0,0), new THREE.Vector3(0,1,0)),
      DIST, Z.clone()
    ],
  };
  Object.entries(camPresets).forEach(([id,[q,dist,tgt]])=>{
    const btn=document.getElementById(id);
    if(btn) btn.addEventListener('click',()=>animateCameraTo(q,dist,tgt));
  });
"""

def threejs_script(n_points_js, chunk_files_json, n_chunks_js):
    top = f"""
<script src="{THREE_CDN}"></script>
<script src="{ORBIT_CDN}"></script>
<script>
(function() {{
  const CHUNK_FILES = {chunk_files_json};
  const N_CHUNKS    = {n_chunks_js};
  const N           = {n_points_js};

  const posArr   = new Float32Array(N * 3);
  const colArr   = new Float32Array(N * 3);
  const ratArr   = new Float32Array(N);
  const baseR    = new Float32Array(N);
  const baseG    = new Float32Array(N);
  const baseB    = new Float32Array(N);
  const traceArr = new Int32Array(N);
  const hoverArr = new Array(N);
  let loaded = 0;

  const progressWrap  = document.getElementById('progress-bar-wrap');
  const progressBar   = document.getElementById('progress-bar');
  const progressLabel = document.getElementById('progress-label');
  progressWrap.style.display  = 'block';
  progressLabel.style.display = 'block';

"""
    bot = """
  async function loadChunk(url) {
    const chunk = await fetch(url).then(r => r.json());
    const n = chunk.x.length;
    for (let i = 0; i < n; i++) {
      const p = loaded + i;
      posArr[p*3]   = origPos[p*3]   = chunk.x[i];
      posArr[p*3+1] = origPos[p*3+1] = chunk.y[i];
      posArr[p*3+2] = origPos[p*3+2] = chunk.z[i];
      const r = chunk.r[i]/255, g = chunk.g[i]/255, b = chunk.b[i]/255;
      colArr[p*3] = baseR[p] = r;
      colArr[p*3+1] = baseG[p] = g;
      colArr[p*3+2] = baseB[p] = b;
      ratArr[p]   = chunk.a != null ? chunk.a[i] : -1;
      traceArr[p] = chunk.t[i];
      hoverArr[p] = chunk.h[i];
    }
    loaded += n;
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    ratAttr.needsUpdate = true;
    geometry.setDrawRange(0, loaded);
  }

  async function loadAll() {
    let done = 0;
    for (const f of CHUNK_FILES) {
      await loadChunk(f);
      done++;
      const pct = Math.round((done / N_CHUNKS) * 100);
      progressBar.style.width   = pct + '%';
      progressLabel.textContent = 'Loading ' + done + ' / ' + N_CHUNKS + ' (' + pct + '%)';
    }
    progressWrap.style.display  = 'none';
    progressLabel.style.display = 'none';
    initSidebar();
  }

  loadAll();
})();
</script>
"""
    return top + THREEJS_BODY + bot


def threejs_inline_script(point_data_js):
    top = f"""
<script src="{THREE_CDN}"></script>
<script src="{ORBIT_CDN}"></script>
<script>
(function() {{
  const DATA = {point_data_js};
  const N = DATA.x.length;
  let loaded = N;

  const posArr   = new Float32Array(N * 3);
  const colArr   = new Float32Array(N * 3);
  const ratArr   = new Float32Array(N);
  const baseR    = new Float32Array(N);
  const baseG    = new Float32Array(N);
  const baseB    = new Float32Array(N);
  const traceArr = new Int32Array(N);
  const hoverArr = DATA.h;

  for (let i = 0; i < N; i++) {{
    posArr[i*3]   = DATA.x[i];
    posArr[i*3+1] = DATA.y[i];
    posArr[i*3+2] = DATA.z[i];
    const r = DATA.r[i]/255, g = DATA.g[i]/255, b = DATA.b[i]/255;
    colArr[i*3] = baseR[i] = r;
    colArr[i*3+1] = baseG[i] = g;
    colArr[i*3+2] = baseB[i] = b;
    ratArr[i]   = DATA.a != null ? DATA.a[i] : -1;
    traceArr[i] = DATA.t[i];
  }}

"""
    bot = """
  initSidebar();
})();
</script>
"""
    return top + THREEJS_BODY + bot


# -- Chunk serialisation -------------------------------------------------------

def build_chunks(out_dir, chunk_bytes=1_000_000):
    os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
    chunk_files = []
    chunk_idx   = 0
    buf = {"x":[],"y":[],"z":[],"r":[],"g":[],"b":[],"t":[],"h":[],"a":[]}
    buf_size = 0

    def flush(buf, idx):
        fname = f"chunks/{idx}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(buf, f, separators=(",", ":"))
        chunk_files.append(fname)
        return idx+1, {"x":[],"y":[],"z":[],"r":[],"g":[],"b":[],"t":[],"h":[],"a":[]}, 0

    for i in range(N_POINTS):
        row = {"x": all_x[i], "y": all_y[i], "z": all_z[i],
               "r": all_r[i], "g": all_g[i], "b": all_b[i],
               "t": all_trace[i], "h": all_hover[i], "a": all_rating[i]}
        sz = len(json.dumps(row))
        if buf_size + sz > chunk_bytes and buf["x"]:
            chunk_idx, buf, buf_size = flush(buf, chunk_idx)
        for k, v in row.items():
            buf[k].append(v)
        buf_size += sz

    if buf["x"]:
        chunk_idx, buf, buf_size = flush(buf, chunk_idx)

    return chunk_files


# -- Output: single HTML --------------------------------------------------------

def write_single_html(out_path, trace_names_js, sidebar_js):
    point_data_js = json.dumps({
        "x": all_x, "y": all_y, "z": all_z,
        "r": all_r, "g": all_g, "b": all_b,
        "t": all_trace, "h": all_hover, "a": all_rating,
    })
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Recipe Reviews · 3D UMAP</title>
{UI_STYLE}
</head>
<body>
<canvas id="three-canvas"></canvas>
<div id="page-title">Recipe Reviews from Food.com &middot; 3D UMAP</div>
{UI_HTML}
{ui_script(trace_names_js, sidebar_js, auto_init=False)}
{threejs_inline_script(point_data_js)}
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(html)
    print(f"  Saved -> {out_path}")


# -- Output: chunked static site ------------------------------------------------

def write_chunked_site(out_dir, trace_names_js, sidebar_js, n_clusters, n_texts):
    os.makedirs(out_dir, exist_ok=True)
    print("  Writing chunks...")
    chunk_files = build_chunks(out_dir)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"n_points": N_POINTS, "n_chunks": len(chunk_files),
                   "chunk_files": chunk_files}, f, indent=2)
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Recipe Reviews · 3D UMAP</title>
{UI_STYLE}
</head>
<body>
<canvas id="three-canvas"></canvas>
<div id="page-title">Recipe Reviews from Food.com &middot; 3D UMAP</div>
{UI_HTML}
{ui_script(trace_names_js, sidebar_js, auto_init=False)}
{threejs_script(N_POINTS, json.dumps(chunk_files), len(chunk_files))}
</body>
</html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8", errors="replace") as f:
        f.write(html)
    print(f"  Saved {len(chunk_files)} chunks + index.html -> {out_dir}/")


# -- Run ------------------------------------------------------------------------

if args.output_mode == "chunked":
    out_dir = base + "_site"
    write_chunked_site(out_dir, trace_names_js, sidebar_js, n_clusters, len(texts))
    print(f"\nTo host: drag '{out_dir}/' to https://app.netlify.com/drop")
    print(f"To preview: cd {out_dir} && python -m http.server 8080\n")
else:
    write_single_html(base + "_plot.html", trace_names_js, sidebar_js)

import webbrowser, subprocess, time

if args.output_mode == "chunked":
    srv = subprocess.Popen(
        ["python", "-m", "http.server", "8080"],
        cwd=os.path.join(os.getcwd(), base + "_site"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    webbrowser.open("http://localhost:8080")
    print("Press Ctrl+C to stop the local server.")
    try:
        srv.wait()
    except KeyboardInterrupt:
        srv.terminate()
else:
    webbrowser.open("file://" + os.path.abspath(base + "_plot.html"))