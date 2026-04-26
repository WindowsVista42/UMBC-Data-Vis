"""
Interactive inspection of UMAP projection output.
Generates a standalone plotly HTML file. Open in any browser.

Auto-detects 3D vs 2D projection and cuisine assignment.
Colors by cuisine (with legend) if assignment exists, otherwise uniform.
Hover shows recipe name, cuisine, time, ingredients, and description.

Usage:
    uv run preview.py
    uv run preview.py --max-rows 10000
    uv run preview.py --dims 2
    uv run preview.py --output my_preview.html
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(SCRIPT_DIR, "pipeline")
JSONL_PATH = os.path.join(SCRIPT_DIR, "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows", type=int, default=None, help="Max points to display (default: all)")
    parser.add_argument("--dims", type=int, choices=[2, 3], default=None, help="Force 2D or 3D (default: auto-detect)")
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "preview.html"), help="Output HTML path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--assignment", default=os.path.join(PIPELINE_DIR, "recipes_cuisines.json"),
                        help="Path to assignment JSON from assign.py (default: pipeline/recipes_cuisines.json)")
    return parser.parse_args()


def find_coords(dims_override):
    for d in ([dims_override] if dims_override else [3, 2]):
        coords_path = os.path.join(PIPELINE_DIR, f"recipes_umap{d}d.npy")
        index_path  = os.path.join(PIPELINE_DIR, f"recipes_umap{d}d_index.json")
        if os.path.exists(coords_path) and os.path.exists(index_path):
            return coords_path, index_path, d
    return None, None, None


def load_recipes(target_ids):
    target = set(target_ids)
    recipes = {}
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["id"] in target:
                recipes[r["id"]] = r
    return recipes


def make_hover(recipe, cuisine_entry):
    name        = recipe.get("name", "Unknown")
    ingredients = recipe.get("ingredients") or []
    description = (recipe.get("description") or "").strip()
    minutes     = recipe.get("minutes")

    lines = [f"<b>{name}</b>"]
    if cuisine_entry:
        score = cuisine_entry.get("score", 0)
        lines.append(f"Cuisine: {cuisine_entry['category']} ({score:.2f})")
    if minutes:
        lines.append(f"Time: {minutes} min")
    if ingredients:
        shown = ", ".join(str(x) for x in ingredients[:6])
        if len(ingredients) > 6:
            shown += f" +{len(ingredients) - 6} more"
        lines.append(f"Ingredients: {shown}")
    if description:
        lines.append(f"<i>{description[:200]}{'...' if len(description) > 200 else ''}</i>")
    return "<br>".join(lines)


def build_traces(coords_s, ids_s, hover, cuisine_map, dims):
    groups = defaultdict(lambda: defaultdict(list))
    for i, rid in enumerate(ids_s):
        label = (cuisine_map[rid]["category"] if rid in cuisine_map else None) or "Unassigned"
        groups[label]["x"].append(float(coords_s[i, 0]))
        groups[label]["y"].append(float(coords_s[i, 1]))
        if dims == 3:
            groups[label]["z"].append(float(coords_s[i, 2]))
        groups[label]["hover"].append(hover[i])

    traces = []
    for label in sorted(groups):
        g = groups[label]
        if dims == 3:
            traces.append(go.Scatter3d(
                x=g["x"], y=g["y"], z=g["z"],
                mode="markers",
                name=label,
                marker=dict(size=2, opacity=0.6),
                text=g["hover"],
                hoverinfo="text",
            ))
        else:
            traces.append(go.Scattergl(
                x=g["x"], y=g["y"],
                mode="markers",
                name=label,
                marker=dict(size=3, opacity=0.6),
                text=g["hover"],
                hoverinfo="text",
            ))
    return traces


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    coords_path, index_path, dims = find_coords(args.dims)
    if coords_path is None:
        print("ERROR: No UMAP output found. Run pipeline/project.py first.")
        sys.exit(1)
    print(f"Using {dims}D projection: {coords_path}")

    coords = np.load(coords_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    n = min(args.max_rows, len(index)) if args.max_rows else len(index)
    sample_pos = rng.choice(len(index), n, replace=False)
    coords_s = coords[sample_pos]
    ids_s = [index[i]["id"] for i in sample_pos]
    print(f"Sampled {n:,} of {len(index):,} points")

    cuisine_map = {}
    if os.path.exists(args.assignment):
        with open(args.assignment, encoding="utf-8") as f:
            for entry in json.load(f):
                cuisine_map[entry["id"]] = entry
        print(f"Assignment loaded from {args.assignment} ({len(cuisine_map):,} entries)")
    else:
        print(f"No assignment file found at {args.assignment} - using uniform color")

    print("Loading recipe metadata...")
    recipes = load_recipes(ids_s)

    hover = [make_hover(recipes.get(rid, {"name": str(rid)}), cuisine_map.get(rid)) for rid in ids_s]

    traces = build_traces(coords_s, ids_s, hover, cuisine_map, dims)

    title = f"Recipe UMAP ({dims}D) - {n:,} of {len(index):,} points"
    if dims == 3:
        layout = go.Layout(title=title, scene=dict(aspectmode="cube"))
    else:
        layout = go.Layout(title=title)

    fig = go.Figure(data=traces, layout=layout)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
