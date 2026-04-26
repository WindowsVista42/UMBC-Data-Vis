"""
Interactive preview of UMAP projection output.
Generates a standalone plotly HTML file. Open in any browser.

Auto-detects all assignment JSON files (recipes_*.json) in the pipeline
directory and adds a "Color by" dropdown to switch between them.

Hover text and coordinates are stored once. Only the color array and
legend change when switching categories.

Usage:
    uv run preview.py
    uv run preview.py --max-rows 50000
    uv run preview.py --dims 2
    uv run preview.py --output my_preview.html
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.join(SCRIPT_DIR, "pipeline")
JSONL_PATH    = os.path.join(SCRIPT_DIR, "raw", "RAW_recipes.jsonl")
SKIP_SUFFIXES = ("_index.json", "_classes.json", "_taxonomy.json")

PALETTE = (
    pc.qualitative.Plotly +
    pc.qualitative.D3 +
    pc.qualitative.G10 +
    pc.qualitative.Alphabet
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-rows", type=int, default=None, help="Max points to display (default: all)")
    parser.add_argument("--dims",     type=int, choices=[2, 3], default=None, help="Force 2D or 3D (default: auto-detect)")
    parser.add_argument("--output",   default=os.path.join(SCRIPT_DIR, "preview.html"), help="Output HTML path")
    parser.add_argument("--seed",     type=int, default=None, help="Random seed for reproducible sampling")
    return parser.parse_args()


def find_coords(dims_override):
    for d in ([dims_override] if dims_override else [3, 2]):
        coords_path = os.path.join(PIPELINE_DIR, f"recipes_umap{d}d.npy")
        index_path  = os.path.join(PIPELINE_DIR, f"recipes_umap{d}d_index.json")
        if os.path.exists(coords_path) and os.path.exists(index_path):
            return coords_path, index_path, d
    return None, None, None


def find_assignment_files():
    """Find all per-recipe assignment JSON files in the pipeline directory."""
    results = []
    for path in sorted(glob.glob(os.path.join(PIPELINE_DIR, "recipes_*.json"))):
        if any(path.endswith(s) for s in SKIP_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                if '"category"' not in f.read(512):
                    continue
        except Exception:
            continue
        name = (os.path.basename(path)
                .replace("recipes_", "").replace(".json", "")
                .replace("_", " ").title())
        results.append((path, name))
    return results


def load_recipes(target_ids):
    target  = set(target_ids)
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


def make_hover(recipe, all_assignments):
    name        = recipe.get("name", "Unknown")
    ingredients = recipe.get("ingredients") or []
    description = (recipe.get("description") or "").strip()
    minutes     = recipe.get("minutes")

    lines = [f"<b>{name}</b>"]
    for family_name, entry in all_assignments.items():
        if entry and entry.get("category"):
            lines.append(f"{family_name}: {entry['category']} ({entry.get('score', 0):.2f})")
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


def make_color_map(labels):
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(sorted(labels))}



def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    coords_path, index_path, dims = find_coords(args.dims)
    if coords_path is None:
        print("ERROR: No UMAP output found. Run pipeline/project.py first.")
        sys.exit(1)
    print(f"Using {dims}D projection: {coords_path}")

    coords = np.load(coords_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    n          = min(args.max_rows, len(index)) if args.max_rows else len(index)
    sample_pos = rng.choice(len(index), n, replace=False)
    coords_s   = coords[sample_pos]
    ids_s      = [index[i]["id"] for i in sample_pos]
    print(f"Sampled {n:,} of {len(index):,} points")

    # Load all assignment JSON files
    assignment_files = find_assignment_files()
    all_maps = {}
    for path, name in assignment_files:
        m = {}
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                m[entry["id"]] = entry
        all_maps[name] = m
        print(f"Loaded {name} ({len(m):,} entries)")

    print("Loading recipe metadata...")
    recipes = load_recipes(ids_s)

    # Build hover text (same for all families, includes all assignments)
    hover_by_id = {}
    for rid in ids_s:
        recipe      = recipes.get(rid, {"name": str(rid)})
        assignments = {name: all_maps[name].get(rid) for name in all_maps}
        hover_by_id[rid] = make_hover(recipe, assignments)

    # Build one trace per label per family.
    # Only the first family's traces are visible initially.
    # Legend clicks natively show/hide individual label traces.
    all_traces        = []
    family_trace_counts = []

    xs = coords_s[:, 0].tolist()
    ys = coords_s[:, 1].tolist()
    zs = coords_s[:, 2].tolist() if dims == 3 else None

    for fi, (family_name, assignment_map) in enumerate(all_maps.items()):
        visible = (fi == 0)
        labels  = sorted({(assignment_map.get(rid) or {}).get("category") or "Unassigned" for rid in ids_s})
        cmap    = make_color_map(labels)

        groups = {label: {"x": [], "y": [], "z": [], "hover": []} for label in labels}
        for i, rid in enumerate(ids_s):
            entry = assignment_map.get(rid)
            label = (entry["category"] if entry else None) or "Unassigned"
            g = groups[label]
            g["x"].append(xs[i])
            g["y"].append(ys[i])
            if dims == 3:
                g["z"].append(zs[i])
            g["hover"].append(hover_by_id[rid])

        count = 0
        for label in labels:
            g = groups[label]
            if dims == 3:
                all_traces.append(go.Scatter3d(
                    x=g["x"], y=g["y"], z=g["z"],
                    mode="markers",
                    name=label,
                    marker=dict(color=cmap[label], size=2, opacity=0.6),
                    text=g["hover"], hoverinfo="text",
                    visible=visible,
                ))
            else:
                all_traces.append(go.Scattergl(
                    x=g["x"], y=g["y"],
                    mode="markers",
                    name=label,
                    marker=dict(color=cmap[label], size=3, opacity=0.6),
                    text=g["hover"], hoverinfo="text",
                    visible=visible,
                ))
            count += 1
        family_trace_counts.append(count)
        print(f"  {family_name}: {count} traces")

    # Dropdown: show one family's traces at a time
    buttons = []
    for fi, family_name in enumerate(all_maps.keys()):
        visible = []
        for fj, count in enumerate(family_trace_counts):
            visible.extend([fi == fj] * count)
        buttons.append(dict(
            label=family_name,
            method="update",
            args=[{"visible": visible}, {}],
        ))

    dropdown = dict(
        buttons=buttons, direction="down", showactive=True,
        x=0.0, xanchor="left", y=1.08, yanchor="top",
        bgcolor="white", bordercolor="#cccccc", borderwidth=1,
    )

    title = f"Recipe UMAP ({dims}D) - {n:,} of {len(index):,} points"
    layout = go.Layout(
        title=title,
        updatemenus=[dropdown],
        **({"scene": dict(aspectmode="cube")} if dims == 3 else {}),
    )

    fig = go.Figure(data=all_traces, layout=layout)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
