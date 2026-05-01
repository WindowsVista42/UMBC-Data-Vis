"""
Show a histogram of a numeric field across all recipes.

Useful for deciding bin edges/centers for encode_ordinal.py configs.
Prints percentile stats and opens an interactive plotly histogram.

Reads from RAW_recipes.jsonl by default. Use --contrib for fields that
come from a contrib file (e.g. avg_rating, n_ratings).

Usage:
    uv run histogram.py minutes
    uv run histogram.py n_steps --cap 100
    uv run histogram.py avg_rating --contrib artifacts/recipe_contrib_ratings.json.gz
    uv run histogram.py n_ratings  --contrib artifacts/recipe_contrib_ratings.json.gz --cap 200
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(SCRIPT_DIR, "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("field",      help="Field name to plot (e.g. minutes, n_steps, avg_rating)")
    parser.add_argument("--contrib",  default=None, help="Path to recipe_contrib_*.json.gz to read field from")
    parser.add_argument("--cap",      type=float,   default=None, help="Cap values at this maximum before plotting (default: p99)")
    parser.add_argument("--n-bins",   type=int,     default=50,   help="Number of histogram bins (default: 50)")
    return parser.parse_args()


def extract_value(record, field):
    val = record.get(field)
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


def load_values(field, contrib_path):
    values = []
    if contrib_path:
        if not os.path.exists(contrib_path):
            print(f"ERROR: File not found: {contrib_path}")
            sys.exit(1)
        with gzip.open(contrib_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for rec in data.values():
            val = extract_value(rec, field)
            if val is not None:
                values.append(val)
    else:
        if not os.path.exists(JSONL_PATH):
            print(f"ERROR: File not found: {JSONL_PATH}")
            sys.exit(1)
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                val = extract_value(json.loads(line), field)
                if val is not None:
                    values.append(val)
    return np.array(values, dtype=np.float64)


def print_stats(values, field, cap):
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(values, pcts)

    print(f"\nField: {field}  ({len(values):,} values)")
    print(f"  min    {values.min():.2f}")
    print(f"  max    {values.max():.2f}")
    print(f"  mean   {values.mean():.2f}")
    print(f"  std    {values.std():.2f}")
    print(f"\n  Percentiles:")
    for p, v in zip(pcts, pct_vals):
        bar = "#" * int((p / 100) * 30)
        print(f"    p{p:<3}  {v:>10.2f}  {bar}")

    if cap is not None:
        n_capped = int((values > cap).sum())
        print(f"\n  Values above cap ({cap}): {n_capped:,}  ({n_capped * 100 / len(values):.1f}%)")


def main():
    args = parse_args()

    print(f"Loading '{args.field}'...")
    values = load_values(args.field, args.contrib)

    if len(values) == 0:
        print(f"ERROR: No values found for field '{args.field}'")
        sys.exit(1)

    cap = args.cap if args.cap is not None else float(np.percentile(values, 99))
    print_stats(values, args.field, cap)

    plot_values = np.clip(values, None, cap)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=plot_values,
        nbinsx=args.n_bins,
        marker_color="#5b8dee",
        opacity=0.85,
    ))

    n_capped = int((values > cap).sum())
    title = f"{args.field}  (n={len(values):,},  capped at p99={cap:.1f},  {n_capped:,} outliers excluded)"
    if args.cap is not None:
        title = f"{args.field}  (n={len(values):,},  capped at {cap},  {n_capped:,} outliers excluded)"

    fig.update_layout(
        title=title,
        xaxis_title=args.field,
        yaxis_title="count",
        bargap=0.02,
        template="plotly_dark",
    )

    fig.show()


if __name__ == "__main__":
    main()
