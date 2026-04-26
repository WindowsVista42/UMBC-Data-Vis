"""
Compute per-recipe average rating and review count from RAW_interactions.csv.

Excludes zero ratings — Food.com uses 0 to indicate no rating was given,
not an actual zero-star rating.

Writes pipeline/recipe_contrib_ratings.json.gz, which export.py picks up
automatically as part of the contrib file discovery.

Usage:
    uv run pipeline/process_ratings.py
"""

import csv
import gzip
import json
import os
from collections import defaultdict

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
INTERACTIONS_PATH = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_interactions.csv")
OUTPUT_PATH       = os.path.join(SCRIPT_DIR, "recipe_contrib_ratings.json.gz")


def main():
    ratings = defaultdict(list)

    print(f"Reading {INTERACTIONS_PATH}...")
    with open(INTERACTIONS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n_rows = n_zero = 0
        for row in reader:
            n_rows += 1
            rating = int(row["rating"])
            if rating == 0:
                n_zero += 1
                continue
            ratings[row["recipe_id"]].append(rating)

    print(f"  {n_rows:,} interactions read")
    print(f"  {n_zero:,} zero-rating rows excluded")
    print(f"  {len(ratings):,} recipes have at least one non-zero rating")

    contrib = {
        rid: {
            "avg_rating": round(sum(rs) / len(rs), 3),
            "n_ratings":  len(rs),
        }
        for rid, rs in ratings.items()
    }

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(contrib, f, separators=(",", ":"))
    print(f"\nSaved {OUTPUT_PATH}  ({len(contrib):,} entries)")

    all_avg = [v["avg_rating"] for v in contrib.values()]
    print(f"  avg_rating range: {min(all_avg):.2f} - {max(all_avg):.2f}")


if __name__ == "__main__":
    main()
