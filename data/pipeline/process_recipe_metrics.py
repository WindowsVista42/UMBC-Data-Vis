"""
Compute per-recipe metrics from RAW_interactions.csv and write one JSON file
per recipe into data/recipe_metrics/<recipe_id>.json.

Outputs per recipe:
- avg_rating
- n_ratings
- count_5, count_4, count_3, count_2, count_1
- count_no_rating
- ratings: [{date, rating, review}, ...]

Usage:
    uv run pipeline/process_recipe_metrics.py
"""

import csv
import json
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = os.path.join(DATA_DIR, "recipe_metrics")
INTERACTIONS_PATH = os.path.join(RAW_DIR, "RAW_interactions.csv")
RECIPES_JSONL_PATH = os.path.join(RAW_DIR, "RAW_recipes.jsonl")
RECIPES_CSV_PATH = os.path.join(RAW_DIR, "RAW_recipes.csv")


def load_recipes():
    recipes_path = RECIPES_JSONL_PATH if os.path.exists(RECIPES_JSONL_PATH) else RECIPES_CSV_PATH
    if not os.path.exists(recipes_path):
        raise FileNotFoundError(
            "Could not find RAW_recipes.jsonl or RAW_recipes.csv in data/raw/"
        )

    recipes = {}
    print(f"Reading {recipes_path}...")

    if recipes_path.endswith(".jsonl"):
        with open(recipes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row["id"])
                recipes[rid] = {
                    "recipe_id": rid,
                    "name": row.get("name", ""),
                }
    else:
        with open(recipes_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = str(row["id"])
                recipes[rid] = {
                    "recipe_id": rid,
                    "name": row.get("name", ""),–
                }

    print(f"  {len(recipes):,} recipes loaded")
    return recipes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recipes = load_recipes()

    metrics = defaultdict(
        lambda: {
            "count_5": 0,
            "count_4": 0,
            "count_3": 0,
            "count_2": 0,
            "count_1": 0,
            "count_no_rating": 0,
            "ratings": [],
        }
    )

    print(f"Reading {INTERACTIONS_PATH}...")
    with open(INTERACTIONS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n_rows = 0
        for row in reader:
            n_rows += 1
            recipe_id = row["recipe_id"]
            rating = int(row["rating"])
            date = (row.get("date") or "").strip()
            review = (row.get("review") or "").strip()

            bucket = metrics[recipe_id]
            if rating == 0:
                bucket["count_no_rating"] += 1
                continue

            if 1 <= rating <= 5:
                bucket[f"count_{rating}"] += 1
                bucket["ratings"].append({
                    "date": date,
                    "rating": rating,
                    "review": review,
                })

    print(f"  {n_rows:,} interactions read")
    print(f"  {len(metrics):,} recipes had at least one interaction")

    written = 0
    for rid in sorted(recipes.keys(), key=lambda value: int(value)):
        bucket = metrics.get(rid)
        ratings = []
        if bucket:
            ratings = sorted(
                bucket["ratings"],
                key=lambda entry: (
                    entry.get("date") or "",
                    entry.get("rating") or 0,
                    entry.get("review") or "",
                ),
            )

        n_ratings = len(ratings)
        avg_rating = round(sum(entry["rating"] for entry in ratings) / n_ratings, 3) if n_ratings else None

        record = {
            "recipe_id": rid,
            "name": recipes[rid]["name"],
            "metrics": {
                "avg_rating": avg_rating,
                "n_ratings": n_ratings,
                "count_5": bucket["count_5"] if bucket else 0,
                "count_4": bucket["count_4"] if bucket else 0,
                "count_3": bucket["count_3"] if bucket else 0,
                "count_2": bucket["count_2"] if bucket else 0,
                "count_1": bucket["count_1"] if bucket else 0,
                "count_no_rating": bucket["count_no_rating"] if bucket else 0,
            },
            "ratings": ratings,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{rid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
        written += 1

    print(f"  {written:,} per-recipe JSON files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
