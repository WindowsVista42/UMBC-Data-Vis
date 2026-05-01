"""
gen_category_metrics.py

Generates per-category and per-recipe metric JSON files.

Inputs (expected in --category-dir):
  recipes_{family}.json          - per-recipe category assignments
  recipes_{family}_classes.json  - ordered list of category labels

Inputs (expected in --raw-dir, downloaded from Kaggle if missing):
  RAW_recipes.csv
  RAW_interactions.csv

Outputs written to --out-dir:
  recipes/
    {recipe_id}.json             - per-recipe summary (reviews/year, avg score)
  categories/
    {family}_{category}.json     - per-category summary (reviews/year, avg score, ingredients)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


# run command formatting
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category-dir", default=".",
                   help="Directory containing recipes_*.json files from the pipeline")
    p.add_argument("--raw-dir", default="raw",
                   help="Directory containing RAW_recipes.csv and RAW_interactions.csv")
    p.add_argument("--out-dir", default="category_metrics",
                   help="Output directory for generated JSON files")
    p.add_argument("--download", action="store_true",
                   help="Download dataset from Kaggle if raw files are missing")
    return p.parse_args()


# load data

def maybe_download(raw_dir: Path):
    recipes_path = raw_dir / "RAW_recipes.csv"
    interactions_path = raw_dir / "RAW_interactions.csv"
    if recipes_path.exists() and interactions_path.exists():
        return
    print("Raw files not found — downloading from Kaggle...")
    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download(
            "shuyangli94/food-com-recipes-and-user-interactions"
        )
        dataset_path = Path(dataset_path)
        raw_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for fname in ["RAW_recipes.csv", "RAW_interactions.csv"]:
            src = dataset_path / fname
            if src.exists():
                shutil.copy(src, raw_dir / fname)
            else:
                print(f"  Warning: {fname} not found in downloaded dataset at {dataset_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please place RAW_recipes.csv and RAW_interactions.csv in:", raw_dir)
        sys.exit(1)


def load_raw(raw_dir: Path):
    print("Loading RAW_recipes.csv ...")
    recipes = pd.read_csv(
        raw_dir / "RAW_recipes.csv",
        usecols=["id", "ingredients", "name"],
        dtype={"id": int},
    )
    # ['salt', 'pepper', ...]
    recipes["ingredients"] = recipes["ingredients"].apply(_parse_list)

    print("Loading RAW_interactions.csv ...")
    interactions = pd.read_csv(
        raw_dir / "RAW_interactions.csv",
        usecols=["recipe_id", "date", "rating"],
        dtype={"recipe_id": int, "rating": float},
    )
    interactions["year"] = pd.to_datetime(interactions["date"], errors="coerce").dt.year
    return recipes, interactions


def _parse_list(s):
    try:
        import ast
        return ast.literal_eval(s)
    except Exception:
        return []


def load_category_mappings(category_dir: Path):
    """
    Returns:
        families: dict[family_name -> list[recipe_entry]]   (raw loaded JSON)
        classes:  dict[family_name -> list[str]]
    """
    families = {}
    classes = {}
    for path in sorted(category_dir.glob("recipes_*.json")):
        name = path.stem
        if name.endswith("_classes"):
            family = name[len("recipes_"):-len("_classes")]
            classes[family] = json.loads(path.read_text())
        else:
            family = name[len("recipes_"):]
            families[family] = json.loads(path.read_text())
    return families, classes


# computations

def build_indexes(families):
    """Build recipe_id -> {family: category} lookup."""
    # recipe_id -> dict
    recipe_categories = {}  # {id: {family: category}}
    for family, entries in families.items():
        for entry in entries:
            rid = entry["id"]
            if rid not in recipe_categories:
                recipe_categories[rid] = {}
            recipe_categories[rid][family] = entry["category"]
    return recipe_categories


def compute_recipe_summaries(interactions: pd.DataFrame):
    """
    Returns dict: recipe_id -> {
        "reviews_per_year": {year: count},
        "avg_rating": float | null,
        "total_reviews": int
    }
    """
    print("Computing per-recipe summaries ...")
    summaries = {}

    grouped = interactions.groupby("recipe_id")
    for rid, group in grouped:
        reviews_by_year = (
            group.dropna(subset=["year"])
            .groupby("year")
            .size()
            .to_dict()
        )
        # convert numpy int64 keys/values to plain Python ints
        reviews_by_year = {int(y): int(c) for y, c in reviews_by_year.items()}

        valid_ratings = group["rating"].dropna()
        avg_rating = round(float(valid_ratings.mean()), 4) if len(valid_ratings) > 0 else None

        summaries[int(rid)] = {
            "reviews_per_year": reviews_by_year,
            "avg_rating": avg_rating,
            "total_reviews": int(len(group)),
        }

    return summaries


def compute_category_summaries(families, classes, recipes: pd.DataFrame, interactions: pd.DataFrame):
    """
    Returns dict: (family, category) -> {
        "family": str,
        "category": str,
        "recipe_count": int,
        "reviews_per_year": {year: count},
        "avg_rating": float | null,
        "ingredients": {ingredient: recipe_count}   (# recipes using it)
    }
    """
    print("Building recipe -> ingredients lookup ...")
    recipe_ingredients = dict(zip(recipes["id"].astype(int), recipes["ingredients"]))

    print("Building recipe -> interactions lookup ...")
    # group interactions by recipe_id
    interactions_by_recipe = {
        int(rid): grp
        for rid, grp in interactions.groupby("recipe_id")
    }

    print("Computing per-category summaries ...")
    summaries = {}

    for family, entries in families.items():
        # group recipe ids by category
        cat_to_ids = defaultdict(list)
        for entry in entries:
            cat = entry["category"]
            if cat is not None:
                cat_to_ids[cat].append(entry["id"])

        family_classes = classes.get(family, sorted(cat_to_ids.keys()))

        for cat in family_classes:
            rids = cat_to_ids.get(cat, [])

            # reviews and avg ratings
            year_counts = defaultdict(int)
            all_ratings = []
            for rid in rids:
                idf = interactions_by_recipe.get(rid)
                if idf is None:
                    continue
                for _, row in idf.iterrows():
                    if pd.notna(row["year"]):
                        year_counts[int(row["year"])] += 1
                    if pd.notna(row["rating"]):
                        all_ratings.append(row["rating"])

            avg_rating = round(float(sum(all_ratings) / len(all_ratings)), 4) if all_ratings else None

            # ingredients
            ingredient_counts = defaultdict(int)
            for rid in rids:
                for ing in recipe_ingredients.get(rid, []):
                    ingredient_counts[ing] += 1

            summaries[(family, cat)] = {
                "family": family,
                "category": cat,
                "recipe_count": len(rids),
                "reviews_per_year": dict(sorted(year_counts.items())),
                "avg_rating": avg_rating,
                "total_reviews": sum(year_counts.values()),
                "ingredients": dict(
                    sorted(ingredient_counts.items(), key=lambda x: -x[1])
                ),
            }
            print(f"  [{family}] {cat}: {len(rids)} recipes")

    return summaries


# outputs

def write_recipe_jsons(recipe_summaries, recipe_categories, out_dir: Path):
    recipes_dir = out_dir / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(recipe_summaries)} recipe JSON files ...")
    for rid, summary in recipe_summaries.items():
        data = {
            "id": rid,
            **summary,
            "categories": recipe_categories.get(rid, {}),
        }
        (recipes_dir / f"{rid}.json").write_text(json.dumps(data))


def write_category_jsons(category_summaries, out_dir: Path):
    cats_dir = out_dir / "categories"
    cats_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(category_summaries)} category JSON files ...")
    for (family, cat), summary in category_summaries.items():
        filename = f"{family}_{cat}.json"
        (cats_dir / filename).write_text(json.dumps(summary))


def write_index(families, classes, out_dir: Path):
    """Write an index.json listing all families and their categories."""
    index = {}
    for family, class_list in classes.items():
        index[family] = class_list
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print("Wrote index.json")


# main
def main():
    args = parse_args()
    category_dir = Path(args.category_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.download:
        maybe_download(raw_dir)

    # check for files
    for fname in ["RAW_recipes.csv", "RAW_interactions.csv"]:
        if not (raw_dir / fname).exists():
            print(f"Error: {raw_dir / fname} not found.")
            print("Run with --download to fetch from Kaggle, or place files manually.")
            sys.exit(1)

    # load category mappings and raw data
    families, classes = load_category_mappings(category_dir)
    print(f"Loaded {len(families)} category families: {sorted(families.keys())}")

    recipes, interactions = load_raw(raw_dir)

    recipe_categories = build_indexes(families)

    # compute summaries
    recipe_summaries = compute_recipe_summaries(interactions)
    category_summaries = compute_category_summaries(
        families, classes, recipes, interactions
    )

    #output
    out_dir.mkdir(parents=True, exist_ok=True)
    write_recipe_jsons(recipe_summaries, recipe_categories, out_dir)
    write_category_jsons(category_summaries, out_dir)
    write_index(families, classes, out_dir)

    print(f"\nDone! Output written to: {out_dir}/")
    print(f"  {out_dir}/recipes/       - {len(recipe_summaries)} files (one per recipe)")
    print(f"  {out_dir}/categories/    - {len(category_summaries)} files (one per family+category)")
    print(f"  {out_dir}/index.json     - family/category index")


if __name__ == "__main__":
    main()