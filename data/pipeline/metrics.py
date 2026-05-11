"""
metrics.py

Generates per-category and per-recipe metric JSON files.

Inputs (from --category-dir):
  recipes_{family}.json          - per-recipe category assignments
  recipes_{family}_classes.json  - ordered list of category labels

Inputs (from --raw-dir):
  RAW_recipes.csv
  RAW_interactions.csv

Outputs:
  --recipe-out-dir/   (default: artifacts/recipe_metrics/)
    {shard}.json.gz              - 100 files (00..99), each a JSON object keyed by recipe_id
                                   string: {avg_rating, n_reviews, n_ratings, count_5..count_1,
                                   n_per_year}, or {} for recipes with no interactions
  --category-out-dir/ (default: artifacts/category_metrics/)
    {family}_{slug}.json.gz      - per-category summary (reviews/year, avg score, ingredients)
    index.json                   - family/category index (plain JSON, not compressed)

Usage:
    uv run pipeline/metrics.py
    uv run pipeline/metrics.py --recipe-out-dir /tmp/recipes --category-out-dir /tmp/categories
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def slugify(s: str) -> str:
    s = s.replace("+", "plus").replace("<", "lt").replace(">", "gt")
    s = re.sub(r"[^a-zA-Z0-9.\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s.lower()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category-dir", type=Path, default=SCRIPT_DIR.parent / "artifacts",
                   help="Directory containing recipes_*.json files (default: ../artifacts)")
    p.add_argument("--raw-dir", type=Path, default=SCRIPT_DIR.parent / "raw",
                   help="Directory containing RAW_recipes.csv and RAW_interactions.csv (default: ../raw)")
    p.add_argument("--recipe-out-dir", type=Path, default=SCRIPT_DIR.parent / "artifacts" / "recipe_metrics",
                   help="Output directory for per-recipe JSONs (default: ../artifacts/recipe_metrics)")
    p.add_argument("--category-out-dir", type=Path, default=SCRIPT_DIR.parent / "artifacts" / "category_metrics",
                   help="Output directory for per-category JSONs (default: ../artifacts/category_metrics)")
    return p.parse_args()


# load data

def load_raw(raw_dir: Path):
    print("Loading RAW_recipes.csv ...")
    recipes = pd.read_csv(
        raw_dir / "RAW_recipes.csv",
        usecols=["id", "ingredients"],
        dtype={"id": int},
    )
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
        families: dict[family_name -> list[recipe_entry]]
        classes:  dict[family_name -> list[str]]
    """
    classes = {}
    for path in sorted(category_dir.glob("recipes_*_classes.json")):
        family = path.stem[len("recipes_"):-len("_classes")]
        classes[family] = json.loads(path.read_text())

    families = {}
    for family in classes:
        assignment_path = category_dir / f"recipes_{family}.json"
        if assignment_path.exists():
            families[family] = json.loads(assignment_path.read_text())

    return families, classes


# computations

def compute_recipe_summaries(recipes: pd.DataFrame, interactions: pd.DataFrame):
    """
    Returns dict: recipe_id -> reviews dict, or None for recipes with no interactions.
    """
    print("Computing per-recipe summaries ...")
    all_recipe_ids = set(recipes["id"].astype(int))

    print("  n_reviews ...", flush=True)
    n_reviews_d = interactions.groupby("recipe_id").size().to_dict()

    print("  n_per_year ...", flush=True)
    n_per_year_d: dict[int, dict] = defaultdict(dict)
    for (rid, year), count in (
        interactions.dropna(subset=["year"])
        .groupby(["recipe_id", "year"])
        .size()
        .items()
    ):
        n_per_year_d[int(rid)][int(year)] = int(count)

    print("  ratings ...", flush=True)
    rated = interactions[interactions["rating"] > 0].copy()
    rated["rating"] = rated["rating"].astype(int)
    n_ratings_d  = rated.groupby("recipe_id").size().to_dict()
    avg_rating_d = rated.groupby("recipe_id")["rating"].mean().round(3).to_dict()
    star_d = {
        star: rated[rated["rating"] == star].groupby("recipe_id").size().to_dict()
        for star in [5, 4, 3, 2, 1]
    }

    n_total = len(all_recipe_ids)
    print(f"  Building {n_total} records ...", flush=True)
    summaries = {}
    completed = 0
    for rid in all_recipe_ids:
        if rid not in n_reviews_d:
            summaries[rid] = None
        else:
            summaries[rid] = {
                "avg_rating": avg_rating_d.get(rid),
                "n_reviews":  int(n_reviews_d[rid]),
                "n_ratings":  int(n_ratings_d.get(rid, 0)),
                "count_5":    int(star_d[5].get(rid, 0)),
                "count_4":    int(star_d[4].get(rid, 0)),
                "count_3":    int(star_d[3].get(rid, 0)),
                "count_2":    int(star_d[2].get(rid, 0)),
                "count_1":    int(star_d[1].get(rid, 0)),
                "n_per_year": n_per_year_d.get(rid, {}),
            }
        completed += 1
        if completed % 10_000 == 0 or completed == n_total:
            print(f"  {completed} / {n_total}", flush=True)

    return summaries


def compute_category_summaries(families, classes, recipes: pd.DataFrame, interactions: pd.DataFrame):
    """
    Returns dict: (family, category) -> {
        "family": str,
        "category": str,
        "recipe_count": int,
        "reviews_per_year": {year: count},
        "avg_rating": float | null,
        "total_reviews": int,
        "ingredients": {ingredient: recipe_count}
    }
    """
    print("Building recipe -> ingredients lookup ...")
    recipe_ingredients = dict(zip(recipes["id"].astype(int), recipes["ingredients"]))

    # Pre-build secondary bin lookups for distribution metrics.
    # Skipped gracefully if a family is absent (e.g. pipeline not fully run).
    SEC_FAMILIES = ["minutes", "n_ingredients", "n_steps", "n_ratings", "avg_rating", "submitted", "cuisines", "meal_types"]
    sec_lookups = {
        name: {e["id"]: e["category"] for e in families[name] if e.get("category") is not None}
        for name in SEC_FAMILIES
        if name in families
    }

    print("Computing per-category summaries ...")
    summaries = {}

    for family, entries in families.items():
        rid_to_cat = {e["id"]: e["category"] for e in entries if e.get("category") is not None}
        cat_to_rids = defaultdict(list)
        for rid, cat in rid_to_cat.items():
            cat_to_rids[cat].append(rid)

        fam_iact = interactions.assign(category=interactions["recipe_id"].map(rid_to_cat))
        fam_iact = fam_iact.dropna(subset=["category"])

        # Vectorised aggregations
        year_counts_ser = (
            fam_iact.dropna(subset=["year"])
            .groupby(["category", "year"])
            .size()
        )
        avg_ratings_ser = (
            fam_iact[fam_iact["rating"] > 0]
            .groupby("category")["rating"]
            .mean()
        )
        total_reviews_ser = fam_iact.groupby("category").size()

        rated = fam_iact[fam_iact["rating"] > 0].copy()
        rated["rating"] = rated["rating"].astype(int)
        star_counts_ser = rated.groupby(["category", "rating"]).size()

        # Secondary bin distributions (minutes, n_ingredients, n_steps)
        sec_dists = {}
        for sec_name, sec_lookup in sec_lookups.items():
            sec_classes = classes.get(sec_name, [])
            if not sec_classes:
                continue
            all_rids = list(rid_to_cat.keys())
            df = pd.DataFrame({
                "primary":   [rid_to_cat[r] for r in all_rids],
                "secondary": [sec_lookup.get(r) for r in all_rids],
            }).dropna(subset=["secondary"])
            sec_dists[sec_name] = (df.groupby(["primary", "secondary"]).size(), sec_classes)

        family_classes = classes.get(family, sorted(cat_to_rids.keys()))

        for cat in family_classes:
            rids = cat_to_rids.get(cat, [])

            try:
                rpy = {int(y): int(c) for y, c in year_counts_ser[cat].items()}
            except KeyError:
                rpy = {}

            avg_rating = round(float(avg_ratings_ser[cat]), 4) if cat in avg_ratings_ser.index else None
            total = int(total_reviews_ser[cat]) if cat in total_reviews_ser.index else 0

            try:
                sc = star_counts_ser[cat]
                count_1, count_2, count_3, count_4, count_5 = (int(sc.get(s, 0)) for s in [1, 2, 3, 4, 5])
            except KeyError:
                count_1 = count_2 = count_3 = count_4 = count_5 = 0

            ingredient_counts = defaultdict(int)
            for rid in rids:
                for ing in recipe_ingredients.get(rid, []):
                    ingredient_counts[ing] += 1

            summary = {
                "family":       family,
                "category":     cat,
                "recipe_count": len(rids),
                "reviews": {
                    "avg_rating":    avg_rating,
                    "total_reviews": total,
                    "count_5":       count_5,
                    "count_4":       count_4,
                    "count_3":       count_3,
                    "count_2":       count_2,
                    "count_1":       count_1,
                    "n_per_year":    dict(sorted(rpy.items())),
                },
                "ingredients": dict(sorted(ingredient_counts.items(), key=lambda x: -x[1])),
            }

            for sec_name, (ct, sec_classes) in sec_dists.items():
                try:
                    sc_ser = ct[cat]
                    summary[sec_name] = {lbl: int(sc_ser.get(lbl, 0)) for lbl in sec_classes}
                except KeyError:
                    summary[sec_name] = {lbl: 0 for lbl in sec_classes}

            summaries[(family, cat)] = summary
            print(f"  [{family}] {cat}: {len(rids)} recipes")

    return summaries


# outputs

def write_recipe_jsons(recipe_summaries, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    shards: dict[str, dict] = defaultdict(dict)
    for rid, summary in recipe_summaries.items():
        shard = f"{rid:02d}"[-2:]
        shards[shard][str(rid)] = {} if summary is None else summary

    n_total = len(shards)
    print(f"Writing {n_total} recipe shard files ...")
    for completed, (shard, records) in enumerate(sorted(shards.items()), 1):
        with gzip.open(out_dir / f"{shard}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(records, f, separators=(",", ":"))
        if completed % 10 == 0 or completed == n_total:
            print(f"  {completed} / {n_total}", flush=True)


def write_category_jsons(category_summaries, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(category_summaries)} category JSON files ...")
    for (family, cat), summary in category_summaries.items():
        filename = f"{family}_{slugify(cat)}.json.gz"
        with gzip.open(out_dir / filename, "wt", encoding="utf-8") as f:
            json.dump(summary, f, separators=(",", ":"))


def write_index(families, classes, out_dir: Path):
    index = {
        family: {
            label: f"{family}_{slugify(label)}.json.gz"
            for label in label_list
        }
        for family, label_list in classes.items()
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {out_dir}/index.json")


def main():
    args = parse_args()

    families, classes = load_category_mappings(args.category_dir)
    print(f"Loaded {len(families)} category families: {sorted(families.keys())}")

    recipes, interactions = load_raw(args.raw_dir)

    recipe_summaries = compute_recipe_summaries(recipes, interactions)
    category_summaries = compute_category_summaries(families, classes, recipes, interactions)

    write_recipe_jsons(recipe_summaries, args.recipe_out_dir)
    write_category_jsons(category_summaries, args.category_out_dir)
    write_index(families, classes, args.category_out_dir)

    print(f"\nDone.")
    print(f"  {args.recipe_out_dir}/    {len(recipe_summaries)} files")
    print(f"  {args.category_out_dir}/  {len(category_summaries)} files + index.json")


if __name__ == "__main__":
    main()
