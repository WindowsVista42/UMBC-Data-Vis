"""
Assign each recipe to the nearest category by cosine similarity.

Takes a .txt file of category names (one per line) and assigns each recipe
to the closest one using dot product similarity against the recipe embeddings
from embed.py. Both sides are L2-normalized, so dot product == cosine similarity.

The output filename is derived from the input .txt filename:
  cuisines.txt      -> recipes_cuisines.json
  meal_types.txt    -> recipes_meal_types.json

Output format per recipe:
  {"id": 137739, "cuisine": "Mexican", "score": 0.42,
   "runners_up": [{"cuisine": "Spanish", "score": 0.38}, ...]}

Recipes below --min-score get null as their category but still show their
best score, so you can see what the assignment would have been.

Usage:
    uv run pipeline/assign.py pipeline/cuisines.txt
    uv run pipeline/assign.py pipeline/cuisines.txt --min-score 0.15
    uv run pipeline/assign.py pipeline/cuisines.txt --category-template "{category} cuisine"
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMBEDDINGS = os.path.join(SCRIPT_DIR, "recipes_embeddings.npy")
DEFAULT_INDEX      = os.path.join(SCRIPT_DIR, "recipes_index.json")
MODEL_NAME         = "all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("categories", help="Path to .txt file with one category name per line")
    parser.add_argument("--embeddings",  default=DEFAULT_EMBEDDINGS, help="Path to recipes_embeddings.npy")
    parser.add_argument("--index",       default=DEFAULT_INDEX,      help="Path to recipes_index.json")
    parser.add_argument("--min-score",   type=float, default=0.0,
                        help="Minimum cosine similarity to assign a category. "
                             "Recipes below this get null. (default: 0.0, off)")
    parser.add_argument("--category-template", default="{category}",
                        help="Template for embedding category names. Use {category} as placeholder. "
                             "e.g. '{category} cuisine' embeds 'Italian cuisine' but stores 'Italian'. "
                             "(default: '{category}')")
    return parser.parse_args()


def load_categories(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def output_path(categories_file):
    name = os.path.splitext(os.path.basename(categories_file))[0]
    return os.path.join(SCRIPT_DIR, f"recipes_{name}.json")


def main():
    args = parse_args()

    for path in [args.categories, args.embeddings, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    categories = load_categories(args.categories)
    print(f"Categories ({len(categories)}): {', '.join(categories)}")

    print(f"\nLoading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"  Shape: {embeddings.shape}")

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)

    print(f"\nEmbedding category names with '{MODEL_NAME}'...")
    if args.category_template != "{category}":
        print(f"  Using template: '{args.category_template}'")
    embed_labels = [args.category_template.replace("{category}", c) for c in categories]
    model = SentenceTransformer(MODEL_NAME)
    category_embeddings = model.encode(
        embed_labels,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Computing similarities ({len(index):,} recipes x {len(categories)} categories)...")
    sims = embeddings @ category_embeddings.T  # (N, M)

    top3 = np.argsort(sims, axis=1)[:, -3:][:, ::-1]
    top3_scores = sims[np.arange(len(sims))[:, None], top3]

    out = output_path(args.categories)
    results = []
    for i, entry in enumerate(index):
        winning_score = float(top3_scores[i, 0])
        assigned = categories[top3[i, 0]] if winning_score >= args.min_score else None
        results.append({
            "id":       entry["id"],
            "category": assigned,
            "score":    round(winning_score, 4),
            "runners_up": [
                {"category": categories[top3[i, 1]], "score": round(float(top3_scores[i, 1]), 4)},
                {"category": categories[top3[i, 2]], "score": round(float(top3_scores[i, 2]), 4)},
            ],
        })

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"Saved {out}  ({len(results):,} entries)")

    dist = Counter(r["category"] for r in results)
    print(f"\nDistribution (top 10):")
    for label, count in dist.most_common(10):
        pct = count * 100 / len(results)
        print(f"  {str(label):<20} {count:>6,}  ({pct:.1f}%)")
    if args.min_score > 0:
        n_null = dist[None]
        print(f"\n  {n_null:,} recipes below min-score {args.min_score} -> null ({n_null*100/len(results):.1f}%)")


if __name__ == "__main__":
    main()
