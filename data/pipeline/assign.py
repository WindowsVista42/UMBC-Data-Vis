"""
Assign each recipe to a cuisine using k-NN classification.

Recipes tagged with cuisines from cuisines.txt are used as training data.
The taxonomy from cuisine_taxonomy.json is used to assign the most specific
label when a recipe has multiple matching cuisine tags (e.g. a recipe tagged
both 'american' and 'southern-united-states' gets 'southern-united-states').

A KNeighborsClassifier with distance weighting is then fitted on those tagged
recipes and used to classify all recipes. predict_proba gives a confidence
score across all classes.

Output filename is derived from the input .txt filename:
  cuisines.txt -> recipes_cuisines.json

Output format:
  [{"id": 137739, "category": "mexican", "score": 0.72,
    "runners_up": [{"category": "spanish", "score": 0.15}, ...]}, ...]

Recipes below --min-score get null as their category but still show their
best score so you can see what the assignment would have been.

Usage:
    uv run pipeline/assign.py pipeline/cuisines.txt
    uv run pipeline/assign.py pipeline/cuisines.txt --n-neighbors 100
    uv run pipeline/assign.py pipeline/cuisines.txt --min-score 0.2
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR      = os.path.join(SCRIPT_DIR, "..", "artifacts")
DEFAULT_EMBEDDINGS = os.path.join(ARTIFACTS_DIR, "recipes_embeddings.npy")
DEFAULT_INDEX      = os.path.join(ARTIFACTS_DIR, "recipes_index.json")
JSONL_PATH         = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")
PREDICT_BATCH_SIZE = 10_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("categories",    help="Path to .txt file with one cuisine tag per line")
    parser.add_argument("--embeddings",  default=DEFAULT_EMBEDDINGS, help="Path to recipes_embeddings.npy")
    parser.add_argument("--index",       default=DEFAULT_INDEX,      help="Path to recipes_index.json")
    parser.add_argument("--taxonomy",    default=None,
                        help="Path to taxonomy JSON (default: auto-derived from categories filename)")
    parser.add_argument("--n-neighbors", type=int, default=100,
                        help="Number of neighbors (default: 100)")
    parser.add_argument("--min-score",   type=float, default=0.0,
                        help="Minimum probability to assign a category. Below this gets null. (default: 0.0)")
    parser.add_argument("--no-prior-correction", action="store_true",
                        help="Disable prior correction (by default, class frequencies are divided out to "
                             "remove imbalance bias from the KNN posterior)")
    return parser.parse_args()


def load_categories(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def output_path(categories_file):
    name = os.path.splitext(os.path.basename(categories_file))[0]
    return os.path.join(ARTIFACTS_DIR, f"recipes_{name}.json")


def most_specific_tag(matching_tags, parent_of):
    """Return the most specific (leaf) tags from a set of matching cuisine tags."""
    leaves = set(matching_tags)
    for tag in matching_tags:
        ancestor = parent_of.get(tag)
        while ancestor:
            leaves.discard(ancestor)
            ancestor = parent_of.get(ancestor)
    return leaves


def main():
    args = parse_args()

    for path in [args.categories, args.embeddings, args.index]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    categories  = load_categories(args.categories)
    cuisine_set = set(categories)
    print(f"Categories ({len(categories)}): {', '.join(categories)}")

    base     = os.path.splitext(os.path.basename(args.categories))[0]
    taxonomy_path = args.taxonomy or os.path.join(ARTIFACTS_DIR, f"{base}_taxonomy.json")

    parent_of = {}
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path, encoding="utf-8") as f:
            parent_of = json.load(f).get("parent_of", {})
        print(f"Taxonomy loaded from {taxonomy_path} ({len(parent_of)} parent-child relationships)")
    else:
        print(f"No taxonomy found at {taxonomy_path} - treating all categories as equal specificity")

    print(f"\nLoading embeddings...")
    embeddings = np.load(args.embeddings)
    print(f"  Shape: {embeddings.shape}")

    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    id_to_idx = {entry["id"]: entry["index"] for entry in index}

    # Scan JSONL to collect labels for tagged recipes
    print(f"\nScanning JSONL for tagged recipes...")
    tagged    = {}
    ambiguous = 0

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r        = json.loads(line)
            tags     = set(r.get("tags") or [])
            matching = tags & cuisine_set
            if not matching:
                continue
            specific = most_specific_tag(matching, parent_of)
            if len(specific) == 1:
                tagged[r["id"]] = specific.pop()
            else:
                ambiguous += 1

    print(f"  {len(tagged):,} labeled  |  {ambiguous:,} skipped (ambiguous)")

    # Build training arrays
    train_ids = [rid for rid in tagged if rid in id_to_idx]
    X_train   = embeddings[[id_to_idx[rid] for rid in train_ids]]
    y_train   = np.array([tagged[rid] for rid in train_ids])

    print(f"\nTraining label distribution:")
    for label, count in Counter(y_train).most_common():
        print(f"  {label:<35} {count:>6,}")

    # Fit
    k = min(args.n_neighbors, len(X_train))
    print(f"\nFitting KNN (k={k}, metric=cosine, weights=distance)...")
    clf = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predict in batches for progress visibility
    print(f"Predicting on {len(index):,} recipes (batch size {PREDICT_BATCH_SIZE:,})...")
    all_probs = []
    for start in range(0, len(index), PREDICT_BATCH_SIZE):
        end = min(start + PREDICT_BATCH_SIZE, len(index))
        all_probs.append(clf.predict_proba(embeddings[start:end]))
        print(f"  {end:,} / {len(index):,}", flush=True)
    probs = np.vstack(all_probs)  # (N, n_classes)

    # For tagged recipes, replace self-inclusive probabilities with leave-one-out
    # (k+1 neighbors, drop self at distance 0, recompute weighted proba)
    print("Computing leave-one-out probabilities for tagged recipes...")
    class_to_idx   = {c: i for i, c in enumerate(clf.classes_)}
    tagged_row_idx = [id_to_idx[rid] for rid in train_ids if rid in id_to_idx]
    loo_distances, loo_neighbors = clf.kneighbors(X_train, n_neighbors=k + 1)
    loo_distances = loo_distances[:, 1:]   # drop self (distance 0)
    loo_neighbors = loo_neighbors[:, 1:]   # drop self

    for i, row_idx in enumerate(tagged_row_idx):
        dists   = loo_distances[i]
        weights = 1.0 / np.maximum(dists, 1e-10)
        weights /= weights.sum()
        proba_row = np.zeros(len(clf.classes_), dtype=np.float32)
        for w, ni in zip(weights, loo_neighbors[i]):
            proba_row[class_to_idx[y_train[ni]]] += w
        probs[row_idx] = proba_row

    # Prior correction: divide out class frequencies to remove imbalance bias.
    # KNN posteriors bake in the training distribution as an implicit prior.
    # Dividing by priors converts to a likelihood: P(x|class) ∝ P(class|x) / P(class).
    if not args.no_prior_correction:
        freq   = Counter(y_train)
        priors = np.array([freq[c] / len(y_train) for c in clf.classes_], dtype=np.float32)
        probs  = probs / priors
        probs /= probs.sum(axis=1, keepdims=True)
        print(f"Prior correction applied  (priors: { {c: round(freq[c]/len(y_train), 3) for c in clf.classes_} })")

    # Build output
    top3        = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_scores = probs[np.arange(len(probs))[:, None], top3]

    results = []
    for i, entry in enumerate(index):
        winning_score = float(top3_scores[i, 0])
        assigned = clf.classes_[top3[i, 0]] if winning_score >= args.min_score else None
        results.append({
            "id":       entry["id"],
            "category": assigned,
            "score":    round(winning_score, 4),
            "runners_up": [
                {"category": clf.classes_[top3[i, 1]], "score": round(float(top3_scores[i, 1]), 4)},
                {"category": clf.classes_[top3[i, 2]], "score": round(float(top3_scores[i, 2]), 4)},
            ],
        })

    base        = output_path(args.categories).replace(".json", "")
    out         = base + ".json"
    proba_out   = base + "_proba.npy"
    classes_out = base + "_classes.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"\nSaved {out}  ({len(results):,} entries)")

    np.save(proba_out, probs.astype(np.float32))
    print(f"Saved {proba_out}  shape={probs.shape}")

    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump(list(clf.classes_), f)
    print(f"Saved {classes_out}")

    dist = Counter(r["category"] for r in results)
    print(f"\nDistribution:")
    for label, count in dist.most_common():
        print(f"  {str(label):<35} {count:>6,}  ({count * 100 / len(results):.1f}%)")
    if args.min_score > 0 and None in dist:
        n_null = dist[None]
        print(f"\n  {n_null:,} below min-score {args.min_score} -> null ({n_null * 100 / len(results):.1f}%)")


if __name__ == "__main__":
    main()
