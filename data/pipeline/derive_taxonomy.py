"""
Derive parent-child relationships between cuisine tags from co-occurrence data.

For every pair of tags in cuisines.txt, checks what percentage of recipes
with tag A also have tag B. If A is always a subset of B, A is a child of B.

Outputs pipeline/cuisine_taxonomy.json for use by the classifier, and
pipeline/cuisine_taxonomy.txt as a human-readable report to cross-reference.

Usage:
    uv run pipeline/derive_taxonomy.py
"""

import json
import os
from collections import Counter, defaultdict

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH   = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")
CUISINES_TXT = os.path.join(SCRIPT_DIR, "cuisines.txt")
OUT_JSON     = os.path.join(SCRIPT_DIR, "cuisine_taxonomy.json")
OUT_TXT      = os.path.join(SCRIPT_DIR, "cuisine_taxonomy.txt")

# A tag is considered a child of another if this % of its recipes also have the parent
CHILD_THRESHOLD = 0.99


def load_cuisines():
    with open(CUISINES_TXT, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    cuisines = load_cuisines()
    cuisine_set = set(cuisines)

    # Count individual tags and co-occurrences
    tag_counts = Counter()
    co_counts  = defaultdict(Counter)  # co_counts[a][b] = recipes with both a and b

    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tags = set(json.loads(line).get("tags") or [])
            present = tags & cuisine_set
            for tag in present:
                tag_counts[tag] += 1
            for a in present:
                for b in present:
                    if a != b:
                        co_counts[a][b] += 1

    # Derive parent-child: A is child of B if nearly 100% of A's recipes also have B
    # and B has more recipes than A (B is more general)
    children  = defaultdict(list)   # parent -> [children]
    parent_of = {}                  # child  -> parent (most immediate)

    pairs = []
    for a in cuisines:
        for b in cuisines:
            if a == b:
                continue
            count_a  = tag_counts[a]
            count_b  = tag_counts[b]
            count_ab = co_counts[a][b]
            if count_a == 0:
                continue
            pct = count_ab / count_a
            if pct >= CHILD_THRESHOLD and count_b > count_a:
                pairs.append((a, b, count_a, count_b, count_ab, pct))

    # For each child, keep only the most immediate parent (highest count that still qualifies)
    # i.e. the smallest parent that still has more recipes than the child
    child_to_parents = defaultdict(list)
    for a, b, ca, cb, cab, pct in pairs:
        child_to_parents[a].append((cb, b))

    for child, parent_options in child_to_parents.items():
        # Most immediate parent = smallest count that is still larger than child
        parent_options.sort()  # sort by count ascending
        immediate_parent = parent_options[0][1]
        parent_of[child] = immediate_parent
        children[immediate_parent].append(child)

    # --- Write human-readable report ---
    lines = []
    lines.append("CUISINE TAXONOMY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Parent-child relationships (>= 99% co-occurrence):")
    lines.append("")

    # Print all qualifying pairs sorted by child count desc
    lines.append(f"  {'child':<35} {'parent':<30} {'child count':>11}  {'overlap':>10}")
    lines.append(f"  {'-'*35} {'-'*30} {'-'*11}  {'-'*10}")
    for a, b, ca, cb, cab, pct in sorted(pairs, key=lambda x: -x[2]):
        # Only show the immediate parent relationship
        if parent_of.get(a) == b:
            lines.append(f"  {a:<35} {b:<30} {ca:>11,}  {pct*100:>9.1f}%")

    lines.append("")
    lines.append("Derived hierarchy (immediate parents only):")
    lines.append("")

    # Print tree rooted at tags with no parent in the list
    roots = [c for c in cuisines if c not in parent_of]

    def print_tree(tag, indent=0):
        count = tag_counts[tag]
        lines.append(f"  {'  ' * indent}{tag} ({count:,})")
        for child in sorted(children.get(tag, []), key=lambda x: -tag_counts[x]):
            print_tree(child, indent + 1)

    for root in sorted(roots, key=lambda x: -tag_counts[x]):
        print_tree(root)

    lines.append("")
    lines.append(f"Roots (no parent in class list): {len(roots)}")
    lines.append(f"Children (have a parent):        {len(parent_of)}")

    report = "\n".join(lines)
    print(report)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved {OUT_TXT}")

    # --- Write JSON for classifier use ---
    taxonomy = {
        "parent_of": parent_of,
        "children":  {k: sorted(v) for k, v in children.items()},
        "counts":    {tag: tag_counts[tag] for tag in cuisines},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
