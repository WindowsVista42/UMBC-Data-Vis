"""
List all cuisine tags from RAW_recipes.jsonl with their recipe counts.

Identifies cuisine tags by finding all tags that exclusively appear on recipes
that also carry the "cuisine" parent tag. Filters to tags with at least
--min-count recipes. Use this to decide what goes in cuisines.txt.

Usage:
    uv run pipeline/cuisine_tags.py
    uv run pipeline/cuisine_tags.py --min-count 500
"""

import argparse
import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSONL = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_JSONL, help="Path to RAW_recipes.jsonl")
    parser.add_argument("--min-count", type=int, default=200, help="Minimum recipe count to include (default: 200)")
    return parser.parse_args()


def main():
    args = parse_args()

    total_counts = Counter()
    cuisine_counts = Counter()

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tags = r.get("tags") or []
            is_cuisine = "cuisine" in tags
            for tag in tags:
                if tag == "cuisine":
                    continue
                total_counts[tag] += 1
                if is_cuisine:
                    cuisine_counts[tag] += 1

    # Keep only tags that are 100% cuisine-tagged and meet the min count
    cuisine_tags = {
        tag: count
        for tag, count in cuisine_counts.items()
        if count == total_counts[tag] and count >= args.min_count
    }

    print(f"{'count':>7}  tag")
    print(f"{'-'*7}  {'-'*30}")
    for tag, count in sorted(cuisine_tags.items(), key=lambda x: -x[1]):
        print(f"{count:>7,}  {tag}")

    print(f"\n{len(cuisine_tags)} tags with >= {args.min_count} recipes")


if __name__ == "__main__":
    main()
