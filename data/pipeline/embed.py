"""
Embed recipes from RAW_recipes.jsonl using sentence-transformers.

Streams the JSONL, applies a configurable template to combine recipe fields
into a single text string, and encodes each recipe with all-MiniLM-L6-v2
(384-dim, L2-normalized). No checkpointing -- runs to completion.

Outputs:
  recipes_embeddings.npy   (N, 384) float32, L2-normalized
  recipes_index.json       [{"id": <recipe_id>, "index": <int>}, ...]

Usage:
    uv run pipeline/embed.py
    uv run pipeline/embed.py --max-rows 1000
    uv run pipeline/embed.py --batch-size 128
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "..", "raw", "RAW_recipes.jsonl")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "embed_config.json")
DEFAULT_OUTPUT_PREFIX = os.path.join(SCRIPT_DIR, "recipes")
MODEL_NAME = "all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to RAW_recipes.jsonl")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to embed_config.json")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for output files")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size (default: 64)")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most N recipes (for testing)")
    return parser.parse_args()


def render_template(template: str, record: dict) -> str:
    """Apply template string to a recipe record. Lists are joined with ', '; missing fields become ''."""
    fields = {}
    for key, val in record.items():
        if isinstance(val, list):
            fields[key] = ", ".join(str(x) for x in val if x is not None)
        elif val is None:
            fields[key] = ""
        else:
            fields[key] = str(val)
    result = template
    for m in re.finditer(r"\{(\w+)\}", template):
        key = m.group(1)
        result = result.replace(f"{{{key}}}", fields.get(key, ""))
    return result.strip()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for path in [args.input, args.config]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    template = config["template"]
    print(f"Template: {template}")

    print(f"\nReading {args.input}...")
    texts = []
    ids = []
    skipped = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            recipe_id = obj.get("id")
            if recipe_id is None:
                skipped += 1
                continue

            texts.append(render_template(template, obj))
            ids.append(recipe_id)

            if len(texts) % 10_000 == 0:
                print(f"  {len(texts):,} recipes read...", flush=True)

            if args.max_rows and len(texts) >= args.max_rows:
                break

    print(f"  {len(texts):,} recipes loaded ({skipped} skipped)")
    if not texts:
        print("No recipes found. Exiting.")
        sys.exit(1)

    print(f"\nLoading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Encoding {len(texts):,} recipes (batch_size={args.batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    embeddings_path = args.output_prefix + "_embeddings.npy"
    index_path = args.output_prefix + "_index.json"

    np.save(embeddings_path, embeddings)
    print(f"Saved {embeddings_path}  shape={embeddings.shape}")

    index = [{"id": rid, "index": i} for i, rid in enumerate(ids)]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"Saved {index_path}  ({len(index):,} entries)")

    print(f"\n{'='*50}")
    print(f"  Recipes embedded : {len(texts):,}")
    print(f"  Embedding dim    : {embeddings.shape[1]}")
    print(f"  Model            : {MODEL_NAME}")
    print(f"  Embeddings       : {embeddings_path}")
    print(f"  Index            : {index_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
