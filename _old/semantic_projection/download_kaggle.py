#!/usr/bin/env python3
"""
Download the arXiv metadata dataset from Kaggle and copy it to the current directory.

Usage:
    uv run download_kaggle.py

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
Get them from: https://www.kaggle.com/settings -> API -> Create New Token
"""

import os
import shutil
import sys
import time

KAGGLE_DATASET  = "Cornell-University/arxiv"
KAGGLE_FILENAME = "arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH     = KAGGLE_FILENAME


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("  Install with:  uv add kagglehub")
        sys.exit(1)

    if os.path.exists(OUTPUT_PATH):
        size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
        print(f"Already exists: {OUTPUT_PATH} ({size_gb:.2f} GB)")
        print("Delete it first if you want to re-download.")
        sys.exit(0)

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    print("(kagglehub will use its cache if already downloaded)\n")

    t0 = time.time()
    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    elapsed = time.time() - t0

    if elapsed < 2.0:
        print(f"Using cached dataset at: {dataset_dir}")
    else:
        print(f"Download complete ({elapsed / 60:.1f} min): {dataset_dir}")

    # Find the JSONL file
    src = os.path.join(dataset_dir, KAGGLE_FILENAME)
    if not os.path.exists(src):
        json_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(dataset_dir)
            for f in files if f.endswith(".json")
        ]
        if not json_files:
            print(f"ERROR: No .json files found in {dataset_dir}")
            print(f"Contents: {os.listdir(dataset_dir)}")
            sys.exit(1)
        src = json_files[0]
        print(f"Expected '{KAGGLE_FILENAME}' not found, using: {src}")

    print(f"\nCopying to {OUTPUT_PATH} ...")
    shutil.copy2(src, OUTPUT_PATH)
    size_gb = os.path.getsize(OUTPUT_PATH) / 1e9
    print(f"Done. {OUTPUT_PATH} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
