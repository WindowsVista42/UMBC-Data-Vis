#!/usr/bin/env python3
"""
Generate human-readable cluster names from paper titles using a local Ollama LLM.

Usage:
    python name_clusters.py <prefix> --jsonl <path> [--ollama-model llama3] [--samples 20]

Output:
    <prefix>_cluster_names.json  — {"2": "Quantum Optics", "-1": "Unclustered", ...}

Run this before export_web.py. The output is auto-detected by export_web.py.
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading

import requests

KAGGLE_DATASET  = "Cornell-University/arxiv"
KAGGLE_FILENAME = "arxiv-metadata-oai-snapshot.json"


def resolve_jsonl_path(args: argparse.Namespace) -> str:
    """Return the JSONL path, downloading from Kaggle if --kaggle was given."""
    if args.kaggle:
        try:
            import kagglehub
        except ImportError:
            print("ERROR: --kaggle requires the kagglehub package.")
            print("  Install with:  uv add kagglehub")
            sys.exit(1)

        import time
        print(f"Checking Kaggle for dataset: {KAGGLE_DATASET}...")
        t0 = time.time()
        dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
        elapsed = time.time() - t0
        if elapsed < 2.0:
            print(f"  Using cached dataset: {dataset_dir}")
        else:
            print(f"  Download complete ({elapsed/60:.1f} min): {dataset_dir}")

        jsonl_path = os.path.join(dataset_dir, KAGGLE_FILENAME)
        if not os.path.exists(jsonl_path):
            json_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(dataset_dir)
                for f in files if f.endswith(".json")
            ]
            if json_files:
                jsonl_path = json_files[0]
                print(f"  Expected '{KAGGLE_FILENAME}' not found, using: {jsonl_path}")
            else:
                print(f"  ERROR: No .json files found in {dataset_dir}")
                sys.exit(1)

        print(f"  Using: {jsonl_path} ({os.path.getsize(jsonl_path)/1e9:.2f} GB)")
        return jsonl_path

    if args.jsonl is None:
        print("ERROR: provide --jsonl <path> or use --kaggle to download from Kaggle.")
        sys.exit(1)
    if not os.path.exists(args.jsonl):
        print(f"ERROR: File not found: {args.jsonl}")
        sys.exit(1)
    return args.jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Name HDBSCAN clusters using an Ollama LLM"
    )
    parser.add_argument("prefix", help="Output prefix (e.g. arxiv-metadata-oai-snapshot)")
    parser.add_argument("--jsonl", default=None, help="Path to arXiv JSONL metadata file")
    parser.add_argument(
        "--kaggle", action="store_true",
        help=f"Download the arXiv dataset from Kaggle ({KAGGLE_DATASET}) instead of --jsonl"
    )
    parser.add_argument(
        "--ollama-model", default="llama3", metavar="MODEL",
        help="Ollama model name (default: llama3)"
    )
    parser.add_argument(
        "--samples", type=int, default=20, metavar="N",
        help="Number of titles to sample per cluster (default: 20)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file (default: <prefix>_cluster_names.json)"
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Re-name all clusters even if output file already exists"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, metavar="N",
        help="Number of parallel Ollama requests (default: 5)"
    )
    return parser.parse_args()


def ask_ollama(prompt: str, model: str, timeout: int = 120) -> str | None:
    """POST to Ollama generate API. Returns stripped response text or None on connection error.
    Raises requests.Timeout on timeout so the caller can retry."""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.Timeout:
        raise
    except requests.ConnectionError:
        return None
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)
        return None


MAX_NAME_WORDS = 6
MAX_RETRIES = 5


def name_single_cluster(
    cid: int,
    titles: list[str],
    model: str,
    n_samples: int,
) -> tuple[int, str | None]:
    """Name one cluster with retries for timeouts and too-long responses.

    Returns (cid, name) where name is None if Ollama is unreachable (connection error).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        sample = random.sample(titles, min(n_samples, len(titles)))
        titles_str = "; ".join(sample)
        prompt = (
            "These are titles of academic papers in the same research cluster. "
            "Give this cluster a concise 2-4 word name that describes the research area. "
            "Reply with only the name, nothing else.\n\n"
            f"Titles: {titles_str}"
        )

        try:
            result = ask_ollama(prompt, model)
        except requests.Timeout:
            if attempt < MAX_RETRIES:
                print(
                    f"  Cluster {cid}: timeout, retrying ({attempt}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                continue
            print(f"  Cluster {cid}: timed out after {MAX_RETRIES} attempts.", file=sys.stderr)
            return cid, f"Cluster {cid}"

        if result is None:
            # Connection error — Ollama is down; signal caller immediately.
            return cid, None

        name = result.strip().strip("\"'")
        word_count = len(name.split())
        if word_count > MAX_NAME_WORDS:
            if attempt < MAX_RETRIES:
                print(
                    f"  Cluster {cid}: name too long ({word_count} words), retrying "
                    f"({attempt}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                continue
            print(
                f"  Cluster {cid}: still too long after {MAX_RETRIES} attempts, using fallback.",
                file=sys.stderr,
            )
            return cid, f"Cluster {cid}"

        return cid, name

    return cid, f"Cluster {cid}"


def load_titles_by_cluster(
    jsonl_path: str, index: list[dict]
) -> dict[int, list[str]]:
    """Stream JSONL once, collecting titles keyed by cluster ID."""
    id_to_cluster: dict[str, int] = {entry["id"]: entry["cluster"] for entry in index}
    titles_by_cluster: dict[int, list[str]] = {}

    print("Reading titles from JSONL...", flush=True)
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            arxiv_id = record.get("id", "")
            if arxiv_id not in id_to_cluster:
                continue
            cid = id_to_cluster[arxiv_id]
            title = record.get("title", "").strip().replace("\n", " ")
            if title:
                titles_by_cluster.setdefault(cid, []).append(title)
            if lineno % 100_000 == 0:
                print(f"  {lineno:,} lines read...", flush=True)

    return titles_by_cluster


def generate_cluster_names(
    titles_by_cluster: dict[int, list[str]],
    model: str,
    n_samples: int,
    output_path: str,
    existing_names: dict[str, str],
    concurrency: int = 5,
) -> dict[str, str]:
    """Call Ollama for each cluster in parallel. Saves incrementally to output_path.

    Skips clusters already present in existing_names. Returns string-keyed dict of names.
    """
    names: dict[str, str] = dict(existing_names)
    all_cluster_ids = sorted(k for k in titles_by_cluster if k != -1)
    pending_ids = [cid for cid in all_cluster_ids if str(cid) not in names]
    total = len(all_cluster_ids)
    skipped = total - len(pending_ids)

    if skipped:
        print(f"  Skipping {skipped} already-named clusters (use --redo to rename all).")

    if not pending_ids:
        names.setdefault("-1", "Unclustered")
        return names

    ollama_down = threading.Event()
    completed = skipped
    lock = threading.Lock()

    def save_incremental() -> None:
        tmp = dict(names)
        tmp.setdefault("-1", "Unclustered")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, indent=2, ensure_ascii=False)

    def process(cid: int) -> tuple[int, str | None]:
        if ollama_down.is_set():
            return cid, None
        return name_single_cluster(cid, titles_by_cluster[cid], model, n_samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process, cid): cid for cid in pending_ids}

        for future in concurrent.futures.as_completed(futures):
            cid, name = future.result()
            n_papers = len(titles_by_cluster[cid])

            with lock:
                completed += 1
                idx = completed

                if name is None:
                    # Connection error — Ollama is unreachable.
                    if not ollama_down.is_set():
                        print(
                            "\nOllama unavailable — falling back to 'Cluster N' for remaining clusters.",
                            file=sys.stderr,
                        )
                        ollama_down.set()
                    names[str(cid)] = f"Cluster {cid}"
                else:
                    names[str(cid)] = name

                print(f"  [{idx}/{total}] Cluster {cid} ({n_papers:,} papers)... {names[str(cid)]}")
                save_incremental()

    # Fill in any clusters skipped due to Ollama going down mid-run.
    for cid in all_cluster_ids:
        names.setdefault(str(cid), f"Cluster {cid}")

    names["-1"] = "Unclustered"
    return names


def main() -> None:
    args = parse_args()

    index_path = f"{args.prefix}_umap3d_index.json"
    output_path = args.output or f"{args.prefix}_cluster_names.json"

    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    jsonl_path = resolve_jsonl_path(args)

    print(f"Loading {index_path}...")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    cluster_ids = {entry["cluster"] for entry in index}
    n_clusters = sum(1 for c in cluster_ids if c != -1)
    print(f"Found {len(index):,} points, {n_clusters} clusters (+ noise cluster -1)")

    titles_by_cluster = load_titles_by_cluster(jsonl_path, index)

    existing_names: dict[str, str] = {}
    if not args.redo and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing_names = json.load(f)
        print(f"Resuming from {output_path} ({len(existing_names)} entries already saved).")

    print(f"\nNaming {n_clusters} clusters with Ollama ({args.ollama_model}), "
          f"concurrency={args.concurrency}...")
    names = generate_cluster_names(
        titles_by_cluster,
        args.ollama_model,
        args.samples,
        output_path,
        existing_names,
        concurrency=args.concurrency,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)

    named = sum(1 for k, v in names.items() if k != "-1" and not v.startswith("Cluster "))
    print(f"\nWrote {len(names)} entries to {output_path}")
    print(f"  {named} LLM-named, {len(names) - named - 1} fallback, 1 unclustered")


if __name__ == "__main__":
    main()
