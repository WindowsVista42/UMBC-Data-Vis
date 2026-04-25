#!/usr/bin/env python3
"""
Export UMAP 3D point cloud data to a static website for hosting on GitHub Pages,
Cloudflare Pages, or any static host.

Usage:
    python export_web.py <prefix> --jsonl <path> [options]

Example:
    python export_web.py arxiv-metadata-oai-snapshot \\
        --jsonl path/to/arxiv-metadata-oai-snapshot.jsonl

Then serve with:
    cd site && python -m http.server 8080
"""

import argparse
import gzip
import json
import math
import os
import shutil
import sys

import numpy as np

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
        description="Export arXiv 3D point cloud to a static website"
    )
    parser.add_argument(
        "prefix",
        help="Output prefix (e.g. arxiv-metadata-oai-snapshot)",
    )
    parser.add_argument(
        "--jsonl", default=None,
        help="Path to arXiv JSONL metadata file (for titles, abstracts, DOIs)",
    )
    parser.add_argument(
        "--kaggle", action="store_true",
        help=f"Download the arXiv dataset from Kaggle ({KAGGLE_DATASET}) instead of --jsonl",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, metavar="N",
        help="Points per chunk file (default: 500, ~150-200KB/chunk compressed)",
    )
    parser.add_argument(
        "--output-dir", default="site", metavar="DIR",
        help="Output directory (default: site)",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize coordinates to [-0.5, 0.5]³ (centers cloud and fixes scale for default camera)",
    )
    parser.add_argument(
        "--cluster-names", default=None, metavar="FILE",
        help=(
            "Path to cluster names JSON (default: auto-detect <prefix>_cluster_names.json). "
            "Falls back to 'Cluster N' if not found."
        ),
    )
    return parser.parse_args()


def load_jsonl_metadata(jsonl_path: str) -> dict[str, dict]:
    """Stream JSONL, return {arxiv_id: {title, abstract, doi}}. Never loads full file into memory."""
    metadata: dict[str, dict] = {}
    print("Reading metadata from JSONL...", flush=True)
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  Warning: JSON parse error at line {lineno}, skipping", file=sys.stderr)
                continue
            arxiv_id = record.get("id", "")
            if not arxiv_id:
                continue
            metadata[arxiv_id] = {
                "title":    record.get("title", "").strip().replace("\n", " "),
                "abstract": record.get("abstract", "").strip().replace("\n", " "),
                "doi":      record.get("doi") or None,
            }
            if lineno % 100_000 == 0:
                print(f"  {lineno:,} lines read...", flush=True)
    print(f"  Loaded metadata for {len(metadata):,} papers")
    return metadata


def resolve_doi(doi_field: str | None, arxiv_id: str) -> str:
    """Return a full URL: journal DOI if available, otherwise arXiv abstract page."""
    if doi_field and doi_field.strip():
        doi = doi_field.strip()
        if doi.startswith("http"):
            return doi
        return f"https://doi.org/{doi}"
    return f"https://arxiv.org/abs/{arxiv_id}"


def recenter_coords(coords: np.ndarray) -> np.ndarray:
    """Shift coordinates so their center of mass is at the origin."""
    return (coords - coords.mean(axis=0)).astype(np.float32)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize (N, 3) float array to [-0.5, 0.5] using a single uniform scale factor
    so that spatial ratios are preserved (same approach as text_umap_3d_v10_OLD.py lines 574-586).
    """
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
    scale = float(np.ptp(coords, axis=0).max())
    if scale == 0.0:
        scale = 1.0
    return ((coords - center) / scale).astype(np.float32)


def build_points(
    umap3d_index: list[dict],
    coords_norm: np.ndarray,
    metadata: dict[str, dict],
) -> list[dict]:
    """Merge coordinates, cluster IDs, and JSONL metadata into per-point dicts."""
    points = []
    missing = 0
    for entry in umap3d_index:
        arxiv_id = entry["id"]
        idx      = entry["index"]
        cluster  = entry["cluster"]
        meta = metadata.get(arxiv_id)
        if meta is None:
            missing += 1
            meta = {"title": "", "abstract": "", "doi": None}

        x, y, z = coords_norm[idx]
        points.append({
            "id":       arxiv_id,
            "x":        round(float(x), 6),
            "y":        round(float(y), 6),
            "z":        round(float(z), 6),
            "cluster":  cluster,
            "title":    meta["title"],
            "abstract": meta["abstract"],
            "doi":      resolve_doi(meta["doi"], arxiv_id),
        })

    if missing:
        print(
            f"  Warning: {missing:,} points had no JSONL metadata "
            "(will have empty title/abstract)"
        )
    return points


def write_chunks(points: list[dict], output_dir: str, chunk_size: int) -> int:
    """Write per-chunk binary geometry and gzip'd text. Returns number of chunks.

    Each chunk N produces two files:
      chunk_XXXXXX.bin.gz  — gzip of float32[N×3] positions + int32[N] cluster IDs
      chunk_XXXXXX.json.gz — gzip of JSON [{id, title, abstract, doi}, ...]

    Byte layout of the raw (pre-gzip) binary:
      bytes 0 .. N*12-1  : N × 3 × float32  (x,y,z interleaved, little-endian)
      bytes N*12 .. N*16-1: N × int32        (cluster IDs, little-endian)
    """
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    n_chunks = math.ceil(len(points) / chunk_size)
    for i in range(n_chunks):
        chunk = points[i * chunk_size : (i + 1) * chunk_size]

        # ── Binary geometry ────────────────────────────────────────────────────
        positions = np.array(
            [[p["x"], p["y"], p["z"]] for p in chunk], dtype="<f4"   # little-endian float32
        )
        clusters = np.array([p["cluster"] for p in chunk], dtype="<i4")  # little-endian int32
        bin_raw   = positions.tobytes() + clusters.tobytes()
        bin_bytes = gzip.compress(bin_raw, compresslevel=9)

        bin_path = os.path.join(chunks_dir, f"chunk_{i:06d}.bin.gz")
        with open(bin_path, "wb") as f:
            f.write(bin_bytes)

        # ── Text metadata ──────────────────────────────────────────────────────
        text_records = [
            {"id": p["id"], "title": p["title"], "abstract": p["abstract"], "doi": p["doi"]}
            for p in chunk
        ]
        text_raw   = json.dumps(text_records, ensure_ascii=False, separators=(",", ":")).encode()
        text_bytes = gzip.compress(text_raw, compresslevel=9)

        text_path = os.path.join(chunks_dir, f"chunk_{i:06d}.json.gz")
        with open(text_path, "wb") as f:
            f.write(text_bytes)

    return n_chunks


def write_meta(
    output_dir: str,
    total: int,
    n_chunks: int,
    cluster_names: dict[str, str],
) -> None:
    meta = {
        "total":         total,
        "chunks":        n_chunks,
        "chunk_format":  "bin+gz",   # chunk_XXXXXX.bin.gz + chunk_XXXXXX.json.gz
        "cluster_names": cluster_names,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))


def copy_web_files(web_dir: str, output_dir: str) -> None:
    for fname in ("index.html", "style.css", "app.js"):
        src = os.path.join(web_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    index_path = f"{args.prefix}_umap3d_index.json"
    npy_path   = f"{args.prefix}_umap3d.npy"
    web_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

    jsonl_path = resolve_jsonl_path(args)

    # Validate inputs
    for path in (index_path, npy_path):
        if not os.path.exists(path):
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
    for fname in ("index.html", "style.css", "app.js"):
        if not os.path.exists(os.path.join(web_dir, fname)):
            print(f"Error: web/{fname} not found", file=sys.stderr)
            sys.exit(1)

    # Load cluster names (optional)
    names_path = args.cluster_names or f"{args.prefix}_cluster_names.json"
    cluster_names: dict[str, str] = {}
    if os.path.exists(names_path):
        print(f"Loading cluster names from {names_path}...")
        with open(names_path, encoding="utf-8") as f:
            cluster_names = json.load(f)
    else:
        print(f"No cluster names file found at {names_path} — will use 'Cluster N' labels")
        print("  Run name_clusters.py first to generate LLM-based names.")

    # Load index and coordinates
    print(f"Loading {index_path}...")
    with open(index_path, encoding="utf-8") as f:
        umap3d_index: list[dict] = json.load(f)

    print(f"Loading {npy_path}...")
    coords = np.load(npy_path)

    # Load JSONL metadata
    metadata = load_jsonl_metadata(jsonl_path)

    if args.normalize:
        print("Normalizing coordinates to [-0.5, 0.5]³...")
        coords_norm = normalize_coords(coords)
    else:
        print("Recentering coordinates to origin (center of mass)...")
        coords_norm = recenter_coords(coords)

    # Ensure all cluster IDs have a name
    all_cluster_ids = {entry["cluster"] for entry in umap3d_index}
    for cid in sorted(all_cluster_ids):
        key = str(cid)
        if key not in cluster_names:
            cluster_names[key] = "Unclustered" if cid == -1 else f"Cluster {cid}"

    # Build merged point list
    print(f"Merging {len(umap3d_index):,} points...")
    points = build_points(umap3d_index, coords_norm, metadata)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Writing {math.ceil(len(points) / args.chunk_size)} chunks "
          f"(size {args.chunk_size}) to {args.output_dir}/chunks/...")
    n_chunks = write_chunks(points, args.output_dir, args.chunk_size)

    print("Writing meta.json...")
    write_meta(args.output_dir, len(points), n_chunks, cluster_names)

    print("Copying web files...")
    copy_web_files(web_dir, args.output_dir)

    # Summary
    chunks_dir = os.path.join(args.output_dir, "chunks")
    bin_mb  = sum(
        os.path.getsize(os.path.join(chunks_dir, f))
        for f in os.listdir(chunks_dir) if f.endswith(".bin.gz")
    ) / (1024 * 1024)
    text_mb = sum(
        os.path.getsize(os.path.join(chunks_dir, f))
        for f in os.listdir(chunks_dir) if f.endswith(".json.gz")
    ) / (1024 * 1024)
    total_mb = bin_mb + text_mb

    n_named = sum(
        1 for k, v in cluster_names.items()
        if k != "-1" and not v.startswith("Cluster ")
    )

    print(f"""
Done!
  {len(points):,} points  →  {n_chunks} chunk pairs  ({total_mb:.1f} MB total)
    geometry (.bin.gz):  {bin_mb:.1f} MB
    text     (.json.gz): {text_mb:.1f} MB
  {len(all_cluster_ids)} clusters  ({n_named} LLM-named)
  Output: {os.path.abspath(args.output_dir)}/

To preview locally (must use HTTP server, not file://):
  cd {args.output_dir} && python -m http.server 8080
  Then open http://localhost:8080
""")


if __name__ == "__main__":
    main()
