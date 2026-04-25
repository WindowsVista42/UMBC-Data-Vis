# -*- coding: utf-8 -*-
"""
Embed arXiv paper titles + abstracts from a large JSONL file using allenai/specter2.

Streams the JSONL line-by-line (memory-safe for multi-GB files), batches text
through the model on GPU, and writes two output files:

  1. <base>_embeddings.npy  — (N, 768) float32 array of embedding vectors
  2. <base>_index.json      — list of {"id": "<arxiv_id>", "index": <int>}

DATA SOURCE: Pass --kaggle to download the arXiv dataset directly from Kaggle
(requires kagglehub + Kaggle API credentials). Otherwise, pass a local JSONL path.

SUPPORTS STOP/RESUME: progress is checkpointed every --checkpoint-every entries.
If interrupted (Ctrl-C, crash, etc.), re-run the same command and it picks up
where it left off. A .raw file accumulates embeddings and a small checkpoint
JSON tracks how far we got. Use --restart to force a clean start.

The text sent to the model is:  "<title> [SEP] <abstract>"
(SPECTER2 was trained with [SEP]-delimited title+abstract input.)

Usage:
    python embed_abstracts.py --kaggle                      # download from Kaggle
    python embed_abstracts.py --kaggle --max-rows 10000     # Kaggle + test run
    python embed_abstracts.py data.jsonl                    # local file
    python embed_abstracts.py data.jsonl --batch-size 256
    python embed_abstracts.py data.jsonl --restart           # ignore checkpoint

Requirements:
    uv add transformers torch numpy adapters   # for SPECTER2 (default)
    uv add sentence-transformers               # only if using a different model
    uv add kagglehub                           # only if using --kaggle
"""

import os
import sys
import json
import argparse
import time
import signal
import numpy as np
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

# -- Config --------------------------------------------------------------------

DEFAULT_MODEL = "allenai/specter2"
EMBEDDING_DIM = 768
BATCH_SIZE = 128
CHECKPOINT_EVERY = 10_000  # flush to disk every N entries

# -- Args ----------------------------------------------------------------------

KAGGLE_DATASET = "Cornell-University/arxiv"
KAGGLE_FILENAME = "arxiv-metadata-oai-snapshot.json"

parser = argparse.ArgumentParser(
    description="Embed arXiv titles+abstracts from JSONL with SPECTER2 (resumable)"
)
parser.add_argument("jsonl", nargs="?", default=None,
                    help="Path to the JSONL file (one JSON object per line). "
                         "Not needed if --kaggle is used.")
parser.add_argument("--kaggle", action="store_true",
                    help=f"Download the arXiv dataset from Kaggle ({KAGGLE_DATASET}) "
                         "instead of using a local file. Requires kagglehub and "
                         "Kaggle API credentials (~/.kaggle/kaggle.json).")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"Encoding batch size (default: {BATCH_SIZE})")
parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                    help=f"Checkpoint every N entries (default: {CHECKPOINT_EVERY})")
parser.add_argument("--output-prefix", default=None,
                    help="Prefix for output files (default: derived from input filename)")
parser.add_argument("--max-rows", type=int, default=None,
                    help="Process at most this many rows (useful for testing)")
parser.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"SentenceTransformers model name (default: {DEFAULT_MODEL})")
parser.add_argument("--restart", action="store_true",
                    help="Ignore existing checkpoint and start from scratch")
args = parser.parse_args()

# -- Resolve input file --------------------------------------------------------

if args.kaggle:
    try:
        import kagglehub
    except ImportError:
        print("ERROR: --kaggle requires the kagglehub package.")
        print("  Install with:  pip install kagglehub")
        sys.exit(1)

    print(f"Checking Kaggle for dataset: {KAGGLE_DATASET}...")
    print(f"  (kagglehub will download if not already cached)")
    t_dl = time.time()
    dataset_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    dl_elapsed = time.time() - t_dl
    if dl_elapsed < 2.0:
        print(f"  Using cached dataset: {dataset_dir}")
    else:
        print(f"  Download complete ({dl_elapsed/60:.1f} min): {dataset_dir}")

    # Find the JSONL file in the downloaded directory
    jsonl_path = os.path.join(dataset_dir, KAGGLE_FILENAME)
    if not os.path.exists(jsonl_path):
        # Search for any .json file in the directory
        json_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for fname in files:
                if fname.endswith(".json"):
                    json_files.append(os.path.join(root, fname))
        if json_files:
            jsonl_path = json_files[0]
            print(f"  Expected '{KAGGLE_FILENAME}' not found, using: {jsonl_path}")
        else:
            print(f"  ERROR: No .json files found in {dataset_dir}")
            print(f"  Contents: {os.listdir(dataset_dir)}")
            sys.exit(1)

    print(f"  Using: {jsonl_path} ({os.path.getsize(jsonl_path)/1e9:.2f} GB)")
    args.jsonl = jsonl_path

elif args.jsonl is None:
    parser.error("Please provide a JSONL file path, or use --kaggle to download from Kaggle.")

if not os.path.exists(args.jsonl):
    print(f"ERROR: File not found: {args.jsonl}")
    sys.exit(1)

if args.output_prefix:
    base = args.output_prefix
else:
    base = os.path.splitext(os.path.basename(args.jsonl))[0]

embed_path      = base + "_embeddings.npy"
index_path      = base + "_index.json"
checkpoint_path = base + "_checkpoint.json"
embed_raw_path  = base + "_embeddings.raw"  # incremental raw float32 file

# -- Graceful shutdown ---------------------------------------------------------

shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        print("\n  Force quitting...")
        sys.exit(1)
    shutdown_requested = True
    print("\n  Shutdown requested — finishing current batch and saving checkpoint...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# -- Checkpoint helpers --------------------------------------------------------

def load_checkpoint():
    """Load checkpoint. Returns dict with lines_to_skip, ids, n_embedded or None."""
    if args.restart:
        clean_checkpoint()
        return None
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        n_embedded = ckpt["n_embedded"]
        ids_done = ckpt["ids"]
        valid_entries_seen = ckpt["valid_entries_seen"]

        # Verify raw file has the right number of bytes
        expected_bytes = n_embedded * EMBEDDING_DIM * 4
        if not os.path.exists(embed_raw_path):
            print("  Checkpoint found but raw embeddings file missing — restarting.")
            clean_checkpoint()
            return None
        actual_bytes = os.path.getsize(embed_raw_path)
        if actual_bytes < expected_bytes:
            print(f"  Raw file truncated ({actual_bytes} < {expected_bytes}) — restarting.")
            clean_checkpoint()
            return None
        elif actual_bytes > expected_bytes:
            print(f"  Truncating {actual_bytes - expected_bytes} excess bytes from raw file.")
            with open(embed_raw_path, "r+b") as f:
                f.truncate(expected_bytes)

        print(f"  Resuming from checkpoint: {n_embedded:,} entries already embedded.")
        return {
            "ids": ids_done,
            "n_embedded": n_embedded,
            "valid_entries_seen": valid_entries_seen,
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  Corrupt checkpoint ({e}) — restarting.")
        clean_checkpoint()
        return None


def save_checkpoint(ids, n_embedded, valid_entries_seen):
    """Save checkpoint atomically."""
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "n_embedded": n_embedded,
            "valid_entries_seen": valid_entries_seen,
            "ids": ids,
            "model": args.model,
            "embedding_dim": EMBEDDING_DIM,
        }, f)
    os.replace(tmp, checkpoint_path)


def clean_checkpoint():
    """Remove checkpoint and raw files for a fresh start."""
    for p in [checkpoint_path, embed_raw_path]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed {p}")


# -- Load checkpoint state -----------------------------------------------------

print(f"Checking for existing checkpoint...")
ckpt = load_checkpoint()

if ckpt:
    ids = list(ckpt["ids"])
    n_already_embedded = ckpt["n_embedded"]
    skip_valid_entries = ckpt["valid_entries_seen"]
else:
    ids = []
    n_already_embedded = 0
    skip_valid_entries = 0

# -- Stream JSONL and embed on-the-fly ----------------------------------------
#
# Instead of two passes (parse all then embed), we do a single streaming pass:
#   - Skip entries we've already checkpointed
#   - Accumulate a batch of texts
#   - Encode the batch and append to raw file
#   - Checkpoint periodically
#
# This means we only hold one batch of texts in memory at a time.

print(f"\nStreaming '{args.jsonl}'...")
if n_already_embedded > 0:
    print(f"  Skipping first {skip_valid_entries:,} valid entries (already embedded)...")

# Load model (only if there's work to do — we check after scanning)
_model = None
_tokenizer = None
_use_adapters = False  # True for SPECTER2, False for SentenceTransformer models

SPECTER2_MODELS = {"allenai/specter2", "allenai/specter2_proximity",
                   "allenai/specter2_classification", "allenai/specter2_regression",
                   "allenai/specter2_adhoc_query"}

def get_model():
    global _model, _tokenizer, _use_adapters
    if _model is not None:
        return

    print(f"\n  Loading model '{args.model}'...")
    print(f"  (first run will download model weights — may take a minute)")
    t_model = time.time()

    if args.model in SPECTER2_MODELS:
        # SPECTER2 requires the adapters library, not SentenceTransformer
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        _use_adapters = True
        _tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        _model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        _model.load_adapter(args.model, source="hf", load_as="specter2", set_active=True)
        _model = _model.to("cuda")
        _model.eval()
        device_name = "cuda"
    else:
        from sentence_transformers import SentenceTransformer
        _use_adapters = False
        _model = SentenceTransformer(args.model, device="cuda")
        device_name = str(_model.device)

    model_elapsed = time.time() - t_model
    print(f"  Model ready on {device_name} ({model_elapsed:.1f}s)")


def encode_batch(texts):
    """Encode a batch of texts, returns (N, EMBEDDING_DIM) float32 numpy array."""
    import torch

    if _use_adapters:
        inputs = _tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", return_token_type_ids=False, max_length=512,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            output = _model(**inputs)
        # Mean pooling over token embeddings (mask out padding)
        token_embs = output.last_hidden_state  # (batch, seq_len, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (token_embs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embs = (summed / counts).cpu().numpy().astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embs = embs / norms
        return embs
    else:
        return _model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

t0 = time.time()
valid_count = 0      # total valid entries seen in file so far
new_embedded = 0     # new entries embedded this session
skipped_invalid = 0
batch_texts = []
batch_ids = []

fout = open(embed_raw_path, "ab")  # append mode

try:
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if shutdown_requested:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            arxiv_id = obj.get("id")
            title = obj.get("title", "").strip()
            abstract = obj.get("abstract", "").strip()

            if not arxiv_id or not abstract:
                skipped_invalid += 1
                continue

            valid_count += 1

            # Skip entries we already embedded
            if valid_count <= skip_valid_entries:
                continue

            # SPECTER2 format: title [SEP] abstract
            # (BERT's sep_token is [SEP], so this works for both code paths)
            if title:
                text = f"{title} [SEP] {abstract}"
            else:
                text = abstract

            batch_texts.append(text)
            batch_ids.append(arxiv_id)

            # When batch is full, encode it
            if len(batch_texts) >= args.batch_size:
                get_model()
                embs = encode_batch(batch_texts)
                fout.write(embs.astype(np.float32).tobytes())
                ids.extend(batch_ids)
                new_embedded += len(batch_texts)
                batch_texts.clear()
                batch_ids.clear()

                total_done = n_already_embedded + new_embedded

                # Periodic checkpoint
                if new_embedded % args.checkpoint_every < args.batch_size:
                    fout.flush()
                    os.fsync(fout.fileno())
                    save_checkpoint(ids, total_done, valid_count)
                    elapsed = time.time() - t0
                    rate = new_embedded / elapsed if elapsed > 0 else 0
                    print(f"  {total_done:,} embedded "
                          f"- {rate:.0f} texts/s "
                          f"- elapsed {elapsed/60:.1f} min "
                          f"[checkpoint saved]", flush=True)

                elif new_embedded % 1_000 < args.batch_size:
                    elapsed = time.time() - t0
                    rate = new_embedded / elapsed if elapsed > 0 else 0
                    print(f"  {total_done:,} embedded "
                          f"- {rate:.0f} texts/s "
                          f"- elapsed {elapsed/60:.1f} min", flush=True)

            # Check max rows
            if args.max_rows and (n_already_embedded + new_embedded + len(batch_texts)) >= args.max_rows:
                # Flush remaining batch
                if batch_texts:
                    get_model()
                    embs = encode_batch(batch_texts)
                    fout.write(embs.astype(np.float32).tobytes())
                    ids.extend(batch_ids)
                    new_embedded += len(batch_texts)
                    batch_texts.clear()
                    batch_ids.clear()
                break

    # Flush any remaining partial batch
    if batch_texts and not shutdown_requested:
        get_model()
        embs = encode_batch(batch_texts)
        fout.write(embs.astype(np.float32).tobytes())
        ids.extend(batch_ids)
        new_embedded += len(batch_texts)
        batch_texts.clear()
        batch_ids.clear()

    fout.flush()
    os.fsync(fout.fileno())

finally:
    fout.close()

total_embedded = n_already_embedded + new_embedded

# -- Handle shutdown -----------------------------------------------------------

if shutdown_requested:
    save_checkpoint(ids, total_embedded, valid_count)
    print(f"\n  Checkpoint saved: {total_embedded:,} entries embedded.")
    print(f"  Re-run the same command to resume.")
    sys.exit(0)

# -- If nothing new was embedded -----------------------------------------------

if new_embedded == 0 and n_already_embedded == 0:
    print("No valid entries found. Exiting.")
    clean_checkpoint()
    sys.exit(1)

elapsed = time.time() - t0
if new_embedded > 0:
    print(f"\n  Session complete: {new_embedded:,} new vectors in {elapsed:.1f}s "
          f"({new_embedded/elapsed:.0f} texts/s)")
print(f"  Total embedded: {total_embedded:,}")

# -- Convert raw -> .npy -------------------------------------------------------

print(f"\nConverting raw embeddings to .npy...")

raw_bytes = os.path.getsize(embed_raw_path)
expected_bytes = total_embedded * EMBEDDING_DIM * 4
assert raw_bytes == expected_bytes, \
    f"Raw file size mismatch: {raw_bytes} vs expected {expected_bytes}"

raw_data = np.memmap(embed_raw_path, dtype=np.float32, mode="r",
                     shape=(total_embedded, EMBEDDING_DIM))
np.save(embed_path, raw_data)
del raw_data

# Verify
embeddings_check = np.load(embed_path, mmap_mode="r")
assert embeddings_check.shape == (total_embedded, EMBEDDING_DIM), \
    f"Shape mismatch: expected ({total_embedded}, {EMBEDDING_DIM}), got {embeddings_check.shape}"
print(f"  Saved -> {embed_path}  ({os.path.getsize(embed_path)/1e9:.2f} GB)")

# -- Write index file ----------------------------------------------------------

print(f"Writing index file...")
index_data = [{"id": aid, "index": i} for i, aid in enumerate(ids)]

with open(index_path, "w", encoding="utf-8") as f:
    json.dump(index_data, f, separators=(",", ":"))

print(f"  Saved -> {index_path}  ({os.path.getsize(index_path)/1e6:.1f} MB)")

# -- Clean up checkpoint -------------------------------------------------------

print("Cleaning up checkpoint files...")
clean_checkpoint()

# -- Summary -------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"  Entries embedded : {total_embedded:,}")
print(f"  Embedding dim    : {EMBEDDING_DIM}")
print(f"  Model            : {args.model}")
print(f"  Input format     : title [SEP] abstract")
print(f"  Embeddings file  : {embed_path}")
print(f"  Index file       : {index_path}")
print(f"{'='*60}")
print(f"\nTo load:")
print(f"  import numpy as np, json")
print(f"  embeddings = np.load('{embed_path}')           # shape ({total_embedded}, {EMBEDDING_DIM})")
print(f"  index = json.load(open('{index_path}'))        # [{{'id':..., 'index':...}}, ...]")
print(f"  vec = embeddings[index[0]['index']]             # embedding for first paper")
