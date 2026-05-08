"""
Full pipeline runner. Reads config.json for the list of category and
encoding steps, then runs every step in order.

Usage:
    uv run run.py                        # full run
    uv run run.py --from embed           # skip download, start from embedding
    uv run run.py --from project         # skip to UMAP projection
    uv run run.py --config my_config.json
    uv run run.py --no-log               # disable log file

Steps (in order):
    download  - download dataset from Kaggle
    embed     - embed recipes as 384-dim vectors
    assign    - classify by each category file listed in config.json
    ratings   - compute per-recipe rating aggregates
    encode    - encode numeric/tag features listed in config.json
    metrics   - generate per-recipe and per-category metric JSON files
    project   - UMAP projection
    export    - package for web app and copy to site/data/
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_LOG    = os.path.join(SCRIPT_DIR, "pipeline.log")

STEPS = ["download", "embed", "assign", "ratings", "encode", "metrics", "project", "export"]


class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


def run(cmd):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[{ts}] >>> {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        cmd, cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        print(f"ERROR: step failed (exit {process.returncode})")
        sys.exit(process.returncode)


def uv(*args):
    return ["uv", "run", *args]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start_from", default="download",
                        choices=STEPS, help="Start from this step (default: download)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to pipeline config JSON (default: config.json)")
    parser.add_argument("--no-log", dest="no_log", action="store_true",
                        help="Disable log file (default: writes to pipeline.log)")
    parser.add_argument("--log", default=DEFAULT_LOG,
                        help=f"Log file path (default: {DEFAULT_LOG})")
    return parser.parse_args()


def main():
    args = parse_args()

    log_file = None
    if not args.no_log:
        log_file = open(args.log, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        print(f"Logging to: {args.log}", flush=True)

    try:
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            sys.exit(1)

        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)

        embed_config       = config.get("embed_config",       "pipeline/configs/embed_config.json")
        projection_weights = config.get("projection_weights", "pipeline/configs/projection_weights.json")
        assign_files       = config.get("assign", [])
        encode_steps       = config.get("encode", [])

        required = [
            embed_config,
            projection_weights,
            *assign_files,
            *[step["config"] for step in encode_steps],
        ]
        missing = [p for p in required if not os.path.exists(os.path.join(SCRIPT_DIR, p))]
        if missing:
            print("ERROR: Missing required config files:")
            for p in missing:
                print(f"  {p}")
            sys.exit(1)

        start  = STEPS.index(args.start_from)
        active = lambda step: STEPS.index(step) >= start

        print(f"Pipeline config: {args.config}")
        print(f"Starting from:   {args.start_from}")

        if active("download"):
            run(uv("download.py"))

        if active("embed"):
            run(uv("pipeline/embed.py", "--config", embed_config))

        if active("assign"):
            for txt_file in assign_files:
                run(uv("pipeline/derive_taxonomy.py", txt_file))
                run(uv("pipeline/assign.py",          txt_file))

        if active("ratings"):
            run(uv("pipeline/process_ratings.py"))

        if active("encode"):
            for step in encode_steps:
                cmd = uv("pipeline/encode_ordinal.py", step["config"])
                if "contrib" in step:
                    cmd += ["--contrib", step["contrib"]]
                run(cmd)

        if active("metrics"):
            run(uv("pipeline/metrics.py"))

        if active("project"):
            run(uv("pipeline/project.py", "--weights", projection_weights))

        if active("export"):
            run(uv("pipeline/export.py"))

        print("\nAll steps complete.")

    finally:
        if log_file:
            sys.stdout = sys.__stdout__
            log_file.close()


if __name__ == "__main__":
    main()
