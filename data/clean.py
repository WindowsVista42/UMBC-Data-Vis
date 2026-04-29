"""
Remove all generated pipeline outputs from artifacts/ and export/.

Usage:
    uv run clean.py           # remove all outputs
    uv run clean.py --dry-run # preview what would be removed
"""

import argparse
import os
import shutil

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
EXPORT_DIR    = os.path.join(SCRIPT_DIR, "export")


def dir_size_mb(path):
    return sum(
        os.path.getsize(os.path.join(dp, f)) / 1e6
        for dp, _, files in os.walk(path)
        for f in files
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without deleting")
    args = parser.parse_args()

    targets = [(d, os.path.relpath(d, SCRIPT_DIR)) for d in [ARTIFACTS_DIR, EXPORT_DIR] if os.path.isdir(d)]

    if not targets:
        print("Nothing to remove.")
        return

    verb = "Would remove" if args.dry_run else "Removing"
    total_mb = 0.0

    for path, label in targets:
        mb = dir_size_mb(path)
        total_mb += mb
        print(f"{verb} {label}/  ({mb:.1f} MB)")
        if not args.dry_run:
            shutil.rmtree(path)

    print(f"\nTotal: {total_mb:.1f} MB")
    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
