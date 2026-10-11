#!/usr/bin/env python3
"""Call the unchanged V5 packer using a source-only view of the public package.

The original server layout kept outputs outside its code tree. The portable
layout permits ./runs; staging prevents such outputs being mistaken for source
files by the original packer's recursive source scan. No experiment is altered.
"""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pack_feedback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    blocked_parts = {".git", "runs", "__pycache__", ".pytest_cache", ".venv", "venv", "vendor"}
    suffixes = {".py", ".sh", ".txt", ".md", ".json", ".csv"}
    with tempfile.TemporaryDirectory(prefix="hcxmd-code-snapshot-") as temporary:
        source_view = Path(temporary)
        for source in ROOT.rglob("*"):
            rel = source.relative_to(ROOT)
            if not source.is_file() or source.is_symlink() or blocked_parts.intersection(rel.parts):
                continue
            if source.suffix.lower() not in suffixes or source.resolve().is_relative_to(output_root):
                continue
            target = source_view / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        pack_feedback.ROOT = source_view
        sys.argv = ["pack_feedback.py", "--outdir", str(args.outdir.resolve())]
        pack_feedback.main()


if __name__ == "__main__":
    main()
