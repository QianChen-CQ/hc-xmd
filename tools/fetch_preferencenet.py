#!/usr/bin/env python3
"""Retrieve the exact external PreferenceNet snapshot; never add it to Git.

The upstream snapshot has no explicit license. Its three files are not
redistributed in this repository. This tool retrieves them from their author,
or verifies a user's existing local copy, and checks all recorded SHA-256 values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1419a4654a4c6a5aabffd7eb2a89d077979cd965"
BASE_URL = "https://raw.githubusercontent.com/neeharperi/PreferenceNet/" + COMMIT + "/"
FILES = {
    "README.md": "fe90fd38ddca03adfb1c2ed203f16d16bb948e2bd63e76e45890028a75fdfab7",
    "learnable_preferences/network.py": "bceca30800969dda544f94bee51ce20a0d19293f80ac9a61e30839747201de76",
    "learnable_preferences/regretnet.py": "39bf69999f15ec5ee9352ac1faecde0793fca67bdd6885227dbbb9684bfc232b",
}
DESTINATION = ROOT / "phase3_author/vendor/PreferenceNet"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="No network or writes; verify existing external files")
    parser.add_argument("--source-dir", type=Path, help="Offline copy: directory containing the exact upstream README.md and learnable_preferences/")
    args = parser.parse_args(argv)
    if args.check_only and args.source_dir is not None:
        parser.error("--check-only and --source-dir are mutually exclusive")
    manifest = json.loads((ROOT / "phase3_author/SOURCES.json").read_text())
    records = {record["path"]: record["sha256"] for record in manifest["files"]}
    for name, expected in FILES.items():
        if records.get("vendor/PreferenceNet/" + name) != expected:
            raise RuntimeError("The fixed download hashes differ from SOURCES.json")
    pending = {}
    for name, expected in FILES.items():
        target = DESTINATION / name
        if target.is_file():
            if digest(target.read_bytes()) != expected:
                raise RuntimeError(f"Existing file differs from the recorded snapshot: {target}. Preserve your edits and remove that file before retrieving the exact version.")
            print(f"Verified: {name}")
            continue
        if args.check_only:
            raise FileNotFoundError(f"Missing external source: {target}. Run: python tools/fetch_preferencenet.py (or use --source-dir for offline recovery).")
        if args.source_dir is not None:
            blob = (args.source_dir.expanduser().resolve() / name).read_bytes()
        else:
            request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "HC-XMD-reproducibility-fetcher"})
            with urllib.request.urlopen(request, timeout=60) as response:
                blob = response.read()
        if digest(blob) != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}; no unverified file was installed")
        pending[name] = blob
    # All requested bytes have passed verification before any are installed.
    for name, blob in pending.items():
        target = DESTINATION / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download.tmp")
        temporary.write_bytes(blob)
        temporary.replace(target)
        print(f"Installed verified external file: {name}")
    print("PreferenceNet external source check passed (3 files, fixed commit).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"PreferenceNet setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
