#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p tables figures
python scripts/build_tables.py
python figures/build_figures.py
