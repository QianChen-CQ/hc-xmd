#!/usr/bin/env bash
# Portable entrypoint; the V5 training and evaluation Python files are unchanged.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'EOF'
Usage: bash run_hcxmd.sh {smoke|run} [phase1|phase2|phase3|all] [suite options]

Examples:
  bash run_hcxmd.sh smoke all --device cpu --seeds 0
  bash run_hcxmd.sh run all --device cuda
  bash run_hcxmd.sh run phase2 --device cuda

Defaults: phase=all; outputs=./runs/{smoke,run}/phaseN
Override output root with HCXMD_OUTPUT_ROOT; override Python with PYTHON_BIN.
Phase II reads Phase I from the same output root and mode.
EOF
}
if [[ $# -eq 0 || "$1" == --help || "$1" == -h ]]; then usage; exit 0; fi
MODE="$1"; shift
case "$MODE" in smoke|run) ;; *) usage >&2; exit 2 ;; esac
PHASE=all
if [[ $# -gt 0 && "$1" != --* ]]; then PHASE="$1"; shift; fi
case "$PHASE" in phase1|phase2|phase3|all) ;; *) usage >&2; exit 2 ;; esac
for ARG in "$@"; do
  case "$ARG" in
    --outdir|--outdir=*|--phase1-dir|--phase1-dir=*|--phase|--phase=*|--mode|--mode=*)
      echo 'Use HCXMD_OUTPUT_ROOT and the positional mode/phase; these suite path overrides are not accepted here.' >&2
      exit 2 ;;
  esac
done
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="$("$PYTHON_BIN" -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${HCXMD_OUTPUT_ROOT:-$HERE/runs}")"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "$HERE"
if [[ "$PHASE" == all || "$PHASE" == phase3 ]]; then
  "$PYTHON_BIN" tools/fetch_preferencenet.py --check-only
fi
PHASES=("$PHASE")
if [[ "$PHASE" == all ]]; then PHASES=(phase1 phase2 phase3); fi
for CURRENT_PHASE in "${PHASES[@]}"; do
  RESULT_DIR="$OUTPUT_ROOT/$MODE/$CURRENT_PHASE"
  mkdir -p -- "$RESULT_DIR"
  set +e
  "$PYTHON_BIN" suite.py --phase "$CURRENT_PHASE" --mode "$MODE" \
    --outdir "$RESULT_DIR" --phase1-dir "$OUTPUT_ROOT/$MODE/phase1" \
    "$@" 2>&1 | tee -a "$RESULT_DIR/console.log"
  CODES=("${PIPESTATUS[@]}")
  set -e
  RESULT_CODE="${CODES[0]}"
  if [[ "${CODES[1]}" -ne 0 && "$RESULT_CODE" -eq 0 ]]; then RESULT_CODE="${CODES[1]}"; fi
  set +e
  "$PYTHON_BIN" tools/pack_feedback_portable.py --outdir "$RESULT_DIR" --output-root "$OUTPUT_ROOT"
  PACK_CODE=$?
  set -e
  if [[ "$RESULT_CODE" -ne 0 ]]; then exit "$RESULT_CODE"; fi
  if [[ "$PACK_CODE" -ne 0 ]]; then exit "$PACK_CODE"; fi
done
