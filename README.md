# HC-XMD: training, shared evaluation, and paper artifacts

Reproducibility companion for **Human-Centric Explainable Mechanism Design:
From Incentive Compatibility to Human Alignment**.

This release assembles the three V5 server packages into one source tree. Their
training and evaluation Python files are unchanged. The packages originally
differed only in their default entrypoint and release metadata. The server
feedback bundles contain matching copies of the V5 experiment code, completed
run records, selected checkpoints, per-profile results, and per-bidder deviation
gains. Those feedback archives are not bundled in this public source release; the archived per-model summary exports are provided under `postprocessing/`. This repository adds portable launch/setup scripts and the manuscript's
separate postprocessing workflow; it does not claim that repackaging reran the
experiments.

[中文说明](README_zh.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## 1. Environment and external setup

Commands below run in a Linux Bash terminal, from this repository's root.
The reported server records identify Python 3.12.14, PyTorch 2.5.1+cu121,
NumPy 1.26.4 and CUDA runtime 12.1; the Phase I/II records also identify an
NVIDIA A100-PCIE-40GB. GPU training is recommended for full reference budgets.

Create the training environment once:

```bash
conda create -n hcxmd_revision_v5 python=3.12 -y
conda activate hcxmd_revision_v5
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For a machine without an NVIDIA GPU, replace the PyTorch install command with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Before running Phase III or `all`, retrieve the exact external PreferenceNet
files. They are not redistributed in this repository because the inspected
upstream snapshot had no explicit license. The script downloads three files
from a fixed author commit and checks their recorded SHA-256 values:

```bash
python tools/fetch_preferencenet.py
python tools/fetch_preferencenet.py --check-only
```

For offline setup using an existing copy of that exact upstream snapshot:

```bash
python tools/fetch_preferencenet.py --source-dir /path/to/PreferenceNet
```

The source directory must contain the upstream `README.md` and
`learnable_preferences/network.py` and `learnable_preferences/regretnet.py`.
The original server bundle's `phase3_author/vendor/PreferenceNet/` is also a
valid source directory. After dependencies and external source are installed,
experiment execution does not require GitHub access. Do not add the fetched
files to a public commit or upload them through the GitHub website;
`.gitignore` protects Git operations but does not filter manual web uploads.

## 2. Run the experiments

First test the pipeline with a short CPU run using one seed:

```bash
bash run_hcxmd.sh smoke all --device cpu --seeds 0
```

`smoke` uses deliberately reduced training and audit settings. It is a software
check and must not be reported as a paper experiment. Output is kept separately
under `runs/smoke/phase1`, `phase2`, and `phase3`.

To run the complete reference-budget suite on one GPU:

```bash
bash run_hcxmd.sh run all --device cuda
```

The default seeds are `0 1 2`. The default shared audit uses 512 test profiles per
model and evaluation batches of 16. `all` runs the phases sequentially:

| Phase | Work performed | Dependency |
| --- | --- | --- |
| I | Train RegretNet and AMenuNet for 3 bidders × 10 items; common audit | None |
| II | Re-audit the frozen Phase I models, evaluate released Optimaler checkpoints for the separate 3×10 and 2×2 settings, train 2×2 AMenuNet and GemNet, transform GemNet prices and audit | Completed Phase I outputs from the same mode and output root |
| III | Train/audit CITransNet, PreferenceNet and ProportionNet with their matched controls | External PreferenceNet setup; no Phase I/II result dependency |

Phase II is **not evaluation-only**: it also trains the declared 2×2 extensions.
The separate settings and training/provenance roles must not be pooled into a
single matched-budget ranking.

To run stages separately:

```bash
bash run_hcxmd.sh run phase1 --device cuda
bash run_hcxmd.sh run phase2 --device cuda
bash run_hcxmd.sh run phase3 --device cuda
```

To keep a run alive after SSH disconnects:

```bash
mkdir -p runs
nohup bash run_hcxmd.sh run all --device cuda > runs/run_all.log 2>&1 &
tail -f runs/run_all.log
```

Returning to the shell immediately after `nohup` does not mean completion.
Check `runs/run/phaseN/suite_status.json`: successful phases have
`"state": "COMPLETED"`. Each stage also writes a `console.log` and a feedback
archive such as `runs/run/phase1_feedback.tar.gz`.

To use another output location, set a task-specific output root consistently
for all stages:

```bash
export HCXMD_OUTPUT_ROOT="$HOME/hcxmd_outputs"
bash run_hcxmd.sh run phase1 --device cuda
bash run_hcxmd.sh run phase2 --device cuda
bash run_hcxmd.sh run phase3 --device cuda
```

Those outputs are placed in `$HCXMD_OUTPUT_ROOT/run/phaseN`. Relative custom
paths are resolved from the terminal's current directory. The wrapper always
passes the matching Phase I directory to Phase II. Do not use the underlying
suite's original `/root/` defaults when using this portable layout.

After an interruption, first confirm that the old process has stopped, then
repeat the same command. Completed models are retained and incomplete training
can resume from available state. The code rejects mismatched configurations or
source hashes in an existing output directory. Use a new output root after
changing scientific code or settings. Do not run two processes writing to the
same stage directory. If only the evaluation stage runs out of memory, choose a
fresh output root and add `--eval-batch-size 4`; changing a training batch size
changes the reference protocol and must be reported.

## 3. Outputs and interpretation

Run directories include configurations, provenance, training curves, selected
checkpoints, evaluation arrays, `profile_metrics.csv`, `bidder_gains.csv`,
summary tables and plots. Metrics include revenue, welfare, mean/max regret,
PMR, individual rationality, allocation feasibility, payment checks and
family-specific task metrics. PMR is the fraction of bidder-profile pairs with
deviation gain above `1e-6`.

Feedback archives intentionally exclude large restart caches, duplicate
checkpoints and regenerable training arrays. Identical files are stored once;
`DEDUPLICATION_MAP.json` records the omitted logical paths and their retained
copies. They are selected evidence archives, not full disk images or guaranteed
training-resume snapshots. Preserve the original run directory for resuming.

Finite deviation searches and finite-grid GemNet checks are empirical audits,
not a global continuous-domain certificate. Infeasible candidates are retained
as diagnostics rather than silently treated as valid auction comparisons.
Synthetic preference experiments are not human-participant studies. Seed
standard deviations and upstream checkpoint labels have distinct provenance.
Method-specific implementation choices are documented in `phase1/`, `gemnet/`
and `phase3_author/`.

## 4. Regenerate manuscript tables and figures without training

The `postprocessing/` directory uses the manuscript's archived result rows and
contains its own README and dependency file. Use a **separate environment**:
its pinned NumPy/Matplotlib versions differ from the training environment and
must not overwrite the training requirements.

```bash
conda create -n hcxmd_post python=3.12 -y
conda activate hcxmd_post
python -m pip install -r postprocessing/requirements-postprocessing.txt
```

Then follow `postprocessing/README.md` for the exact regeneration
commands. This operation regenerates presentation artifacts from saved results;
it does not train a new model. Return to training with
`conda activate hcxmd_revision_v5`.

## Source integrity and licensing

`OPTIMALER_MANIFEST.json` and `phase3_author/SOURCES.json` preserve upstream
commits and hashes, and are checked by the loaders. All bundled upstream MIT
notices are retained. PreferenceNet's three external files are excluded from
the public package and fetched separately. The root MIT license covers newly
authored HC-XMD files only; see `THIRD_PARTY_NOTICES.md` for the exact scope.
