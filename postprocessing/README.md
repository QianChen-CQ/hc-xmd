# Manuscript V7.3 numerical postprocessing

This directory contains the 144 archived per-model evaluation rows, figure inputs, provenance and audit summaries used for manuscript V7.3. These are outputs from completed V5 server runs, not placeholder values. The scripts regenerate the five experimental LaTeX tables, numeric macros and statistical Figures 2-6 from those exports; no GPU or model training is required.

Use a separate postprocessing environment from the GPU training environment (the pinned NumPy versions differ). From the repository root:

```bash
conda create -n hcxmd_post python=3.12 -y
conda activate hcxmd_post
python -m pip install -r postprocessing/requirements-postprocessing.txt
bash postprocessing/REGENERATE.sh
```

Tables are written to `postprocessing/tables/`. Figures are written to `postprocessing/figures/` as PNG, PDF and SVG. This does not compile the manuscript PDF. The table snippets use the manuscript's `revcaption` macro and labels.

## Evidence scope

The 144 rows distinguish 30 new training runs, 12 released author checkpoints and 48 checkpoint evaluations, including six Phase II evaluations that reuse Phase I models. Training protocols differ across sources. The retained diagnostics include the all-null selected GemNet reproduction and the collapsed seed in the no-context control. Keep all three seeds and both task sizes distinct. The context revenue change of +56.19% is a difference between three-seed means, not a stable causal effect of adding context. Synthetic preference and fairness criteria are not human-subject measures.

Raw per-profile arrays, all new trained checkpoints and complete training logs are not bundled in this public source package. The supplied summaries permit regenerating displayed aggregates; a fresh experiment with `run_hcxmd.sh` produces its own training, evaluation and feedback artifacts. Released upstream Optimaler checkpoints are supplied separately under `vendor/optimaler/`.
