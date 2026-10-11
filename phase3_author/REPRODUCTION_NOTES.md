# Phase III: reproducible scope and provenance

This module REPLACES the prior custom-proxy training path for the named literature
methods. It does not rename old proxy results as author-method results.

## Methods and task boundaries

| Method | Executed mechanism | Training | Original measurable outcomes |
|---|---|---|---|
| CITransNet | Original author `TransformerMechanism`; three layers, four heads, hidden 64, continuous context dimension 10 | Released Setting H schedule and augmented-Lagrangian objective | Revenue, mean bidder regret, worst bidder regret; context-conditioned outcomes |
| PreferenceNet | Original author RegretNet mechanism and original allocation-preference classifier | Binary classifier pretraining, revenue/regret/preference objective, iterative self-training | Payment/revenue, summed regret, PCA, TVF, entropy, quota |
| ProportionNet | RegretNet architecture and an independently implemented original paper training objective | Positive-part unfairness from original Eq.(4), augmented Lagrangian Eq.(5), Algorithm 1 | Revenue, regret, per-item and summed unfairness |
| RegretNetPreferenceBaseline | Same author RegretNet mechanism as PreferenceNet | Same regret/revenue optimizer without preference loss | Paired baseline for the non-contextual preference/fairness task |
| RegretNetProportionBaseline | Same depth, size and initialization as ProportionNet | Same 120-epoch/640000-profile paper-directed protocol, with fairness terms removed | Controlled revenue/fairness comparison |
| CITransNetNoContext | Same author CITransNet architecture and Setting H valuation distribution | Same training protocol; bidder/item contexts are zeroed inside the mechanism during training AND evaluation | Controlled contextual-information ablation |

The three comparator/ablation variants are review-added controlled experiments,
not claims that these exact named variants appeared in the original papers.
PreferenceNet must be compared with its own matched baseline; ProportionNet with
its own matched baseline. Differences in architecture and training budget between
the two papers prevent attributing their direct difference solely to fairness.
CITransNetNoContext retains context-conditioned valuation samples; it only removes
the mechanism's access to public features, preserving the task distribution.

CITransNet uses `v = U[0,1] * sigmoid(bidder_context @ item_context.T)`;
both public contexts are independent U[-1,1] vectors. Setting H is 3 bidders,
10 items and 10 context dimensions. Values lie in [0,1]. The contextual data are
different from the unconditional U[0,1] task. DO NOT rank CITransNet revenue
against the latter task as though the valuation distribution were shared.

PreferenceNet's original Table 1 includes 2x2, 2x4, 4x2 and 4x4 additive and
unit-demand auctions. This implementation supports additive valuations;
`--n-agents 2 --n-items 2` selects its small original additive task.
The standalone training CLI's default 3x10 is an explicitly declared REVIEW EXTENSION, not an original
PreferenceNet paper-table configuration. ProportionNet's standalone default 3x10 is also
a review extension; changing size alone does not recover an old paper number.
The top-level revision suite selects **2x2 additive** for PreferenceNet,
ProportionNet and their revenue-only baseline, and **3x10 Setting H** for CITransNet.
ProportionNet follows the Table 1 hidden-layer counts for listed sizes (2 layers
for 2x2), while PreferenceNet and its matched baseline retain the author network's
3 hidden layers. This architecture difference is recorded, not an ablation claim.

## Source-preserving adapters

`bridge.py` compiles only selected ORIGINAL class AST nodes from the vendored
source. Class bodies and numerical network operations are unchanged. Unrelated
upstream training, plot imports and hardcoded `.cuda()` diagnostics are not run.
Source commits and hashes are in `SOURCES.json`.

The portable `train.py` is review-authored code, not an unchanged upstream trainer.
It implements and records the following:

* CITransNet: Adam model LR .001; persistent random misreports, projected Adam
  ascent LR .001, 25 training steps; lambda starts at 5, updated every 2 batches;
  rho starts at 1, increases by 5 every 2 epochs; 80 epochs, 100000 samples,
  batch 500. Training batch indices are sampled with replacement, as upstream.
* PreferenceNet: 200 epochs, 160000 training examples, batch 2048; original
  three hidden layers of 100 nodes (`n_hidden_layers=2` means two additional
  layers in the author source), ReLU, author payment/allocation heads. Regret
  optimization starts at truthful values, 25 projected gradient steps at .1.
  The released trainer's mean-reduced regret terms are retained, lambda updates
  every 25 steps and rho increases by 1 every 2500 steps. Preference weight 1.
  Classifier: 80000 random allocation exemplars, 50 epochs, Adam LR .001,
  betas (.5,.999), weight decay .005, balanced initial sampling. Original
  random-pair ranking labels use 5000 permutations and a strict 80% threshold.
  Every zero-based epoch 0,5,10,... adds 5000 mechanism allocations with classifier
  pseudo-labels, then retrains with a fresh optimizer and unbalanced sampling,
  matching the released schedule. Only synthetic preferences are used.
* ProportionNet: original paper Eq.(4) uses **positive parts** of allocation
  differences, whereas the later PreferenceNet fairness baseline uses absolute
  L1 differences. These are NOT silently equated here. Eq.(5) penalties use the
  squared sum of expected constraints. Defaults: 120 epochs, 640000 profiles,
  100 hidden nodes, both multipliers updated every 100 batches, both rho values
  increased every 2 epochs. The original text does not specify all numeric
  initialization/increment choices: this reconstruction explicitly uses regret
  lambda=5, fairness lambda=1, rho=1, increments=1, model Adam LR .001 and
  misreport gradient LR .1. Three hidden layers and batch 2048 are declared
  implementation choices for the new 3x10 task. This is a paper-directed
  reimplementation, NOT proof of bitwise original-table reproduction.

## Evaluation

All non-contextual methods and all training seeds share the same evaluation
values, generated with dedicated seed 271828 (training data seed 314159).
The evaluation seed does not depend on the optimizer/training seed.
No test data are used to select or update the mechanism or the preference model.

`ContextBatchModel` freezes each evaluation batch's public contexts. A candidate
array `[B,K,n,m]` flattened to `[B*K,n,m]` repeats contexts with
`repeat_interleave(K)`. It never resamples contexts during a deviation.

The common review audit is separate from the papers' native test protocols:
CITransNet used 100 initializations x 200 Adam steps; PreferenceNet and
ProportionNet used 1000 ascent steps. The new shared random+PGD audit must be
labeled as such, with budgets in every output. A larger empirical search can
find additional regret; none of these searches proves DSIC.

PCA (Preference Classification Accuracy) is the fraction of mechanism allocations
passing a ground-truth preference rule; it is NOT the learned classifier's
training accuracy and NOT a human opinion score. We report both `pca` and
`preference_probability`. For this review evaluation, the preference threshold
is frozen at the 80th percentile of scores from the initial synthetic allocation
reference set, rather than recomputing a percentile on evaluated model outputs.
The threshold is stored in `metadata.json` and the classifier training uses the
author's finite-pair ranking rule. That finite-sample labeling is an approximation
to the threshold rule, so the distinction is explicit.

Always export both mean regret across bidder-profile pairs and mean summed regret
across profiles: the former is the CITransNet convention; the latter appears in
PreferenceNet's evaluation. Keep within-test-profile standard deviations separate
from variation across the three training seeds.

No human burden numbers, human-study outcomes, menu sizes, description lengths,
deterministic-allocation counts or four-axis composite scores are invented.

## Files and restart

Each run writes `checkpoint.pt` (tensor state dictionary), `metadata.json`,
`training_curve.csv`, `eval_data.npz`, `data_manifest.json`, `config.json`,
`status.json` and `training_state.pt`. PreferenceNet also writes its classifier.
`--resume` restores the optimizer, multipliers, persistent deviations, preference
dataset and random states from the last fully saved epoch. The scientific
configuration must match exactly. If an interruption occurred before any completed
epoch, restart-from-seed is explicitly printed and recorded as such. Otherwise,
full state is restored and any log rows beyond the saved checkpoint epoch are
preserved separately and removed from the canonical curve before continuing.
Existing results are never silently reused
under different settings. `smoke` is a small process test, never a paper result.

CITransNet has an MIT license included. PreferenceNet has no explicit repository
license in the retrieved snapshot; access to source does not by itself grant
public redistribution. This private research bundle records its origin and must
not be described as wholly cleared for GitHub release. Verify permission or
replace those class definitions with a paper-only independent implementation
before publicly redistributing third-party source.
