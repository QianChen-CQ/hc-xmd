# Phase I V5: reconstructed experiments

These are **new, source-labelled reproductions**, not recovered old-server experiments. No historical CSV value is used in training, checkpoint selection or evaluation. Training completes before the common held-out evaluation starts; the final training checkpoint is used without test-set selection. Smoke outputs are never paper results.

## Implementations and sources

- **RegretNet2019-PyTorchPort**: the original additive network and augmented-Lagrangian algorithm, checked against Duetting et al., ICML 2019, and the MIT author repository `saisrivatsan/deep-opt-auctions`, commit `466e154b123921c142ee811216d0e64415bf31a5`. Architecture: independent allocation/payment MLPs, tanh, width100; allocation includes both dummy bidder and dummy item. 3x10 has five hidden layers per branch; 2x2 has two. Payment equals a sigmoid fraction of reported allocated value. The original license is included. This is not Optimaler's later budget-based training scheme.
- **AMenuNet-PaperReimplementation**: a separately written implementation of Duan et al., NeurIPS 2023. Public representations are fixed bidder/item IDs. The menu network receives no bids. Row attention, column attention and pooled global features generate allocations, positive bidder weights and boosts. A no-sale option is included. Training uses smooth choices; evaluation uses the exact affine-maximizer allocation and pivot payment equations. Author implementation commit `a62ec44e9280ec7a87313d1b8036ccfa5402091e` was used to cross-check dimensions and equations. It is not redistributed because no license was found in that snapshot.

Source links:
- https://proceedings.mlr.press/v97/duetting19a.html
- https://github.com/saisrivatsan/deep-opt-auctions
- https://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html
- https://github.com/Haoran0301/AMenuNet

## Training presets

| Item | RegretNet 3x10 | AMenuNet 3x10 |
|---|---:|---:|
| Synthetic values | fixed640000 iidU[0,1] profiles | fresh32768 profiles/round |
| Training budget | 400000 optimizer updates | 2000 outer rounds =32000 updates |
| Batch size |128|2048|
| Training optimizer | Adam, lr0.001 | Adam, lr0.0003 after warmup |
| Misreport optimization | projected Adam25 steps, lr0.1, one cached report/profile | unnecessary for AMA revenue training |
| Lagrange multiplier | initial5/bidder, update every100 steps | none |
| Quadratic coefficient | initial1; +1 every10000 steps | none |
| Architecture details | width100, five tanh hidden layers per branch | three axial blocks, width64, four heads, embeddings16 |
| Menu and temperatures | not a fixed menu |512 learned options +1 no-sale; allocation10, choice500|

AMenuNet iidU[0,1] **2x2** is a new shared-benchmark extension. Table3 of its paper does not include this setting: 2x2(A) is contextual and is not interchangeable with iidU[0,1]. The extension explicitly uses128 learned menu options, allocation temperature10 and2000 rounds. It must not be described as reproducing an original 2x2(C) table row. RegretNet2x2 uses its own author's configuration, including coefficient increment+5.

## Disclosed differences and implementation choices

1. PyTorch numerical kernels, initial random streams, Adam epsilon convention and CPU/GPU execution differ from the original TensorFlow1 program. We reproduce the architecture and optimization logic, not a bit-identical historical trajectory. Fresh seeds0/1/2 must be described as newly trained runs.
2. RegretNet uses independent reproducible RNGs for training data and permutations. Misreport Adam slots reset every minibatch; misreports themselves are cached. Each bidder deviates individually while all other reports remain fixed. The unnecessary original iteration-zero multiplier update before assigning the sampled adversarial report is omitted; multipliers first update after100 network updates. No test data enter training.
3. AMenuNet AppendixB specifies16-dimensional embeddings while the author script defaults to10; this paper preset follows16. Numerical equivalence to the author code can be checked at matching embedding dimensions, but trained checkpoints are new.
4. AMenuNet uses32768 samples, consistent with2^15 and author code. The text's32784 is not2^15. Warmup reaches the target after100 increments, avoiding the author script's101st increment. The final checkpoint is saved after all configured training updates, avoiding the public script's save-before-final-round behavior.
5. Non-contextual IDs are identical across a minibatch. Computing the menu once and broadcasting it retains the full gradient and gives the same mathematical operation with much lower memory use. There is no batch-dependent normalization or dropout.
6. For AMenuNet only, frozen-menu score/payment arithmetic at evaluation is float64, while neural training stays float32. This reduces cancellation near zero regret. Allocation/payment values are not clipped to hide violations. Positive finite bidder weights are explicitly checked.
7. Training validation from the old TensorFlow script is omitted because it does not control model selection or updates. Common held-out evaluation is performed separately, with its protocol saved by the root runner. A common audit is not the original RegretNet test setting of1000 restarts and2000 optimization steps.

## Artifacts and restart

Run from the full package root:

```bash
python -m phase1.train --method RegretNet --preset paper --setting 3x10 --seed 0 --device auto --output /root/hc_xmd_phase1_v5_results/RegretNet/seed0
python -m phase1.train --method AMenuNet --preset paper --setting 3x10 --seed 0 --device auto --output /root/hc_xmd_phase1_v5_results/AMenuNet/seed0
```

Use the same command plus `--resume` after interruption. Configurations must match. RegretNet resumes from its latest50000-step save; AMenuNet from its latest100-round save. Logs for updates after that saved checkpoint are removed before replay, and the resumed source hashes are recorded. Completed runs can be resumed without retraining.

`checkpoint.pt` and `model_final.pt` are identical final model artifacts. `resume_latest.pt` contains optimizer/RNG state and, for RegretNet, cached misreports, shuffle order and cursor. `train_values.npy` and `training_data.json` preserve RegretNet's fixed training set and hash. `config.json`, `provenance.json`, `train_log.csv`, `status.json`, `resume_history.json` are records. The shared evaluator should load `checkpoint.pt`, save held-out per-profile evidence, and make all manuscript figures/tables from those evidence files.

Feedback can omit `resume_latest.pt`, `train_values.npy`, intermediate `model_step*.pt`, and the duplicate `model_final.pt`. Keep them on the server for resumption. Keep `checkpoint.pt` plus logs/configuration/provenance in feedback.

## Local validation

`python -m phase1.test_training` verifies that interrupting and resuming each training procedure produces bit-identical CPU tensors to an uninterrupted small run. It also checks both setting dimensions and finite, feasible outputs. The separate independent source audit compares mechanism equations against author code and independent calculations. No GPU paper-budget training was performed while building this package.
