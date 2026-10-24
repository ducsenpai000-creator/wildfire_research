# Wildfire Smoke Segmentation on UAV Imagery — Reproduction & Loss Design

Independent research project: **faithful reproduction** of the WACV 2025 paper
*"Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models"* (Pesonen et al.),
a **benchmark of 5 segmentation methods** under one locked protocol, and a **pre-registered loss-function study (BRRC)**
for smoke segmentation trained on SAM pseudo-labels. Everything runs on a single laptop GPU (RTX 3050, 6 GB).

> Status (Oct 2026): reproduction complete; 4 of 5 comparison methods tested; BRRC ablation runs in progress.
> The test set is evaluated **once**, at the end, under a hash-locked evaluation (see [Evaluation discipline](#evaluation-discipline)).

---

## Highlights

- **Reproduced the paper's main result**: PIDNet-S student, test mIoU **0.608** vs **0.594** reported
  (Boreal 0.697 vs 0.689; AIFM 0.518 vs 0.498), within run-to-run noise.
  Re-evaluating the authors' 4 released checkpoints matches Tables 2, 4 and 8 **to 3 decimals**.
- **Found why the released code does not reproduce the paper** (0.474 instead of 0.594), using checkpoint forensics:
  - ImageNet initialisation is disabled in the released code (`if False:`), but the authors' checkpoints were
    ImageNet-initialised (cosine 0.964 to converted PIDNet-S ImageNet weights).
  - The real learning rate is **1e-4** (the paper says 1e-3), proven by a **bit-exact float32 replay** of AdamW's
    decoupled weight decay on zero-gradient parameters; the replay also recovers the selected epoch (S = 37).
  - The boundary loss is identically **zero** in both the code and the released checkpoints (labels divided by 255).
  - A companion dataset paper reports 80-image numbers as if they were 40-image numbers.
- **Benchmarked 5 methods on the same split, labels and evaluator** (PIDNet-S, DSA, Mask2Former, SegFormer-B2, FoSp),
  including running FoSp's official mmseg code on a modern PyTorch stack.
- **Designed and pre-registered BRRC** (Boundary-Reliability & Recall-Calibrated loss) before any training,
  with a fixed ablation grid, a validation-only selection rule and a seed-noise criterion for claiming improvement.
- **Engineering under constraints**: 5 GB VRAM cap with exact gradient accumulation; an unattended job queue with
  safe pause/resume across reboots; and a measured speed-up of ~20 % per epoch from a data-loader change that leaves every metric identical.

## Results (test set, 80 manually annotated images)

Metric: per-image mean IoU of the smoke class (paper Eq. 1, threshold 0.5). Protocol P = the paper's exact split.

| Model | Source | mIoU (80) | Boreal (40) | AIFM (40) | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|
| PIDNet-S, WACV 2025 paper | reported | 0.594 | 0.689 | 0.498 | 0.645 | 0.891 | 0.707 |
| Authors' checkpoints (S/M/L, ablation) | re-evaluated | exact match | | | | | |
| E0: released code as-is | trained | 0.474 | 0.738 | 0.210 | 0.524 | 0.678 | 0.561 |
| **E0-IN: reproduction** (+ ImageNet init) | trained | **0.608** | 0.697 | **0.518** | 0.656 | 0.877 | **0.719** |
| PIDNet-S baseline under 5 GB cap | trained | 0.584 | 0.665 | 0.502 | 0.633 | 0.851 | 0.698 |
| DSA (ECCV 2024) on PIDNet-S | trained | 0.594 | 0.693 | 0.496 | 0.666 | 0.829 | 0.706 |
| Mask2Former Swin-T | trained | 0.571 | 0.694 | 0.448 | 0.604 | 0.850 | 0.674 |
| SegFormer-B2 | trained | 0.485 | 0.672 | 0.298 | 0.522 | 0.811 | 0.590 |
| FoSp (AAAI 2024, official code) | training | – | | | | | |
| BRRC (proposed) | final test pending | – | | | | | |

Single seed for every trained row. Seed noise for this model is at least ~0.04 test mIoU (estimated from the authors'
own runs), so differences smaller than that are not claimed as real.

### BRRC ablation (validation only; the test set is still locked)

| Run | Change vs. baseline | best val mIoU | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Baseline (5 GB) | – | 0.527 | 0.616 | 0.763 | 0.644 |
| E1a | one-sided boundary down-weighting, ρ = 0.10 | 0.507 | 0.632 | 0.803 | 0.629 |
| E1b | one-sided boundary down-weighting, ρ = 0.25 | 0.520 | 0.626 | 0.745 | 0.637 |
| E2a | recall-calibrated Tversky term, λ = 1 | 0.553 | 0.696 | 0.760 | 0.682 |
| **E2b** | recall-calibrated Tversky term, λ = 5 | **0.587** | **0.700** | **0.788** | **0.711** |

The Tversky component helps on validation. Boundary down-weighting alone does not; this negative result is reported, not dropped.
Still to run: the combined run E3, plus 2 extra seeds for both the baseline and the best BRRC variant, then the one-time locked test.

## Method: BRRC loss

SAM pseudo-labels generated from bounding boxes are conservative: in the paper, precision is 0.91 and recall is 0.69 against manual masks.
BRRC adds two weight-only terms to the unchanged PIDNet-S objective (labels are never modified):

1. **Boundary reliability (BR)**: background pixels just outside each SAM mask get a reduced weight. The weight ramps linearly over a
   radius that scales with smoke size, `r = clip(ρ·√A, 3, 40)` px. Distances are computed on the GPU with iterated max-pool dilation.
2. **Recall calibration (RC)**: a per-image soft Tversky loss (α = 0.3, β = 0.7) on the main head, which penalises
   false negatives more than false positives.

This is a combination of known ideas (label-noise down-weighting near boundaries; the Tversky loss), and it is not claimed as a
new algorithm. Design and decision rules are in [`reports/BRRC_DESIGN_PREREGISTRATION.md`](reports/BRRC_DESIGN_PREREGISTRATION.md);
the implementation is [`research_extension/methods/brrc.py`](research_extension/methods/brrc.py), with unit tests showing it reduces exactly to the baseline loss when switched off.

## Evaluation discipline

- **Locked protocol**: [`reports/PROTOCOL_LOCK.md`](reports/PROTOCOL_LOCK.md) fixes the data, splits, metric and checkpoint rules up front.
  - Every deviation (hardware-forced micro-batching, library versions, resume events) is logged there with its reason.
- **Data integrity**: all 4,392 files of Boreal Forest Fire Subset C are verified against the publisher's SHA-256 checksums.
  - Fixed manifests define each split; no samples are added, removed or re-split.
- **Test-set lock**: during development, no test image, mask or statistic is read.
  - Each test run is preceded by a lock file that records SHA-256 hashes of the code, configs, checkpoints and manifests.
  - A separate evaluator refuses to run if anything changed, and refuses to run twice.
  - An early lock was voided after a path bug; the incident is documented in [`reports/STAGE1_TEST_LOCK.md`](reports/STAGE1_TEST_LOCK.md).
- **Selection on validation only**: checkpoints and hyperparameters are chosen on validation; every run is logged, failed and crashed ones included
  ([`reports/EXPERIMENT_LEDGER.csv`](reports/EXPERIMENT_LEDGER.csv)).

## Repository layout

```
research_extension/
  repro/        train_repro.py (PIDNet trainer: micro-batching, full-state resume, DSA/BRRC hooks), pause/resume scripts
  eval/         evaluator_core.py, final_test_eval.py (locked one-shot test), make_lock.py, evaluator unit tests
  methods/      brrc.py, dsa_loss.py (+ tests)
  compare/      train_hf.py (SegFormer/Mask2Former), run_fosp.py (official FoSp + compatibility shims), predictors.py
  probes/       checkpoint forensics (init, exact-LR replay), memory/speed probes
  data_tools/   manifests, SAM-vs-bbox statistics, VOC view for FoSp
configs/        queue.json (all jobs), FoSp config, experiment registry, pinned requirements
tools/scripts/  queue_runner.py (unattended queue, retries, auto status reports), status_report.py, watch_progress.ps1
reports/        protocol, reproduction gaps, literature review (10 methods), research gap, test results, locks, ledger
experiments/    per-run epoch logs (epochs.csv), run metadata and summaries
train.cmd / stop.cmd / watch.cmd   start/resume, safe stop, live progress
```

Detailed reports are written in Vietnamese. Code, configs and this README are in English.

## Reproducing

Not included in this repository: the datasets (Boreal Forest Fire Subset C from Fairdata, the AIFM-remixed data),
model checkpoints, Python environments and third-party code. Third-party code is used unmodified from:
- the official WACV 2025 repository: `gitlab.com/fgi_nls/public/wildfire-real-time-segmentation` @ `2382c295`
- FoSp: `github.com/LujianYao/FoSp` @ `02b9fd1`

The environment is Python 3.8, torch 2.4.1+cu118 ([`configs/requirements_lock_py38.txt`](configs/requirements_lock_py38.txt)); see [`reports/environment.md`](reports/environment.md).
Data download and verification steps are in [`reports/data_integrity.md`](reports/data_integrity.md).

```bat
train.cmd        :: start or resume the job queue (resumes from the last saved epoch) and show live progress
stop.cmd         :: safe stop: waits for the next checkpoint, then stops
```

## Tech stack

PyTorch, torchvision, Hugging Face Transformers (SegFormer, Mask2Former), MMSegmentation/mmcv (FoSp), torcheval,
OpenCV, NumPy, pandas, PowerShell automation, Git.

## References

Pesonen et al., WACV 2025 · Xu et al., PIDNet, CVPR 2023 · Xie et al., SegFormer, NeurIPS 2021 · Cheng et al., Mask2Former, CVPR 2022 ·
Yao et al., FoSp, AAAI 2024 · Yao et al., DSA, ECCV 2024 · Li et al., MultiNatSmoke, WACV 2026 · Salehi et al., Tversky loss, 2017.
The full list with links is in [`paper/README.md`](paper/README.md); the review of 10 related methods is in [`reports/literature_review.md`](reports/literature_review.md).
