"""Evaluator verification WITHOUT the test set.

1. Synthetic unit tests: independent per-image metrics vs torchmetrics/torcheval (as used upstream),
   including empty-GT / empty-prediction edge cases.
2. Real-data agreement on the VALIDATION split (SAM pseudo-labels as GT): upstream eval() vs the
   independent implementation, using an official checkpoint.

Usage: python test_evaluator.py <data_dir> <ckpt> [n_val_images] [device]
"""
import json
import sys

import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex
from torcheval.metrics.functional import binary_accuracy, binary_f1_score, binary_precision, binary_recall

from evaluator_core import build_model, chdir_repo, evaluate_model, make_dataset, per_image_metrics

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "repro"))
from common import assert_not_test  # noqa: E402


def upstream_style(pred, gt):
    p = torch.from_numpy(pred.astype(np.float32))
    g = torch.from_numpy(gt.astype(np.float32))
    iou = 1.0 if gt.sum() == 0 else BinaryJaccardIndex(zero_division=1e-9)(p, g).item()
    pf, gf = p.flatten(), g.flatten()
    return {"iou": iou, "precision": binary_precision(pf, gf).item(),
            "recall": binary_recall(pf.to(torch.uint8), gf.to(torch.uint8)).item(),
            "f1": binary_f1_score(pf, gf).item(), "accuracy": binary_accuracy(pf, gf).item()}


def synthetic_tests():
    rng = np.random.default_rng(0)
    cases = []
    for k in range(200):
        h, w = 37, 53
        gt = rng.random((h, w)) < rng.uniform(0, 0.5)
        pred = rng.random((h, w)) < rng.uniform(0, 0.5)
        if k % 10 == 0:
            gt[:] = False
        if k % 10 == 1:
            pred[:] = False
        if k % 10 == 2:
            gt[:] = False
            pred[:] = False
        cases.append((pred, gt))
    max_err = {}
    for pred, gt in cases:
        a, b = per_image_metrics(pred, gt), upstream_style(pred, gt)
        for key in a:
            max_err[key] = max(max_err.get(key, 0.0), abs(a[key] - b[key]))
    # hand-computed sanity case
    gt = np.zeros((4, 4), bool); gt[:2, :2] = True
    pred = np.zeros((4, 4), bool); pred[:2, :3] = True
    hand = per_image_metrics(pred, gt)
    assert abs(hand["iou"] - 4 / 6) < 1e-9 and abs(hand["precision"] - 4 / 6) < 1e-9 and hand["recall"] == 1.0
    return {"n_cases": len(cases), "max_abs_diff_vs_torchmetrics_torcheval": max_err,
            "pass": all(v < 1e-6 for v in max_err.values())}


def main():
    data_dir, ckpt = sys.argv[1], sys.argv[2]
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    report = {"synthetic": synthetic_tests()}
    assert_not_test("valid")
    chdir_repo()
    ds, args = make_dataset(data_dir, "valid", manual_masks=False)
    # deterministic, spread-out subset of validation (both sources)
    subset = np.linspace(0, len(ds) - 1, n_val).round().astype(int).tolist()
    fm = build_model("s", ckpt, device)
    res = evaluate_model(fm, ds, args, device, "official_pidnet_s_on_valid", subset)
    up, ind = res["upstream"], res["independent"]
    pairs = {"iou": ("mean_iou", "mean_iou"), "precision": ("mean_precision", "mean_precision"),
             "recall": ("mean_recall", "mean_recall"), "f1": ("mean_f1", "mean_f1"),
             "accuracy": ("mean_rand", "mean_accuracy")}
    diffs = {k: abs(up[a] - ind[b]) for k, (a, b) in pairs.items()}
    report["validation_agreement"] = {**res, "abs_diff": diffs, "pass": all(v < 1e-4 for v in diffs.values()),
                                      "note": "GT = SAM pseudo-labels of the validation split"}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
