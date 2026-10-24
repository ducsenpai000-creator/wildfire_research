"""Evaluator core shared by the validation checker and the locked final test evaluator.

Two independent computations are produced for every dataset/model pair:
  * "upstream": the repository's own eval() function (eval.py), unchanged -> official numbers;
  * "independent": a re-implementation of paper Eq. (1) and the per-image binary metrics,
    used to cross-check the upstream numbers.
Only aggregate metrics are returned (no per-image values) so the same code is safe for the test set.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import REPO, repo_args  # noqa: E402

from datasets.segmentation_data import WFSeg  # noqa: E402
from eval import eval as upstream_eval  # noqa: E402
from eval import eval_sam as upstream_eval_sam  # noqa: E402
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
from pidnet_utils.criterion import BondaryLoss  # noqa: E402
from pidnet_utils.utils import FullModel  # noqa: E402


def per_image_metrics(pred, gt):
    """pred, gt: bool numpy arrays of one image. Mirrors upstream conventions:
    IoU = 1 if GT empty; precision/recall/F1 -> 0 when undefined (torcheval nan_to_num)."""
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = pred.size - tp - fp - fn
    iou = 1.0 if gt.sum() == 0 else tp / (tp + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / pred.size
    return {"iou": float(iou), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "accuracy": float(acc)}


def aggregate(rows):
    keys = rows[0].keys()
    return {f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in keys}


def build_model(size, ckpt, device):
    config.MODEL.NAME = "pidnet_" + size
    model = get_seg_model(cfg=config, imgnet_pretrained=True)
    # upstream eval.py uses pos_weight=1 for its reported loss
    fm = FullModel(model, torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1, 1, 1)), BondaryLoss())
    sd = torch.load(ckpt, map_location="cpu")
    res = fm.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    return fm.to(device).eval()


def make_dataset(data_dir, split, manual_masks):
    args = repo_args(data_dir=data_dir, batch_size=1, include_id=False, mode=split,
                     manual_masks=manual_masks, eval_dir="manual_masks")
    return WFSeg(data_dir, split, manual_masks=manual_masks, boundary=True, args=args), args


@torch.no_grad()
def evaluate_model(fm, dataset, args, device, exp_name, subset=None):
    ds = torch.utils.data.Subset(dataset, subset) if subset is not None else dataset
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    args.device, args.exp, args.vis_val, args.include_id = device, exp_name, False, False
    up = upstream_eval(fm, args, loader)
    rows = []
    for batch in loader:
        out = fm.model(batch["img"].to(device))
        out1 = torch.nn.functional.interpolate(out[1], size=batch["img"].shape[-2:], mode="bilinear",
                                               align_corners=True)
        pred = (torch.sigmoid(out1) > 0.5)[0, 0].cpu().numpy()
        gt = (batch["mask"][0, 0] > 0.5).numpy()
        rows.append(per_image_metrics(pred, gt))
    return {"n_images": len(ds),
            "upstream": {k: up[k] for k in ("mean_iou", "mean_rand", "mean_precision", "mean_recall", "mean_f1")},
            "independent": aggregate(rows)}


@torch.no_grad()
def evaluate_predictor(predict, dataset, thresholds=(0.5,)):
    """Comparison methods: `predict(image_path) -> prob [1080,1920]`; ground truth is the same upstream WFSeg
    manual-mask tensor as for PIDNet (resize 1080x1920 + round). Metrics: the independent per-image
    implementation (validated equal to the upstream torchmetrics/torcheval functions, reports/evaluator_tests.json)."""
    rows = {t: [] for t in thresholds}
    for i in range(len(dataset)):
        gt = (dataset[i]["mask"][0] > 0.5).numpy()
        prob = predict(Path(dataset.img_dir) / dataset.img_list[i]).numpy()
        assert prob.shape == gt.shape, (prob.shape, gt.shape)
        for t in thresholds:
            rows[t].append(per_image_metrics(prob > t, gt))
    return {"n_images": len(dataset), "by_threshold": {str(t): aggregate(r) for t, r in rows.items()}}


def evaluate_sam_vs_manual(data_dir, split="test"):
    """Upstream `eval.py --eval-sam` path: SAM pseudo-labels scored against manual masks."""
    args = repo_args(data_dir=data_dir, batch_size=1, mode=split, eval_dir="manual_masks", exp="sam_masks")
    manual = WFSeg(data_dir, split, manual_masks=True, boundary=True, args=args)
    args.sup_dir, args.eval_snake, args.manual_masks = "sam_masks", True, False
    sam = WFSeg(data_dir, split, manual_masks=False, boundary=True, args=args)
    assert len(manual) == len(sam)
    ml = torch.utils.data.DataLoader(manual, batch_size=1, shuffle=False)
    sl = torch.utils.data.DataLoader(sam, batch_size=1, shuffle=False)
    up = upstream_eval_sam(args, sl, ml)
    rows = [per_image_metrics((s["mask"][0, 0] > 0.5).numpy(), (m["mask"][0, 0] > 0.5).numpy())
            for s, m in zip(sl, ml)]
    return {"n_images": len(manual),
            "upstream": {k: up[k] for k in ("mean_iou", "mean_rand", "mean_precision", "mean_recall", "mean_f1")},
            "independent": aggregate(rows)}


def chdir_repo():
    import os
    os.chdir(REPO)
