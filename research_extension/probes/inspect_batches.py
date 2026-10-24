"""D4 checks on TRAIN/VALID batches only (test-set lock): shapes, value sets, mask/boundary
conventions of the upstream WFSeg loader, loss components of the upstream FullModel, and
train/valid visualisations.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import REPO, ROOT, assert_not_test, repo_args  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from datasets.segmentation_data import WFSeg  # noqa: E402
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
from pidnet_utils.criterion import BondaryLoss, weighted_bce  # noqa: E402
from pidnet_utils.utils import FullModel  # noqa: E402
from utils.plot_outputs import inverse_norm  # noqa: E402


def uniq(t):
    u = torch.unique(t)
    return [round(float(x), 6) for x in u[:8]] + (["..."] if len(u) > 8 else [])


def main(data_dir, fig_dir):
    torch.manual_seed(0)
    np.random.seed(42)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    args = repo_args(data_dir=data_dir, include_id=True)
    config.MODEL.NAME = "pidnet_s"
    fm = FullModel(get_seg_model(config, True),
                   torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).view(1, 1, 1)), BondaryLoss()).cuda()
    for mode in ("train", "valid"):
        assert_not_test(mode)
        ds = WFSeg(root_dir=data_dir, mode=mode, boundary=True, include_id=True, args=args)
        # pick a few BFF and AIFM samples that have a SAM mask
        idx = [i for i, n in enumerate(ds.img_list) if n[:-5 if n.endswith('.jpeg') else -4] in ds.mask_list]
        pick = [i for i in idx if ds.img_list[i].endswith(".jpg")][:3] + [i for i in idx if ds.img_list[i].endswith(".jpeg")][:3]
        samples = [ds[i] for i in pick]
        img = torch.stack([s["img"] for s in samples])
        mask = torch.stack([s["mask"] for s in samples])
        bd = torch.stack([s["boundary"] for s in samples])
        r = {"ids": [s["id"] for s in samples], "img_shape": list(img.shape), "img_dtype": str(img.dtype),
             "img_range": [float(img.min()), float(img.max())], "mask_shape": list(mask.shape),
             "mask_values": uniq(mask), "boundary_values": uniq(bd),
             "boundary_eq_1_pixels": int((bd == 1).sum()), "boundary_gt0_pixels": int((bd > 0).sum()),
             "mask_fg_fraction": [round(float(m.mean()), 4) for m in mask]}
        fm.eval()
        with torch.no_grad():
            x, y, b = img.cuda(), mask.cuda(), bd.cuda()
            outs = fm.model(x)
            outs = [torch.nn.functional.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=True) for o in outs]
            r["logit_shapes"] = [list(o.shape) for o in outs]
            r["l2_boundary_loss_upstream_target"] = float(fm.bd_loss(outs[-1], b))
            r["l2_boundary_loss_if_target_binarised"] = float(fm.bd_loss(outs[-1], (b > 0).float()))
            r["weighted_bce_pos_count_upstream"] = int((b.view(-1) == 1).sum())
            r["pred_fg_fraction_untrained"] = float((torch.sigmoid(outs[1]) > 0.5).float().mean())
        report[mode] = r
        fig, ax = plt.subplots(len(samples), 3, figsize=(12, 2.3 * len(samples)))
        for k, s in enumerate(samples):
            ax[k, 0].imshow(inverse_norm(s["img"]).permute(1, 2, 0).clamp(0, 1).numpy())
            ax[k, 1].imshow(s["mask"][0].numpy(), cmap="gray")
            ax[k, 2].imshow((s["boundary"][0] > 0).numpy(), cmap="gray")
            ax[k, 0].set_title(s["id"], fontsize=7)
            for c in range(3):
                ax[k, c].axis("off")
        ax[0, 1].set_title("SAM pseudo-label (train-time transform)", fontsize=8)
        ax[0, 2].set_title("boundary target > 0", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"d4_{mode}_batch.png", dpi=80)
        plt.close()
    print(json.dumps(report, indent=1))
    (fig_dir / "d4_batch_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import os
    os.chdir(REPO)
    main(sys.argv[1], sys.argv[2])
