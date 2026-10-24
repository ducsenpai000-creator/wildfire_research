"""Sanity checks for the DSA re-implementation (CPU, random data)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from dsa_loss import DSALoss, sinkhorn_assign

torch.manual_seed(0)
loss_fn = DSALoss(128)
f = torch.randn(3, 128, 34, 60, requires_grad=True)
lab = (torch.rand(3, 1, 270, 480) > 0.8).float()
loss = loss_fn(f, lab)
loss.backward()
print("L_DSA", float(loss), "finite", bool(torch.isfinite(loss)),
      "| grad->feats", float(f.grad.abs().sum()) > 0, "| grad->proxies", float(loss_fn.proxies.grad.abs().sum()) > 0)

x = F.normalize(torch.randn(5000, 128), dim=1)
p = F.normalize(torch.randn(5, 128), dim=1)
print("sinkhorn pixels per proxy (5000 px, 5 proxies):", torch.bincount(sinkhorn_assign(x, p), minlength=5).tolist())

l0 = loss_fn(torch.randn(2, 128, 34, 60), torch.zeros(2, 1, 270, 480))
print("all-background batch: L_DSA", float(l0), "finite", bool(torch.isfinite(l0)))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "wildfire-real-time-segmentation"))
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
config.MODEL.NAME = "pidnet_s"
print("PIDNet-S final_layer.conv1.in_channels =", get_seg_model(config, True).final_layer.conv1.in_channels)
