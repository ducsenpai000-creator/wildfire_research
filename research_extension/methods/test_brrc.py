"""Checks for the BRRC implementation (CPU, synthetic data):
1. BRRCFullModel with BR and RC off == upstream FullModel (same weights, same inputs) -> identical loss/outputs.
2. Boundary weights: only background pixels in the outer band are < 1; smoke pixels are 1; radius follows
   clip(rho*sqrt(area), r_min, r_max); weight grows linearly with chessboard distance.
3. Tversky term: finite, in [0,1], gradient flows.
"""
import copy
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "wildfire-real-time-segmentation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
from pidnet_utils.criterion import BondaryLoss  # noqa: E402
from pidnet_utils.utils import FullModel  # noqa: E402
from brrc import BRRCFullModel, boundary_reliability_weights, soft_tversky  # noqa: E402

torch.manual_seed(0)
config.MODEL.NAME = "pidnet_s"
net = get_seg_model(config, True)
sem = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]).view(1, 1, 1))
up = FullModel(copy.deepcopy(net), sem, BondaryLoss()).train()
br = BRRCFullModel(copy.deepcopy(net), sem, BondaryLoss()).train()
x = torch.randn(2, 3, 128, 224)
y = (torch.rand(2, 1, 128, 224) > 0.85).float()
bd = (torch.rand(2, 1, 128, 224) > 0.95).float() / 255.0
torch.manual_seed(1)
l_up, o_up, _, _ = up(x, y, bd)
torch.manual_seed(1)
l_br, o_br, _, _ = br(x, y, bd)
print("1. off == upstream: loss", float(l_up), float(l_br), "equal:", torch.allclose(l_up, l_br),
      "| outputs equal:", all(torch.equal(a, b) for a, b in zip(o_up, o_br)),
      "| state_dict keys equal:", list(up.state_dict()) == list(br.state_dict()))

m = torch.zeros(1, 1, 200, 300)
m[0, 0, 80:120, 100:200] = 1                                   # area 4000 -> sqrt 63.2
w = boundary_reliability_weights(m, rho=0.25, r_min=3, r_max=40)
r = min(max(0.25 * math.sqrt(4000), 3), 40)
print(f"2. expected r={r:.2f} | smoke weights all 1: {bool((w[m > 0] == 1).all())} | "
      f"w at d=1,5,15,16: {[round(float(w[0,0,79-k+1-1+0,150]),4) for k in (1,5,15,16)]} "
      f"(expect {[round(k/r,4) if k <= r else 1.0 for k in (1,5,15,16)]}) | "
      f"far background 1: {float(w[0,0,0,0])} | empty image all 1: {bool((boundary_reliability_weights(torch.zeros(1,1,50,50), 0.25) == 1).all())}")

br2 = BRRCFullModel(copy.deepcopy(net), sem, BondaryLoss(), br_rho=0.25, rc_lambda=5).train()
l2, _, _, _ = br2(x, y, bd)
l2.sum().backward()
print("3. BRRC on: loss", float(l2), "terms", br2.last_terms,
      "| grad ok:", all(p.grad is not None for p in br2.model.final_layer.parameters()))
p = torch.rand(2, 1, 32, 32, requires_grad=True)
t = soft_tversky(p, (torch.rand(2, 1, 32, 32) > 0.7).float())
t.backward()
print("   tversky", float(t), "in [0,1]:", 0 <= float(t) <= 1, "grad:", float(p.grad.abs().sum()) > 0)
