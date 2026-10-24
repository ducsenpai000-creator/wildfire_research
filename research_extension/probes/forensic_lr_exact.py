"""Exact float32 replay of AdamW decoupled weight decay on a parameter that receives zero gradient
(boundary-head BN weight, init 1.0): find the step counts T for which (1 - lr*wd)^T (float32, step by
step as in torch.optim.AdamW: p.mul_(1 - lr*wd)) reproduces the checkpoint value bit-exactly."""
import json
import sys

import torch

STEPS_PER_EPOCH = 204  # ceil(3252 / 16) for Protocol P

obs = {}
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    obs[name] = torch.load(path, map_location="cpu")["model.seghead_d.bn1.weight"][0].item()

out = {name: {"value": v} for name, v in obs.items()}
for lr in (1e-4, 1e-3):
    p = torch.ones(1, dtype=torch.float32)
    traj = []
    for _ in range(12000):
        p.mul_(1 - lr * 1e-2)
        traj.append(p.item())
    for name, v in obs.items():
        exact = [t for t, x in enumerate(traj, 1) if x == v]
        out[name][f"lr{lr:g}_exact_steps"] = exact[:6]
        out[name][f"lr{lr:g}_steps/204"] = [round(t / STEPS_PER_EPOCH, 3) for t in exact[:6]]
print(json.dumps(out, indent=1))
