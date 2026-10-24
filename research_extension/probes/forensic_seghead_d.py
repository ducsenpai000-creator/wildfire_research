"""Forensics on checkpoints: parameters of the boundary head (seghead_d) receive zero gradient when
the boundary loss is identically 0 (PROTOCOL_LOCK D-2), so AdamW only applies decoupled weight decay:
    w_T = w_0 * (1 - lr*wd)^T.
BatchNorm weights start at exactly 1 and biases at 0, so lr*wd*T = -ln(w_T) can be read off directly.
"""
import json
import math
import sys

import torch

STEPS_PER_EPOCH = 204  # ceil(3252 / 16)


def analyse(name, path):
    sd = torch.load(path, map_location="cpu")
    out = {"ckpt": name}
    for key in sorted(k for k in sd if k.startswith("model.seghead_d.") and k.endswith((".weight", ".bias"))
                      and ".bn" in k):
        t = sd[key].float()
        out[key] = {"mean": float(t.mean()), "std": float(t.std()), "min": float(t.min()), "max": float(t.max())}
    w = sd["model.seghead_d.bn1.weight"].float()
    uniform = float(w.max() - w.min()) < 1e-6
    out["bn1_weight_all_equal"] = uniform
    if uniform:
        v = float(w.mean())
        lwT = -math.log(v)
        out["lr_wd_T"] = lwT
        out["T_if_lr1e-4_wd1e-2"] = lwT / 1e-6
        out["T_if_lr1e-3_wd1e-2"] = lwT / 1e-5
        out["epochs_if_lr1e-4"] = lwT / 1e-6 / STEPS_PER_EPOCH
        out["epochs_if_lr1e-3"] = lwT / 1e-5 / STEPS_PER_EPOCH
    # compare against a trained head for contrast: main head BN weights are not constant
    m = sd["model.final_layer.bn1.weight"].float()
    out["final_layer_bn1_weight_std"] = float(m.std())
    return out


if __name__ == "__main__":
    res = [analyse(*spec.split("=", 1)) for spec in sys.argv[1:]]
    print(json.dumps(res, indent=1))
