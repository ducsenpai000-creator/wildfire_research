"""Was the checkpoint initialised from ImageNet? Compare first-layer filter structure with a fresh
Kaiming init and with our from-scratch E0. ImageNet-trained first-layer filters are spatially smooth
(adjacent weights strongly correlated) and colour-coherent across RGB; Kaiming-random ones are not,
and a small-lr fine-tune from random keeps them close to random.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import REPO  # noqa: E402,F401
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402


def stats(w):
    """w: [out, in, k, k] conv weights."""
    w = w.float()
    h_pairs = torch.stack([w[..., :, :-1].flatten(), w[..., :, 1:].flatten()])
    v_pairs = torch.stack([w[..., :-1, :].flatten(), w[..., 1:, :].flatten()])
    spatial_corr = float((torch.corrcoef(h_pairs)[0, 1] + torch.corrcoef(v_pairs)[0, 1]) / 2)
    rgb = w.reshape(w.shape[0], 3, -1)  # first conv: in=3
    ch_corr = float(torch.stack([torch.corrcoef(torch.stack([rgb[:, i].flatten(), rgb[:, j].flatten()]))[0, 1]
                                 for i, j in ((0, 1), (0, 2), (1, 2))]).mean())
    return {"spatial_adjacent_corr": round(spatial_corr, 4), "rgb_channel_corr": round(ch_corr, 4),
            "std": round(float(w.std()), 5)}


def first_conv(sd):
    return sd["model.conv1.0.weight"] if "model.conv1.0.weight" in sd else sd["conv1.0.weight"]


if __name__ == "__main__":
    out = {}
    torch.manual_seed(0)
    for size in ("s", "m", "l"):
        config.MODEL.NAME = "pidnet_" + size
        out[f"fresh_kaiming_{size}"] = stats(get_seg_model(config, True).state_dict()["conv1.0.weight"])
    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        out[name] = stats(first_conv(torch.load(path, map_location="cpu")))
    print(json.dumps(out, indent=1))
