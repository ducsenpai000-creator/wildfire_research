"""Convert MMSegmentation's PIDNet ImageNet checkpoints (converted by OpenMMLab from the official PIDNet
repo) back to the official PIDNet parameter names used by the WACV repository, and verify the mapping.

Verification: the authors' released checkpoints were fine-tuned from ImageNet with a small lr (1e-4,
see reports/forensic_lr_exact_match.json), so each mapped tensor should be strongly aligned (cosine)
with the corresponding tensor of the released checkpoint, while a from-scratch run (E0) should not be.

  python convert_mmseg_pidnet_imagenet.py --size s --mmseg <pth> --out <pt> --compare name=path ...
"""
import argparse
import json
import sys

import torch

from common import REPO  # noqa: F401  (adds upstream repo to sys.path)
from models.pidnet import get_seg_model
from pidnet_utils.configs import config

GROUPS = [  # (mmseg prefix, official prefixes in order)
    ("stem.0.", ["conv1.0.", "conv1.1."]),
    ("stem.1.", ["conv1.3.", "conv1.4."]),
    ("stem.2.", ["layer1."]),
    ("stem.4.", ["layer2."]),
    ("i_branch_layers.0.", ["layer3."]),
    ("i_branch_layers.1.", ["layer4."]),
    ("i_branch_layers.2.", ["layer5."]),
    ("compression_1.", ["compression3."]),
    ("compression_2.", ["compression4."]),
    ("pag_1.", ["pag3."]),
    ("pag_2.", ["pag4."]),
    ("p_branch_layers.0.", ["layer3_."]),
    ("p_branch_layers.1.", ["layer4_."]),
    ("p_branch_layers.2.", ["layer5_."]),
]


def convert(mm, size):
    config.MODEL.NAME = "pidnet_" + size
    ref = get_seg_model(config, True).state_dict()
    out, mapping = {}, []
    for mm_pre, off_pres in GROUPS:
        mm_keys = [k for k in mm if k.startswith(mm_pre)]
        off_keys = [k for k in ref if any(k.startswith(p) for p in off_pres)]
        assert len(mm_keys) == len(off_keys), (mm_pre, len(mm_keys), len(off_keys))
        for mk, ok in zip(mm_keys, off_keys):  # both follow module definition order
            assert mm[mk].shape == ref[ok].shape, (mk, ok, mm[mk].shape, ref[ok].shape)
            out[ok] = mm[mk].clone()
            mapping.append((mk, ok))
    unused = [k for k in mm if k not in {m for m, _ in mapping}]
    assert not unused, unused
    return out, mapping, ref


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--mmseg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="*", default=[])
    a = ap.parse_args()
    mm = torch.load(a.mmseg, map_location="cpu")
    mm = mm.get("state_dict", mm)
    sd, mapping, ref = convert(mm, a.size)
    torch.save(sd, a.out)
    report = {"size": a.size, "source": a.mmseg, "n_tensors": len(sd),
              "n_model_tensors": len(ref), "not_initialised": sorted({k.split('.')[0] for k in ref if k not in sd})}
    for spec in a.compare:
        name, path = spec.split("=", 1)
        ck = {k[len("model."):]: v for k, v in torch.load(path, map_location="cpu").items() if k.startswith("model.")}
        cos = [cosine(sd[k], ck[k]) for k in sd if k.endswith("weight") and sd[k].dim() == 4]
        report[f"conv_weight_cosine_vs_{name}"] = {"n": len(cos), "mean": sum(cos) / len(cos), "min": min(cos)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
