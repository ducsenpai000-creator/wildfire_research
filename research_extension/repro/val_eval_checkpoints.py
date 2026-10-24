"""Validation-only diagnostic: score checkpoints with the exact validation routine used during
training (logical batch 16, upstream batch-pooled IoU + per-image IoU). Never touches the test split.

  python val_eval_checkpoints.py --data-dir <P> --out <json> name=size=path [name=size=path ...]
"""
import argparse
import json
import os
from argparse import Namespace

import torch

from common import REPO, assert_not_test, repo_args
from train_repro import build_model, validate

from datasets.segmentation_data import WFSeg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("ckpts", nargs="+", help="name=size=path")
    a = ap.parse_args()
    assert_not_test("valid")
    os.chdir(REPO)
    wf_args = repo_args(data_dir=a.data_dir, batch_size=16, include_id=False)
    valset = WFSeg(root_dir=a.data_dir, mode="valid", boundary=True, include_id=False, args=wf_args)
    loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=a.num_workers)
    results = {}
    for spec in a.ckpts:
        name, size, path = spec.split("=", 2)
        cfg = Namespace(model_size=size, pos_weight=2.5, micro_batch=4, max_val_batches=0)
        fm = build_model(cfg).cuda()
        res = fm.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        assert not res.missing_keys and not res.unexpected_keys
        results[name] = validate(fm, loader, cfg, "cuda")
        print(name, json.dumps(results[name]), flush=True)
        del fm
        torch.cuda.empty_cache()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
