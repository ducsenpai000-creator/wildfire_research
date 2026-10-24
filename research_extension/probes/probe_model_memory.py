"""Probe: (1) official WACV checkpoints load into the repo's PIDNet/FullModel;
(2) GPU forward/backward + peak VRAM at 1080x1920 for several batch sizes.
Uses random tensors only (no dataset access). The upstream repo is imported read-only.
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2] / "third_party" / "wildfire-real-time-segmentation"
sys.path.insert(0, str(REPO))

from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
from pidnet_utils.criterion import BondaryLoss  # noqa: E402
from pidnet_utils.utils import FullModel  # noqa: E402


def build(size, pos_weight=2.5):
    config.MODEL.NAME = "pidnet_" + size
    model = get_seg_model(cfg=config, imgnet_pretrained=True)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).view(1, 1, 1))
    return FullModel(model, crit, BondaryLoss())


def check_ckpt(path, size):
    fm = build(size)
    sd = torch.load(path, map_location="cpu")
    res = fm.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in fm.model.parameters())
    return {"ckpt": Path(path).name, "size": size, "missing": len(res.missing_keys),
            "unexpected": len(res.unexpected_keys), "missing_sample": res.missing_keys[:5],
            "unexpected_sample": res.unexpected_keys[:5], "n_params_model": n_params}


def mem_probe(size, bs, amp):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fm = build(size).cuda().train()
    opt = torch.optim.AdamW(fm.parameters(), lr=1e-4)
    x = torch.randn(bs, 3, 1080, 1920, device="cuda")
    y = (torch.rand(bs, 1, 1080, 1920, device="cuda") > 0.8).float()
    b = (torch.rand(bs, 1, 1080, 1920, device="cuda") > 0.95).float()
    try:
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, out, acc, _ = fm(x, y, b)
            loss.mean().backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        return {"size": size, "bs": bs, "amp": amp, "ok": True,
                "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "out_shape": list(out[1].shape)}
    except torch.cuda.OutOfMemoryError:
        return {"size": size, "bs": bs, "amp": amp, "ok": False}
    finally:
        del fm, opt, x, y, b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ck = Path(sys.argv[1])
    results = {"ckpt_checks": [check_ckpt(ck / f"{n}.pt", s) for n, s in
                               [("sam_sup_pidnet_s", "s"), ("sam_sup_pidnet_m", "m"),
                                ("sam_sup_pidnet_l", "l"), ("loss_ablation_pidnet_s", "s")]]}
    results["mem"] = [mem_probe("s", bs, amp) for amp in (False, True) for bs in (2, 3, 4, 8)]
    print(json.dumps(results, indent=1))
