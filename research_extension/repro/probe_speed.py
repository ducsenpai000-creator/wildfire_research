"""Speed probe: PIDNet-S train step (micro-batches 3,3,3,3,2,2 of a 1080x1920 logical batch of 16) with
cudnn.benchmark off vs on. Random tensors only (no dataset is read). Used to decide whether enabling
cudnn.benchmark is worth it; it changes only which cuDNN kernels are picked, not the computation."""
import argparse
import os
import time

import torch

from train_repro import REPO, build_model, micro_slices


def step(fm, optim, micro):
    x = torch.randn(16, 3, 1080, 1920)
    m = (torch.rand(16, 1, 1080, 1920) > 0.7).float()
    b = (torch.rand(16, 1, 1080, 1920) > 0.95).float()
    for s, e in micro_slices(16, micro):
        losses, _, _, _ = fm(x[s:e].cuda(), m[s:e].cuda(), b[s:e].cuda())
        (losses.mean() * ((e - s) / 16)).backward()
    optim.step()
    optim.zero_grad()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--micro", type=int, default=3)
    p = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 5 * 2**30 / total))
    a = argparse.Namespace(model_size="s", init_imagenet="", pos_weight=2.5, br_rho=0.0, rc_lambda=1.0,
                           rc_alpha=0.3, rc_beta=0.7, br_rmin=3, br_rmax=40, br_wmin=0.0)
    fm = build_model(a).cuda().train()
    optim = torch.optim.AdamW(fm.parameters(), lr=1e-4, weight_decay=1e-2)
    for bench in (False, True):
        torch.backends.cudnn.benchmark = bench
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):  # warm-up (and cuDNN autotuning when bench=True)
            step(fm, optim, p.micro)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(p.steps):
            step(fm, optim, p.micro)
        torch.cuda.synchronize()
        print(f"benchmark={bench}: {(time.time() - t) / p.steps:.3f} s/step, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    os.chdir(REPO)
    main()
