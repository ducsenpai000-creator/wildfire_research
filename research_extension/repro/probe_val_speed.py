"""Speed probe for the per-epoch validation pass: (1) GPU forward time per image (eval mode, micro 3 vs 8,
random tensors) and (2) validation DataLoader throughput with 1 vs 2 workers (validation split only)."""
import argparse
import os
import time

import torch

from common import assert_not_test, repo_args
from datasets.segmentation_data import WFSeg
from train_repro import REPO, _light_worker, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="D:/wildfire_research/data/protocol_P")
    ap.add_argument("--batches", type=int, default=6)
    p = ap.parse_args()
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 5 * 2**30 / total))
    a = argparse.Namespace(model_size="s", init_imagenet="", pos_weight=2.5, br_rho=0.0, rc_lambda=0.0)
    fm = build_model(a).cuda().eval()
    x = torch.randn(16, 3, 1080, 1920).cuda()
    with torch.no_grad():
        for micro in (3, 8):
            for rep in range(3):
                torch.cuda.synchronize()
                t = time.time()
                for s in range(0, 16, micro):
                    fm.model(x[s:s + micro])
                torch.cuda.synchronize()
            print(f"forward micro {micro}: {(time.time() - t) / 16 * 1000:.1f} ms/img, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del x
    assert_not_test("valid")
    ds = WFSeg(root_dir=p.data_dir, mode="valid", boundary=True, include_id=False,
               args=repo_args(data_dir=p.data_dir, batch_size=16, include_id=False))
    for workers in (1, 2):
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=workers,
                                         worker_init_fn=_light_worker, prefetch_factor=1)
        it = iter(dl)
        next(it)  # worker start-up
        t = time.time()
        for _ in range(p.batches):
            next(it)
        print(f"val loader {workers} worker(s): {(time.time() - t) / (16 * p.batches) * 1000:.1f} ms/img", flush=True)
        del it, dl


if __name__ == "__main__":
    os.chdir(REPO)
    main()
