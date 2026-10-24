"""Reproduction training entry point for the WACV 2025 PIDNet student (Protocol P / B).

Faithful to upstream train.py (AdamW, no scheduler, BCE pos_weight 2.5, loss
0.4*l0 + 20*l1 + 1*l2 + 1*l3, WFSeg augmentation, 1080x1920 input, logical batch 16,
50 epochs) with ONE hardware-driven change: each logical batch of 16 is processed as
micro-batches (gradient accumulation) because batch 16 at 1080x1920 does not fit in 6 GB.
The logical batch composition and per-batch loss weighting are identical to upstream;
BatchNorm statistics are computed per micro-batch (documented deviation).

Validation reproduces upstream eval(): logical val batch 16, "mIoU" = BinaryJaccardIndex
over the whole batch (1.0 if the batch has no positive label), val loss = l1 term with the
training criterion. Both upstream checkpoints are kept: best val loss (<exp>.pt, used by
upstream eval.py) and best val mIoU (<exp>_miou.pt, rule stated in the paper).

TEST-SET LOCK: this script can only build 'train' and 'valid' datasets.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import REPO, ROOT, assert_not_test, git_commit, repo_args, sha256_file, sha256_json

from datasets.segmentation_data import WFSeg  # noqa: E402  (upstream, read-only)
from eval import dice, precision, rand_ind, recall  # noqa: E402
from models.pidnet import get_seg_model  # noqa: E402
from pidnet_utils.configs import config  # noqa: E402
from pidnet_utils.criterion import BondaryLoss  # noqa: E402
from pidnet_utils.utils import FullModel  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--protocol", required=True, choices=["P", "B"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True, help="manifest CSV whose hash is logged")
    ap.add_argument("--model-size", default="s")
    ap.add_argument("--init-imagenet", default="", help="converted PIDNet ImageNet state dict (official key names)")
    ap.add_argument("--lr", type=float, default=1e-4)          # upstream train.py
    ap.add_argument("--weight-decay", type=float, default=1e-2)  # AdamW default used upstream
    ap.add_argument("--pos-weight", type=float, default=2.5)    # upstream train.py
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16, help="logical batch (upstream default)")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="resume from <exp>_state.pt after a crash")
    ap.add_argument("--aux", default="none", choices=["none", "dsa"], help="extra training loss (methods/)")
    ap.add_argument("--dsa-lambda", type=float, default=0.4)   # DSA paper Sec. 4.4
    ap.add_argument("--dsa-k", type=int, default=5)            # DSA paper Sec. 4.4
    ap.add_argument("--dsa-eps", type=float, default=0.05)     # not given in paper (SwAV default)
    ap.add_argument("--br-rho", type=float, default=0.0, help="BRRC boundary reliability: radius factor (0 = off)")
    ap.add_argument("--br-rmin", type=int, default=3)
    ap.add_argument("--br-rmax", type=int, default=40)
    ap.add_argument("--br-wmin", type=float, default=0.0)
    ap.add_argument("--rc-lambda", type=float, default=0.0, help="BRRC recall-calibrated Tversky weight (0 = off)")
    ap.add_argument("--rc-alpha", type=float, default=0.3)
    ap.add_argument("--rc-beta", type=float, default=0.7)
    ap.add_argument("--max-vram-gb", type=float, default=0.0, help="hard cap for this process (0 = none)")
    ap.add_argument("--cpu-threads", type=int, default=2, help="torch CPU threads in the main process")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-batches", type=int, default=0, help="smoke-test only")
    ap.add_argument("--max-val-batches", type=int, default=0, help="smoke-test only")
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    return ap.parse_args()


def seed_all(seed):
    np.random.seed(seed)  # upstream: np.random.seed(42)
    random.seed(seed)
    torch.manual_seed(seed)  # not seeded upstream; added for run-to-run traceability


def build_model(a):
    config.MODEL.NAME = "pidnet_" + a.model_size
    model = get_seg_model(cfg=config, imgnet_pretrained=True)  # upstream: pretrained loading is disabled (if False)
    init = getattr(a, "init_imagenet", "")
    if init:
        # P/I-branch ImageNet weights (official PIDNet, converted by OpenMMLab; key names restored by
        # convert_mmseg_pidnet_imagenet.py). Same effect as upstream's commented-out shape-matched loader.
        sd = torch.load(init, map_location="cpu")
        own = model.state_dict()
        bad = [k for k in sd if k not in own or own[k].shape != sd[k].shape]
        assert not bad, bad[:5]
        model.load_state_dict(sd, strict=False)
        print(f"ImageNet init: loaded {len(sd)} / {len(own)} tensors from {init}", flush=True)
    sem = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([a.pos_weight]).view(1, 1, 1))
    if getattr(a, "br_rho", 0) > 0 or getattr(a, "rc_lambda", 0) > 0:   # BRRC (Stage 2; pre-registered design)
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "research_extension" / "methods"))
        from brrc import BRRCFullModel
        return BRRCFullModel(model, sem, BondaryLoss(), br_rho=a.br_rho, br_rmin=a.br_rmin, br_rmax=a.br_rmax,
                             br_wmin=a.br_wmin, rc_lambda=a.rc_lambda, rc_alpha=a.rc_alpha, rc_beta=a.rc_beta)
    return FullModel(model, sem, BondaryLoss())


def micro_slices(n, micro):
    """Split a logical batch of n into ceil(n/micro) contiguous [start, end) slices of near-equal size,
    each <= micro (so peak VRAM is bounded by `micro`) and >= 2 whenever n >= 2 (BatchNorm needs >1
    sample). For n=16, micro=4 this is identical to fixed-size slicing (4 x 4)."""
    k = max(1, -(-n // micro))
    if n >= 2:
        k = min(k, n // 2)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    return list(zip(bounds[:-1], bounds[1:]))


def _light_worker(_):
    """DataLoader worker init: one CPU thread per worker for torch/OpenCV (CPU load only; results unchanged)."""
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def forward_chunks(fm, img, mask, bd, micro):
    """Upstream FullModel forward on micro-batches, returning concatenated outputs (eval mode)."""
    outs0, outs1 = [], []
    for s in range(0, img.shape[0], micro):
        out = fm.model(img[s:s + micro])
        h, w = img.shape[-2:]
        out = [F.interpolate(o, size=(h, w), mode="bilinear", align_corners=True) for o in out]
        outs0.append(out[0])
        outs1.append(out[1])
    return torch.cat(outs0), torch.cat(outs1)


@torch.no_grad()
def validate(fm, loader, a, device):
    fm.eval()
    tot = {"loss": 0.0, "iou": 0.0, "prec": 0.0, "rec": 0.0, "acc": 0.0, "f1": 0.0}
    per_image_iou = []
    n = 0
    for i, batch in enumerate(loader):
        if a.max_val_batches and i >= a.max_val_batches:
            break
        img, mask, bd = batch["img"].to(device), batch["mask"].to(device), batch["boundary"].to(device)
        _, out1 = forward_chunks(fm, img, mask, bd, a.micro_batch)
        # --- identical to upstream FullModel.forward / eval() on the full logical batch ---
        if len(torch.nonzero(mask)) < 1:
            iou = 1.0
        else:
            iou = fm.iou(preds=torch.sigmoid(out1), target=mask).item()
        loss_s = fm.sem_loss(out1, mask).item()
        pred = torch.flatten(torch.round(torch.sigmoid(out1)))
        gt = torch.flatten(mask)
        tot["loss"] += loss_s
        tot["iou"] += iou
        tot["prec"] += precision(pred, gt).item()
        tot["rec"] += recall(pred.to(torch.uint8), gt.to(torch.uint8)).item()
        tot["acc"] += rand_ind(pred, gt).item()
        tot["f1"] += dice(pred, gt).item()
        # additional diagnostic: sample-wise IoU (paper Eq. 1 semantics, empty GT -> 1)
        pb = torch.sigmoid(out1) > 0.5
        gb = mask > 0.5
        for k in range(pb.shape[0]):
            if gb[k].sum() == 0:
                per_image_iou.append(1.0)
            else:
                inter = (pb[k] & gb[k]).sum().item()
                union = (pb[k] | gb[k]).sum().item()
                per_image_iou.append(inter / union)
        n += 1
    res = {f"val_{k}": v / n for k, v in tot.items()}
    res["val_sample_miou"] = float(np.mean(per_image_iou))
    return res


def main():
    a = parse()
    for m in ("train", "valid"):
        assert_not_test(m)
    device = "cuda"
    torch.set_num_threads(a.cpu_threads)
    if a.max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram_gb * 2**30 / total))
    seed_all(a.seed)
    out = Path(a.out_root) / a.exp
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    wf_args = repo_args(data_dir=a.data_dir, batch_size=a.batch_size, include_id=False)
    trainset = WFSeg(root_dir=a.data_dir, mode="train", boundary=True, include_id=False, args=wf_args)
    valset = WFSeg(root_dir=a.data_dir, mode="valid", boundary=True, include_id=False, args=wf_args)
    assert trainset.mode == "train" and valset.mode == "valid"
    # Windows: every worker re-imports torch/CUDA (~2.4 GB commit each); keep the total under the page-file limit
    def loader(ds, shuffle, workers):
        return torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=workers,
                                           persistent_workers=workers > 0,
                                           worker_init_fn=_light_worker if workers > 0 else None,
                                           prefetch_factor=a.prefetch_factor if workers > 0 else None)
    trainloader = loader(trainset, True, a.num_workers)
    valloader = loader(valset, False, a.val_num_workers)

    fm = build_model(a).to(device)
    aux, feats = None, {}
    if a.aux == "dsa":
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "research_extension" / "methods"))
        from dsa_loss import DSALoss
        emb_dim = fm.model.final_layer.conv1.in_channels  # DFM output fed to the final head
        aux = DSALoss(emb_dim, k=a.dsa_k, eps=a.dsa_eps).to(device)
        fm.model.final_layer.register_forward_hook(lambda mod, inp, out: feats.__setitem__("emb", inp[0]))
    params = list(fm.parameters()) + (list(aux.parameters()) if aux is not None else [])
    optim = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    state_path = out / "checkpoints" / f"{a.exp}_state.pt"
    start_epoch, best_loss, best_iou = 0, float("inf"), -float("inf")
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        fm.load_state_dict(st["model"])
        if aux is not None:
            aux.load_state_dict(st["aux"])
        optim.load_state_dict(st["optim"])
        start_epoch, best_loss, best_iou = st["epoch"] + 1, st["best_loss"], st["best_iou"]
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        print(f"RESUMED from epoch {st['epoch']} (best_loss {best_loss:.6f}, best_iou {best_iou:.6f})", flush=True)

    meta = {"argv": vars(a), "repo_commit": git_commit(REPO), "project_commit": git_commit(ROOT),
            "manifest_sha256": sha256_file(a.manifest), "n_train": len(trainset), "n_valid": len(valset),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "init_imagenet_sha256": sha256_file(a.init_imagenet) if a.init_imagenet else None}
    meta["config_sha256"] = sha256_json({k: v for k, v in vars(a).items()
                                         if k not in ("exp", "max_train_batches", "max_val_batches", "out_root", "resume",
                                                      "max_vram_gb", "cpu_threads")})
    meta_name = "run_meta.json" if start_epoch == 0 else f"run_meta_resume_e{start_epoch}.json"
    (out / meta_name).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1), flush=True)

    log_path = out / "epochs.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_iou", "val_sample_miou", "val_prec", "val_rec",
              "val_acc", "val_f1", "epoch_sec", "peak_vram_gib", "saved"]
    if start_epoch == 0:
        with log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(fields)

    t_start = time.time()
    last_aux = float("nan")
    for epoch in range(start_epoch, a.epochs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fm.train()
        epoch_loss, nb = 0.0, 0
        optim.zero_grad()
        for i, batch in enumerate(trainloader):
            if a.max_train_batches and i >= a.max_train_batches:
                break
            img, mask, bd = batch["img"], batch["mask"], batch["boundary"]
            n = img.shape[0]
            batch_loss = 0.0
            for s, e in micro_slices(n, a.micro_batch):
                m_gpu = mask[s:e].to(device, non_blocking=True)
                losses, _, _, _ = fm(img[s:e].to(device, non_blocking=True), m_gpu,
                                     bd[s:e].to(device, non_blocking=True))
                total = losses.mean()
                if aux is not None:
                    l_dsa = aux(feats.pop("emb"), m_gpu)
                    total = total + a.dsa_lambda * l_dsa   # DSA Eq. 9
                    last_aux = float(l_dsa)
                loss = total * ((e - s) / n)
                loss.backward()
                batch_loss += loss.item()
            optim.step()
            optim.zero_grad()
            epoch_loss += batch_loss
            nb += 1
            if i % 20 == 0:
                aux_txt = f" L_aux {last_aux:.6f}" if aux is not None else ""
                for k, v in getattr(fm, "last_terms", {}).items():
                    aux_txt += f" {k} {v:.4f}"
                print(f"epoch {epoch} batch {i}/{len(trainloader)} loss {batch_loss:.4f}{aux_txt} "
                      f"{time.time() - t0:.0f}s", flush=True)
        vres = validate(fm, valloader, a, device)
        saved = []
        if vres["val_loss"] < best_loss:  # upstream rule -> <exp>.pt (loaded by upstream eval.py)
            best_loss = vres["val_loss"]
            torch.save(fm.state_dict(), out / "checkpoints" / f"{a.exp}.pt")
            saved.append("best_val_loss")
        if vres["val_iou"] > best_iou:  # paper rule -> <exp>_miou.pt
            best_iou = vres["val_iou"]
            torch.save(fm.state_dict(), out / "checkpoints" / f"{a.exp}_miou.pt")
            saved.append("best_val_miou")
        torch.save(fm.state_dict(), out / "checkpoints" / f"{a.exp}_last.pt")
        torch.save({"model": fm.state_dict(), "optim": optim.state_dict(), "epoch": epoch,
                    "aux": aux.state_dict() if aux is not None else None,
                    "best_loss": best_loss, "best_iou": best_iou, "torch_rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state(), "py_rng": random.getstate()}, state_path)
        row = {"epoch": epoch, "train_loss": epoch_loss / max(nb, 1), "val_loss": vres["val_loss"],
               "val_iou": vres["val_iou"], "val_sample_miou": vres["val_sample_miou"],
               "val_prec": vres["val_prec"], "val_rec": vres["val_rec"], "val_acc": vres["val_acc"],
               "val_f1": vres["val_f1"], "epoch_sec": round(time.time() - t0, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "saved": "+".join(saved)}
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in fields])
        print(json.dumps(row), flush=True)

    summary = {"best_val_loss": best_loss, "best_val_iou": best_iou,
               "train_hours": round((time.time() - t_start) / 3600, 3),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    os.chdir(REPO)  # upstream code uses repo-relative paths
    main()
