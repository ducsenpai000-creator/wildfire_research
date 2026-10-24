"""SegFormer / Mask2Former on Protocol P, following the MultiNatSmoke (WACV 2026) training recipe
(paper Sec. 4.2 + released code github.com/henryzhao0615/MultiNatSmoke @ f3ab188, code/src/{segformer,mask2former}/train.py):

  * images: cv2 BGR->RGB -> PIL -> Resize((512,512)) -> ToTensor -> Normalize(ImageNet); no augmentation
  * masks:  (mask > 128) -> cv2.resize 512x512 INTER_NEAREST
  * BCEWithLogitsLoss on logits upsampled (bilinear, align_corners=False) to 512x512
  * AdamW lr 1e-4, weight decay 0.01; 20 epochs, batch 32 (paper text; released code uses 15 epochs and
    batch 16 for Mask2Former - paper text followed, deviation recorded)
  * SegFormer: MiT-b2 ImageNet backbone (paper text; code path says mit-b5); Mask2Former: Swin backbone
    pretrained on ADE20K (size not stated -> Swin-Tiny, to fit the 5 GB VRAM budget), per-pixel logits =
    sum_q mask_logits_q * softmax(class_logits_q)[1] exactly as in the released code
  * checkpoint: best validation IoU of the released `compute_metrics` (pixels pooled per batch of 32, averaged
    over batches). MultiNatSmoke selected on its *test* set; here selection uses Protocol P *validation*
    (SAM pseudo-labels) only.

Protocol P data: train/valid image lists = all files in images/<split> (as the upstream WFSeg loader);
images without a SAM mask get an all-zero mask (as upstream). The logical batch of 32 is processed in
micro-batches (gradient accumulation). The backbones use LayerNorm, but the SegFormer decode head contains
one BatchNorm layer (and Mask2Former's pixel decoder GroupNorm), so for SegFormer the head BatchNorm sees
micro-batch statistics instead of 32 images (documented deviation; the loss/gradient composition is otherwise
identical to batch 32).
TEST-SET LOCK: only 'train' and 'valid' are ever built here.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import ROOT, assert_not_test, git_commit, sha256_file, sha256_json  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMAGE_SIZE = (512, 512)


def stem(name):
    return name[:-5] if name.lower().endswith(".jpeg") else name.rsplit(".", 1)[0]


class PSplit(torch.utils.data.Dataset):
    def __init__(self, data_dir, split):
        assert_not_test(split)
        self.img_dir = Path(data_dir) / "images" / split
        self.mask_dir = Path(data_dir) / "sam_masks" / split
        self.items = sorted(self.img_dir.iterdir())
        self.tf = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(),
                                      transforms.Normalize(mean=MEAN, std=STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]
        image = Image.fromarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        mp = self.mask_dir / f"{stem(p.name)}.png"
        if mp.exists():
            mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
        else:
            mask = np.zeros((image.height, image.width), np.uint8)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        return self.tf(image), torch.as_tensor(mask, dtype=torch.long)


def light(_):
    """DataLoader worker init: one CPU thread per worker (CPU budget; results unchanged)."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def build_model(kind, pretrained):
    id2label, label2id = {1: "smoke"}, {"smoke": 1}
    if kind == "segformer":
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    from transformers import Mask2FormerForUniversalSegmentation
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, num_labels=len(id2label), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)


def pixel_logits(kind, model, images, size=IMAGE_SIZE):
    """Released-code logic: per-pixel smoke logit map [B,1,H,W] at `size`."""
    if kind == "segformer":
        out = model(pixel_values=images)
        return F.interpolate(out.logits, size=size, mode="bilinear", align_corners=False)
    out = model(pixel_values=images)
    up = F.interpolate(out.masks_queries_logits, size=size, mode="bilinear", align_corners=False)
    smoke_scores = torch.softmax(out.class_queries_logits, dim=-1)[..., 1].unsqueeze(-1).unsqueeze(-1)
    return (up * smoke_scores).sum(dim=1, keepdim=True)


def compute_metrics(outputs, masks, threshold=0.5, eps=1e-6):
    """Verbatim from the released MultiNatSmoke code (pixels pooled over the batch)."""
    probs = torch.sigmoid(outputs).detach()
    preds = (probs > threshold).float()
    masks = masks.float()
    preds_flat, masks_flat = preds.view(-1), masks.view(-1)
    tp = (preds_flat * masks_flat).sum()
    fp = (preds_flat * (1 - masks_flat)).sum()
    fn = ((1 - preds_flat) * masks_flat).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    mse = torch.mean((probs.view(-1) - masks_flat) ** 2)
    return {"iou": iou.item(), "precision": precision.item(), "recall": recall.item(), "f1": f1.item(),
            "mse": mse.item()}


@torch.no_grad()
def validate(kind, model, loader, micro, device, crit):
    model.eval()
    sums = {"iou": 0, "precision": 0, "recall": 0, "f1": 0, "mse": 0}
    nb, vloss, per_img = 0, 0.0, []
    for images, masks in loader:  # logical batch 32 (released code), forwarded in micro-batches
        outs = torch.cat([pixel_logits(kind, model, images[s:s + micro].to(device))
                          for s in range(0, images.shape[0], micro)])
        m = masks.to(device)
        bm = compute_metrics(outs, m)
        for k in sums:
            sums[k] += bm[k]
        vloss += crit(outs, m.unsqueeze(1).float()).item()
        pred, gt = torch.sigmoid(outs)[:, 0] > 0.5, m > 0
        for k in range(pred.shape[0]):
            if gt[k].sum() == 0:
                per_img.append(1.0)
            else:
                per_img.append(((pred[k] & gt[k]).sum() / (pred[k] | gt[k]).sum()).item())
        nb += 1
    res = {f"val_{k}": v / nb for k, v in sums.items()}
    res["val_loss"] = vloss / nb
    res["val_sample_miou"] = float(np.mean(per_img))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=["segformer", "mask2former"])
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-vram-gb", type=float, default=5.0)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--max-val-batches", type=int, default=0)
    ap.add_argument("--out-root", default=str(ROOT / "experiments"))
    a = ap.parse_args()

    torch.set_num_threads(a.cpu_threads)
    cv2.setNumThreads(1)
    device = "cuda"
    if a.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram_gb * 2**30 /
                                                       torch.cuda.get_device_properties(0).total_memory))
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out_root) / a.exp
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    tr, va = PSplit(a.data_dir, "train"), PSplit(a.data_dir, "valid")
    kw = dict(num_workers=a.num_workers, worker_init_fn=light, persistent_workers=a.num_workers > 0)
    trainloader = torch.utils.data.DataLoader(tr, batch_size=a.batch_size, shuffle=True, **kw)
    valloader = torch.utils.data.DataLoader(va, batch_size=a.batch_size, shuffle=False, **kw)

    model = build_model(a.model, a.pretrained).to(device)
    crit = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    state_path = out / "checkpoints" / f"{a.exp}_state.pt"
    start, best = 0, -1.0
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(st["model"])
        optim.load_state_dict(st["optim"])
        start, best = st["epoch"] + 1, st["best_iou"]
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        print(f"RESUMED from epoch {st['epoch']} (best_iou {best:.6f})", flush=True)

    meta = {"argv": vars(a), "project_commit": git_commit(ROOT), "manifest_sha256": sha256_file(a.manifest),
            "n_train": len(tr), "n_valid": len(va), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__, "gpu": torch.cuda.get_device_name(0),
            "n_params": sum(p.numel() for p in model.parameters()),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_sha256": sha256_json({k: v for k, v in vars(a).items() if k not in (
                "exp", "resume", "max_train_batches", "max_val_batches", "out_root", "max_vram_gb", "cpu_threads",
                "num_workers")})}
    (out / ("run_meta.json" if start == 0 else f"run_meta_resume_e{start}.json")).write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1), flush=True)
    fields = ["epoch", "train_loss", "val_loss", "val_iou", "val_sample_miou", "val_precision", "val_recall",
              "val_f1", "val_mse", "epoch_sec", "peak_vram_gib", "saved"]
    log_path = out / "epochs.csv"
    if start == 0:
        with log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(fields)

    t_all = time.time()
    for epoch in range(start, a.epochs):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train()
        run_loss, nb = 0.0, 0
        for i, (images, masks) in enumerate(trainloader):
            if a.max_train_batches and i >= a.max_train_batches:
                break
            n = images.shape[0]
            optim.zero_grad()
            bl = 0.0
            for s in range(0, n, a.micro_batch):
                x = images[s:s + a.micro_batch].to(device, non_blocking=True)
                y = masks[s:s + a.micro_batch].to(device, non_blocking=True).unsqueeze(1).float()
                loss = crit(pixel_logits(a.model, model, x), y) * (x.shape[0] / n)
                loss.backward()
                bl += loss.item()
            optim.step()
            run_loss += bl
            nb += 1
            if i % 10 == 0:
                print(f"epoch {epoch} batch {i}/{len(trainloader)} loss {bl:.4f} {time.time() - t0:.0f}s", flush=True)
        if a.max_val_batches:
            sub = torch.utils.data.Subset(va, list(range(min(len(va), a.max_val_batches * a.batch_size))))
            vl = torch.utils.data.DataLoader(sub, batch_size=a.batch_size, shuffle=False, num_workers=0)
        else:
            vl = valloader
        v = validate(a.model, model, vl, a.micro_batch, device, crit)
        saved = []
        if v["val_iou"] > best:
            best = v["val_iou"]
            torch.save(model.state_dict(), out / "checkpoints" / f"{a.exp}_best.pt")
            saved.append("best_val_iou")
        torch.save(model.state_dict(), out / "checkpoints" / f"{a.exp}_last.pt")
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "best_iou": best,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                    "py_rng": random.getstate()}, state_path)
        row = {"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_loss": v["val_loss"],
               "val_iou": v["val_iou"], "val_sample_miou": v["val_sample_miou"],
               "val_precision": v["val_precision"], "val_recall": v["val_recall"], "val_f1": v["val_f1"],
               "val_mse": v["val_mse"], "epoch_sec": round(time.time() - t0, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "saved": "+".join(saved)}
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in fields])
        print(json.dumps(row), flush=True)
    summary = {"best_val_iou": best, "train_hours": round((time.time() - t_all) / 3600, 3),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
