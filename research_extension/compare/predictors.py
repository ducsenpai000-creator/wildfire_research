"""Predictors for the comparison methods, used by the locked evaluator (final_test_eval.py).

Each predictor maps an original image file to a smoke probability map at the evaluation resolution
(1080x1920, same as the upstream WFSeg ground-truth pipeline), following the method's own inference recipe:

  segformer / mask2former (MultiNatSmoke recipe): cv2 BGR->RGB -> PIL -> Resize 512x512 -> ToTensor -> Normalize;
      per-pixel logits at 512x512 (released code) -> bilinear (align_corners=False) to 1080x1920 -> sigmoid.
  fosp (official mmseg pipeline): test_pipeline of the config (keep-ratio resize to 512 long side, normalize),
      model.inference(..., rescale=True) -> sigmoid probability at the original image size -> bilinear to 1080x1920.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL_SIZE = (1080, 1920)


def segformer_or_m2f(kind, pretrained, checkpoint, device):
    sys.path.insert(0, str(HERE))
    from train_hf import IMAGE_SIZE, MEAN, STD, build_model, pixel_logits
    from torchvision import transforms
    model = build_model(kind, pretrained)
    res = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.to(device).eval()
    tf = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(),
                             transforms.Normalize(mean=MEAN, std=STD)])

    @torch.no_grad()
    def predict(path):
        image = Image.fromarray(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        logits = pixel_logits(kind, model, tf(image).unsqueeze(0).to(device))  # [1,1,512,512]
        logits = F.interpolate(logits, size=EVAL_SIZE, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)[0, 0].cpu()
    return predict, sum(p.numel() for p in model.parameters())


def fosp(config, checkpoint, device):
    fosp_root = ROOT / "third_party" / "FoSp"
    for p in (fosp_root, fosp_root / "inpainter_setting"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.path.insert(0, str(HERE))
    from run_fosp import patch_fosp_plugins, patch_mmcv_scatter
    patch_mmcv_scatter()
    patch_fosp_plugins()
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmseg.datasets.pipelines import Compose
    from mmseg.models import build_segmentor
    cfg = Config.fromfile(str(config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.to(device).eval()
    from mmseg.apis.inference import LoadImage
    pipeline = Compose([LoadImage()] + cfg.data.test.pipeline[1:])  # = mmseg.apis.inference_segmentor

    @torch.no_grad()
    def predict(path):
        data = pipeline(dict(img=str(path)))
        data = scatter(collate([data], samples_per_gpu=1), [torch.device(device).index or 0])[0]
        prob = model.inference(data["img"][0], data["img_metas"][0], rescale=True)  # sigmoid, original size
        prob = F.interpolate(prob, size=EVAL_SIZE, mode="bilinear", align_corners=False)
        return prob[0, 0].cpu()
    return predict, sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_predictor(entry, device="cuda"):
    kind = entry["kind"]
    if kind in ("segformer", "mask2former"):
        return segformer_or_m2f(kind, entry["pretrained"], entry["checkpoint"], device)
    if kind == "fosp":
        return fosp(entry["config"], entry["checkpoint"], device)
    raise ValueError(kind)
