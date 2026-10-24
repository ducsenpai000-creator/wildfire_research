"""TRAIN-split statistics of SAM pseudo-labels vs human bounding boxes (no test data).

For each training image with a SAM mask and >=1 human box:
  box_fill      = |mask ∩ union(boxes)| / |union(boxes)|   (how much of the boxed region SAM labels smoke)
  outside_frac  = |mask \\ union(boxes)| / |mask|          (mask pixels outside every box)
  boundary band = fraction of mask / background pixels within r px of the mask edge (after the
                  upstream resize to 1080x1920), for r in {2, 4, 8, 16}
Box formats follow the repository: BFF = YOLO txt (class xc yc w h, normalised), AIFM = Pascal-VOC xml
in 640x480 pixel coordinates.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import assert_not_test  # noqa: E402

SPLIT = "train"


def boxes_bff(path, w, h):
    out = []
    for line in Path(path).read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            xc, yc, bw, bh = (float(v) for v in p[1:])
            out.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def boxes_aifm(path, w, h):
    out = []
    for obj in ET.parse(path).getroot().iter("object"):
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        out.append((x0 / 640 * w, y0 / 480 * h, x1 / 640 * w, y1 / 480 * h))
    return out


def main(data_dir, out_json):
    assert_not_test(SPLIT)
    root = Path(data_dir)
    rows = {"BFF": [], "AIFM": []}
    radii = (2, 4, 8, 16)
    for m in sorted((root / "sam_masks" / SPLIT).iterdir()):
        stem = m.stem
        src = "AIFM" if (root / "images" / SPLIT / f"{stem}.jpeg").exists() else "BFF"
        mask = np.asarray(Image.open(m).convert("L")) > 127
        h, w = mask.shape
        lab = root / "labels" / SPLIT / (f"{stem}.xml" if src == "AIFM" else f"{stem}.txt")
        if not lab.exists():
            continue
        bxs = boxes_aifm(lab, w, h) if src == "AIFM" else boxes_bff(lab, w, h)
        if not bxs or mask.sum() == 0:
            continue
        boxmask = np.zeros_like(mask)
        for x0, y0, x1, y1 in bxs:
            boxmask[max(0, int(round(y0))):int(round(y1)), max(0, int(round(x0))):int(round(x1))] = True
        r = {"box_fill": float((mask & boxmask).sum() / max(boxmask.sum(), 1)),
             "outside_frac": float((mask & ~boxmask).sum() / mask.sum())}
        # boundary bands at the training resolution (upstream resizes to 1080x1920)
        mr = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((1920, 1080), Image.BILINEAR)) > 127
        d_in = ndimage.distance_transform_edt(mr)
        d_out = ndimage.distance_transform_edt(~mr)
        for rad in radii:
            r[f"fg_within_{rad}px"] = float(((d_in <= rad) & mr).sum() / max(mr.sum(), 1))
            r[f"bg_within_{rad}px_of_all_px"] = float(((d_out <= rad) & ~mr).sum() / mr.size)
        rows[src].append(r)
    summary = {}
    for src, rs in rows.items():
        keys = rs[0].keys()
        summary[src] = {"n": len(rs), **{k: {"mean": float(np.mean([x[k] for x in rs])),
                                             "median": float(np.median([x[k] for x in rs])),
                                             "p10": float(np.percentile([x[k] for x in rs], 10)),
                                             "p90": float(np.percentile([x[k] for x in rs], 90))} for k in keys}}
    Path(out_json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
