"""Build a PASCAL-VOC style *view* of Protocol P (train + valid only; test is never included) for
MMSegmentation-based methods (FoSp).

  JPEGImages/<stem>.jpg          hardlink to the original image (AIFM .jpeg files get a .jpg name; bytes identical)
  SegmentationClass/<stem>.png   class-index mask: SAM pseudo-label 0/255 -> 0/1 (same pixels); images without a
                                 SAM mask get an all-zero mask (identical to the upstream WFSeg loader behaviour)
  ImageSets/Segmentation/{train,val}.txt

Each converted mask is verified pixel-for-pixel against its source.
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repro"))
from common import assert_not_test  # noqa: E402

SPLITS = {"train": "train", "valid": "val"}


def stem(name):
    return name[:-5] if name.lower().endswith(".jpeg") else name.rsplit(".", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.data_dir), Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    for d in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for split, voc_split in SPLITS.items():
        assert_not_test(split)
        stems = []
        for img in sorted((src / "images" / split).iterdir()):
            s = stem(img.name)
            stems.append(s)
            dst_img = out / "JPEGImages" / f"{s}.jpg"
            os.link(img, dst_img)
            with Image.open(img) as im:
                w, h = im.size
            sam = src / "sam_masks" / split / f"{s}.png"
            if sam.exists():
                m = np.asarray(Image.open(sam).convert("L"))
                lab = (m > 127).astype(np.uint8)
                assert set(np.unique(m)) <= {0, 255}, sam
                kind = "sam"
            else:
                lab = np.zeros((h, w), np.uint8)
                kind = "no_sam_mask_zero"
            dst_m = out / "SegmentationClass" / f"{s}.png"
            Image.fromarray(lab).save(dst_m)
            back = np.asarray(Image.open(dst_m))
            assert back.shape == (h, w) and np.array_equal(back, lab), dst_m
            rows.append({"split": voc_split, "stem": s, "image_src": str(img), "mask_kind": kind,
                         "fg_pixels": int(lab.sum())})
        (out / "ImageSets" / "Segmentation" / f"{voc_split}.txt").write_text("\n".join(stems) + "\n")
    with open(out / "view_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print({sp: sum(r["split"] == sp for r in rows) for sp in ("train", "val")},
          {k: sum(r["mask_kind"] == k for r in rows) for k in ("sam", "no_sam_mask_zero")})


if __name__ == "__main__":
    main()
