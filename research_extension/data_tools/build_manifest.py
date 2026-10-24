"""Build split manifests with SHA-256 and run train/valid integrity checks.

TEST-SET LOCK: files under */test/ are only hashed (bytes); they are never decoded,
displayed or summarised by content.

Usage:
  python build_manifest.py --data-dir data/protocol_P --protocol P --out data/manifests
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SUBDIRS = ["images", "labels", "sam_masks", "manual_masks"]
SPLITS = ["train", "valid", "test"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem(name):
    # mirrors WFSeg: strip '.jpg' (4 chars) or '.jpeg' (5 chars)
    return name[:-5] if name.lower().endswith(".jpeg") else name.rsplit(".", 1)[0]


def source_of(name):
    return "AIFM" if name.lower().endswith(".jpeg") else "BFF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in SUBDIRS:
        for split in SPLITS:
            d = root / sub / split
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                rows.append({"subdir": sub, "split": split, "file": f.name,
                             "bytes": f.stat().st_size, "sha256": sha256(f)})
    man_path = out / f"protocol_{args.protocol}_manifest.csv"
    with man_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest_hash = sha256(man_path)

    report = {"protocol": args.protocol, "manifest": str(man_path), "manifest_sha256": manifest_hash,
              "counts": Counter(f"{r['subdir']}/{r['split']}" for r in rows)}

    # Test: names only (pairing by filename is metadata, not content)
    imgs_test = [r["file"] for r in rows if r["subdir"] == "images" and r["split"] == "test"]
    man_test = {stem(r["file"]) for r in rows if r["subdir"] == "manual_masks" and r["split"] == "test"}
    report["test"] = {"n_images": len(imgs_test),
                      "n_images_by_source": Counter(source_of(x) for x in imgs_test),
                      "n_manual_masks": len(man_test),
                      "images_without_manual_mask": sorted(stem(x) for x in imgs_test if stem(x) not in man_test),
                      "note": "test content not decoded (test-set lock)"}

    for split in ("train", "valid"):
        imgs = [r["file"] for r in rows if r["subdir"] == "images" and r["split"] == split]
        sams = {stem(r["file"]) for r in rows if r["subdir"] == "sam_masks" and r["split"] == split}
        labels = {stem(r["file"]) for r in rows if r["subdir"] == "labels" and r["split"] == split}
        img_stems = {stem(x) for x in imgs}
        st = {"n_images": len(imgs), "n_by_source": Counter(source_of(x) for x in imgs),
              "n_sam_masks": len(sams), "n_labels": len(labels),
              "images_without_sam_mask": sorted(img_stems - sams),
              "sam_masks_without_image": sorted(sams - img_stems),
              "images_without_label": sorted(img_stems - labels)}
        sizes, modes, msizes, mmodes, mvals = Counter(), Counter(), Counter(), Counter(), Counter()
        empty_masks, nonbinary, size_mismatch, fg_frac = [], [], [], {"BFF": [], "AIFM": []}
        for name in imgs:
            with Image.open(root / "images" / split / name) as im:
                sizes[f"{source_of(name)} {im.size[0]}x{im.size[1]}"] += 1
                modes[im.mode] += 1
                isize = im.size
            s = stem(name)
            if s not in sams:
                continue
            with Image.open(root / "sam_masks" / split / f"{s}.png") as m:
                msizes[f"{m.size[0]}x{m.size[1]}"] += 1
                mmodes[m.mode] += 1
                a = np.asarray(m.convert("L"))
            u = tuple(int(v) for v in np.unique(a))
            mvals[str(u)] += 1
            if not set(u) <= {0, 255}:
                nonbinary.append(s)
            if (a > 127).sum() == 0:
                empty_masks.append(s)
            if m.size != isize:
                size_mismatch.append(s)
            fg_frac[source_of(name)].append(float((a > 127).mean()))
        st.update({"image_sizes": sizes, "image_modes": modes, "mask_sizes": msizes, "mask_modes": mmodes,
                   "mask_unique_value_sets": mvals, "empty_sam_masks": empty_masks,
                   "nonbinary_sam_masks": nonbinary, "mask_image_size_mismatch": size_mismatch,
                   "fg_fraction_mean_by_source": {k: (float(np.mean(v)) if v else None) for k, v in fg_frac.items()},
                   "fg_fraction_median_by_source": {k: (float(np.median(v)) if v else None) for k, v in fg_frac.items()}})
        report[split] = st

    rep_path = out / f"protocol_{args.protocol}_integrity.json"
    rep_path.write_text(json.dumps(report, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(report, indent=1, default=dict)[:6000])


if __name__ == "__main__":
    main()
