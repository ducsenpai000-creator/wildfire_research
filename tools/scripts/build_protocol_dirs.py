"""Build the Protocol P data directory (Boreal Subset C + remixed AI For Mankind) with hardlinks.

Mirrors the README instruction "extract them to the same directory". No file is modified,
filtered or duplicated in content; name collisions abort the build.
"""
import os
import sys
from pathlib import Path

SUBDIRS = ["images", "labels", "sam_masks", "manual_masks"]
SPLITS = ["train", "valid", "test"]


def main(bff_root, aifm_root, out_root):
    out = Path(out_root)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} not empty; refusing to overwrite")
    n = 0
    for sub in SUBDIRS:
        for split in SPLITS:
            dst_dir = out / sub / split
            for src_root in (Path(bff_root), Path(aifm_root)):
                src_dir = src_root / sub / split
                if not src_dir.is_dir():
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in sorted(src_dir.iterdir()):
                    dst = dst_dir / f.name
                    if dst.exists():
                        sys.exit(f"name collision: {dst}")
                    os.link(f, dst)
                    n += 1
    print(f"linked {n} files into {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
