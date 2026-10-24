"""Extract the Fairdata Subset C package and verify every file against the Metax SHA-256 manifest.

Only bytes are hashed; no image/mask is decoded (test-set lock).
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PREFIX = "/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C/"


def main(zip_path, manifest_path, out_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    expected = {f["file_path"][len("/Boreal-Forest-Fire/"):]: f["checksum"]["value"]
                for f in manifest if f["file_path"].startswith(PREFIX) and not f.get("removed")}
    out = Path(out_dir)
    seen, mismatch, extra = set(), [], []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename.lstrip("/")
            # package paths are relative to the dataset root
            key = name[name.index("Boreal-Forest-Fire-Subset-C/"):] if "Boreal-Forest-Fire-Subset-C/" in name else name
            data = z.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if key not in expected:
                extra.append(name)
                continue
            seen.add(key)
            if digest != expected[key]:
                mismatch.append(key)
                continue
            dst = out / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    missing = sorted(set(expected) - seen)
    print(f"expected={len(expected)} verified={len(seen) - len(mismatch)} "
          f"mismatch={len(mismatch)} missing={len(missing)} extra_in_zip={len(extra)}")
    for tag, lst in (("MISMATCH", mismatch), ("MISSING", missing), ("EXTRA", extra)):
        for x in lst[:20]:
            print(tag, x)
    sys.exit(0 if not (mismatch or missing) else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
