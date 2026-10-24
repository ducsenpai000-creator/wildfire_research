"""Download files of a Fairdata (Etsin/IDA) dataset under a path prefix and verify SHA-256.

Uses the same public flow as the Etsin web UI:
  POST https://etsin.fairdata.fi/api/v3/download/authorize {cr_id, file} -> {url}
  GET  url
The official per-file SHA-256 comes from the Metax v2 file listing saved by the caller.

Only bytes are hashed; file contents are never decoded (safe for test-set files).
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AUTH_URL = "https://etsin.fairdata.fi/api/v3/download/authorize"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cr_id: str, file_path: str, dst: Path, expected: str, retries: int = 8) -> str:
    if dst.exists() and sha256_of(dst) == expected:
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err = None
    for attempt in range(retries):
        try:
            body = json.dumps({"cr_id": cr_id, "file": file_path}).encode()
            req = urllib.request.Request(AUTH_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                url = json.loads(r.read())["url"]
            with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = sha256_of(tmp)
            if got != expected:
                raise IOError(f"sha256 mismatch {got} != {expected}")
            tmp.replace(dst)
            return "downloaded"
        except Exception as e:  # noqa: BLE001 - retry any transient error
            last_err = e
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"{file_path}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Metax v2 files JSON")
    ap.add_argument("--cr-id", required=True)
    ap.add_argument("--prefix", required=True, help="Dataset path prefix to download")
    ap.add_argument("--strip", required=True, help="Path prefix to strip when writing locally")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    todo = [f for f in files if f["file_path"].startswith(args.prefix) and not f.get("removed")]
    print(f"{len(todo)} files under {args.prefix}", flush=True)
    out = Path(args.out)
    stats, failures = {}, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {}
        for f in todo:
            assert f["checksum"]["algorithm"] == "SHA-256"
            rel = f["file_path"][len(args.strip):].lstrip("/")
            futs[ex.submit(fetch, args.cr_id, f["file_path"], out / rel, f["checksum"]["value"])] = f["file_path"]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                s = fut.result()
                stats[s] = stats.get(s, 0) + 1
            except Exception as e:  # noqa: BLE001
                failures.append(str(e))
            if i % 200 == 0:
                print(f"{i}/{len(todo)} {stats} failures={len(failures)}", flush=True)
    print(f"DONE {stats} failures={len(failures)}")
    for msg in failures:
        print("FAIL", msg)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
