"""Create a test-evaluation lock JSON (hashes of code, checkpoints, manifests).

  python make_lock.py --lock-id S1a_official --out reports/locks/S1a_official.json \
      --model name=official_sam_sup_pidnet_s,size=s,ckpt=checkpoints/official_wacv/sam_sup_pidnet_s.pt \
      --test-set P_all=data/protocol_P --test-set BFF=data/raw/bff/Boreal-Forest-Fire-Subset-C \
      --manifest P=data/manifests/protocol_P_manifest.csv --eval-sam --note "..."
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "repro"))
from common import ROOT, git_commit, sha256_file  # noqa: E402
from final_test_eval import current_code_hashes  # noqa: E402


def kv(s):
    return dict(p.split("=", 1) for p in s.split(","))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--test-set", action="append", default=[])
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--eval-sam", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--skip-missing", action="store_true",
                    help="record models whose checkpoint does not exist (failed runs) instead of aborting")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; locks are immutable")
    models, missing = [], []
    for m in a.model:
        d = kv(m)
        ck = str((ROOT / d["ckpt"]).resolve())
        if a.skip_missing and not Path(ck).exists():
            missing.append({"name": d["name"], "checkpoint": ck})
            continue
        entry = {"name": d["name"], "size": d.get("size", ""), "kind": d.get("kind", "pidnet"),
                 "checkpoint": ck, "sha256": sha256_file(ck)}
        if d.get("pretrained"):
            entry["pretrained"] = d["pretrained"]
        if d.get("config"):
            cfgp = str((ROOT / d["config"]).resolve())
            entry["config"], entry["config_sha256"] = cfgp, sha256_file(cfgp)
        if d.get("thresholds"):
            entry["thresholds"] = [float(t) for t in d["thresholds"].split(";")]
        models.append(entry)
    test_sets = {k: str((ROOT / v).resolve()) for k, v in (t.split("=", 1) for t in a.test_set)}
    manifests = {}
    for m in a.manifest:
        k, v = m.split("=", 1)
        p = str((ROOT / v).resolve())
        manifests[k] = {"path": p, "sha256": sha256_file(p)}
    lock = {"lock_id": a.lock_id, "status": "LOCKED", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "project_commit": git_commit(ROOT), "code_sha256": current_code_hashes(), "models": models,
            "test_sets": test_sets, "manifests": manifests, "eval_sam_pseudo_labels": a.eval_sam,
            "threshold": 0.5, "reporting": "aggregate metrics only (upstream eval() numbers are official; "
                                            "independent re-implementation reported as cross-check)",
            "note": a.note, "missing_models_not_evaluated": missing}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=1), encoding="utf-8")
    print(json.dumps(lock, indent=1))


if __name__ == "__main__":
    main()
