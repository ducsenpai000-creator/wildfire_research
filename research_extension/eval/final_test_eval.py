"""LOCKED test-set evaluator (the only entry point allowed to read test masks).

  python final_test_eval.py --lock <lock.json>

The lock JSON freezes: checkpoints (+SHA-256), model sizes, evaluator/upstream code hashes,
data manifests (+SHA-256), test data directories, threshold (0.5, upstream) and reporting rule.
The evaluator refuses to run if any hash differs or if results for this lock already exist
(no repeated peeking). Only aggregate metrics are written; nothing per image, no images.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "repro"))
from common import REPO, ROOT, git_commit, sha256_file  # noqa: E402
from evaluator_core import (build_model, chdir_repo, evaluate_model, evaluate_predictor,  # noqa: E402
                            evaluate_sam_vs_manual, make_dataset)

CODE_FILES = {
    "evaluator_core": HERE / "evaluator_core.py",
    "predictors": HERE.parent / "compare" / "predictors.py",
    "train_hf_model_defs": HERE.parent / "compare" / "train_hf.py",
    "final_test_eval": HERE / "final_test_eval.py",
    "upstream_eval": REPO / "eval.py",
    "upstream_dataset": REPO / "datasets" / "segmentation_data.py",
    "upstream_pidnet": REPO / "models" / "pidnet.py",
    "upstream_fullmodel": REPO / "pidnet_utils" / "utils.py",
}


def current_code_hashes():
    return {k: sha256_file(p) for k, p in CODE_FILES.items()}


def verify(lock):
    if lock.get("status") != "LOCKED":
        sys.exit("lock status is not LOCKED")
    problems = []
    for k, h in current_code_hashes().items():
        if lock["code_sha256"].get(k) != h:
            problems.append(f"code changed: {k}")
    for m in lock["models"]:
        if sha256_file(m["checkpoint"]) != m["sha256"]:
            problems.append(f"checkpoint changed: {m['name']}")
        if m.get("config") and sha256_file(m["config"]) != m.get("config_sha256"):
            problems.append(f"config changed: {m['name']}")
    for name, man in lock["manifests"].items():
        if sha256_file(man["path"]) != man["sha256"]:
            problems.append(f"manifest changed: {name}")
    if problems:
        sys.exit("LOCK VIOLATION:\n" + "\n".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    lock_path = Path(a.lock).resolve()  # chdir_repo() below changes the working directory
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    out_path = ROOT / "reports" / "test_results" / f"{lock['lock_id']}.json"
    if out_path.exists():
        sys.exit(f"results for lock {lock['lock_id']} already exist: {out_path} (no re-evaluation)")
    verify(lock)
    chdir_repo()
    results = {"lock_id": lock["lock_id"], "lock_sha256": sha256_file(lock_path),
               "project_commit": git_commit(ROOT), "evaluated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "threshold": 0.5, "datasets": {}}
    for ds_name, ds_dir in lock["test_sets"].items():
        ds, args = make_dataset(ds_dir, "test", manual_masks=True)
        entry = {"n_images": len(ds), "models": {}}
        if lock.get("eval_sam_pseudo_labels"):
            entry["SAM_pseudo_labels"] = evaluate_sam_vs_manual(ds_dir, "test")
        for m in lock["models"]:
            if m.get("kind", "pidnet") == "pidnet":
                fm = build_model(m["size"], m["checkpoint"], a.device)
                entry["models"][m["name"]] = evaluate_model(fm, ds, args, a.device, m["name"])
                del fm
            else:
                sys.path.insert(0, str(HERE.parent / "compare"))
                from predictors import make_predictor
                predict, n_params = make_predictor(m, a.device)
                res = evaluate_predictor(predict, ds, tuple(m.get("thresholds", [0.5])))
                res["n_trainable_params"] = n_params
                entry["models"][m["name"]] = res
                del predict
            torch.cuda.empty_cache()
        results["datasets"][ds_name] = entry
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
