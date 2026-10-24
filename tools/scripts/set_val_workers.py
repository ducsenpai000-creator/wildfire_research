"""Set --val-num-workers for the PIDNet jobs that have not finished yet (validation loader only: no shuffle,
no random augmentation -> identical batches and metrics; only throughput changes)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "configs" / "queue.json"
JOBS = {"train_BRRC_E2a", "train_BRRC_E2b", "train_BRRC_E3", "train_E0IN_m3_seed7", "train_E0IN_m3_seed2026",
        "train_BRRC_best_seed7", "train_BRRC_best_seed2026"}
n_workers = sys.argv[1]
q = json.loads(Q.read_text(encoding="utf-8-sig"))
done = set()
for j in q["jobs"]:
    if j["id"] in JOBS:
        c = j["cmd"]
        c[c.index("--val-num-workers") + 1] = n_workers
        done.add(j["id"])
assert done == JOBS, JOBS - done
Q.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
print("updated:", sorted(done))
