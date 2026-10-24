"""Insert the pre-registered BRRC jobs (reports/BRRC_DESIGN_PREREGISTRATION.md) into configs/queue.json.
BRRC jobs are training-only; their test evaluation happens once at FINAL_EVALUATION_LOCK."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "configs" / "queue.json"
q = json.loads(Q.read_text(encoding="utf-8-sig"))

BASE = ["research_extension/repro/train_repro.py", "--protocol", "P",
        "--data-dir", "D:/wildfire_research/data/protocol_P", "--manifest", "D:/wildfire_research/data/manifests/protocol_P_manifest.csv",
        "--model-size", "s", "--init-imagenet", "D:/wildfire_research/checkpoints/pidnet_imagenet_converted/PIDNet_S_ImageNet_official_keys.pt",
        "--lr", "1e-4", "--weight-decay", "1e-2", "--pos-weight", "2.5", "--epochs", "50", "--batch-size", "16", "--micro-batch", "3",
        "--num-workers", "2", "--val-num-workers", "1", "--prefetch-factor", "1", "--max-vram-gb", "5", "--cpu-threads", "2"]


def job(jid, exp, seed, extra, select=None):
    j = {"id": jid, "kind": "pidnet", "env": "wfseg", "exp": exp,
         "cmd": [BASE[0], "--exp", exp] + BASE[1:] + ["--seed", str(seed)] + extra}
    if select:
        j["select"] = select
    return j


E1a, E1b = "BRRC_E1a_rho0.10_P_pidnet_s_seed42", "BRRC_E1b_rho0.25_P_pidnet_s_seed42"
E2a, E2b = "BRRC_E2a_lam1_P_pidnet_s_seed42", "BRRC_E2b_lam5_P_pidnet_s_seed42"
E3 = "BRRC_E3_best_P_pidnet_s_seed42"
grid = [
    job("train_BRRC_E1a", E1a, 42, ["--br-rho", "0.10"]),
    job("train_BRRC_E1b", E1b, 42, ["--br-rho", "0.25"]),
    job("train_BRRC_E2a", E2a, 42, ["--rc-lambda", "1", "--rc-alpha", "0.3", "--rc-beta", "0.7"]),
    job("train_BRRC_E2b", E2b, 42, ["--rc-lambda", "5", "--rc-alpha", "0.3", "--rc-beta", "0.7"]),
]
late = [
    job("train_BRRC_E3", E3, 42, ["--br-rho", "{BR_RHO}", "--rc-lambda", "{RC_LAM}", "--rc-alpha", "0.3", "--rc-beta", "0.7"],
        select={"BR_RHO": {E1a: "0.10", E1b: "0.25"}, "RC_LAM": {E2a: "1", E2b: "5"}}),
    job("train_E0IN_m3_seed7", "E0IN_m3_P_pidnet_s_seed7", 7, []),
    job("train_E0IN_m3_seed2026", "E0IN_m3_P_pidnet_s_seed2026", 2026, []),
]
best = {E1a: "--br-rho 0.10", E1b: "--br-rho 0.25", E2a: "--rc-lambda 1", E2b: "--rc-lambda 5", E3: "FROM_META"}
# FROM_META: queue_runner rebuilds E3's BRRC arguments from experiments/<E3>/run_meta.json (values selected at run time).
for seed in (7, 2026):
    late.append(job(f"train_BRRC_best_seed{seed}", f"BRRC_best_P_pidnet_s_seed{seed}", seed, ["{BEST}"], select={"BEST": best}))

ids = [j["id"] for j in q["jobs"]]
assert not any(j["id"] in ids for j in grid + late), "BRRC jobs already present"
pos = ids.index("eval_DSA_m3") + 1
q["jobs"] = q["jobs"][:pos] + grid + q["jobs"][pos:] + late
q["description"] += " BRRC jobs (training only) added 2026-10-18 per BRRC_DESIGN_PREREGISTRATION.md."
Q.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
print([j["id"] for j in q["jobs"]])
