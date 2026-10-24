"""Append the FINAL_EVALUATION_LOCK job (reports/FINAL_EVALUATION_LOCK.md) to configs/queue.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "configs" / "queue.json"
q = json.loads(Q.read_text(encoding="utf-8-sig"))
assert "eval_FINAL" not in [j["id"] for j in q["jobs"]]
exps = ["E0IN_m3_P_pidnet_s_seed7", "E0IN_m3_P_pidnet_s_seed2026",
        "BRRC_E1a_rho0.10_P_pidnet_s_seed42", "BRRC_E1b_rho0.25_P_pidnet_s_seed42",
        "BRRC_E2a_lam1_P_pidnet_s_seed42", "BRRC_E2b_lam5_P_pidnet_s_seed42",
        "BRRC_E3_best_P_pidnet_s_seed42", "BRRC_best_P_pidnet_s_seed7", "BRRC_best_P_pidnet_s_seed2026"]
args = []
for e in exps:
    args += ["--model", f"name={e}__primary_best_val_miou,size=s,ckpt=experiments/{e}/checkpoints/{e}_miou.pt",
             "--model", f"name={e}__secondary_best_val_loss,size=s,ckpt=experiments/{e}/checkpoints/{e}.pt"]
args += ["--test-set", "P_all=data/protocol_P", "--test-set", "BFF=data/raw/bff/Boreal-Forest-Fire-Subset-C",
         "--test-set", "AIFM=data/raw/aifm/data",
         "--manifest", "P=data/manifests/protocol_P_manifest.csv", "--manifest", "B=data/manifests/protocol_B_manifest.csv",
         "--skip-missing", "--note", "FINAL_EVALUATION_LOCK (Stage 2 BRRC); see reports/FINAL_EVALUATION_LOCK.md"]
q["jobs"].append({"id": "eval_FINAL", "kind": "eval", "lock_id": "FINAL_BRRC", "lock_args": args})
Q.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
print(len(q["jobs"]), "jobs; last:", q["jobs"][-1]["id"])
