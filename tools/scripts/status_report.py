"""Daily status report for the user (Vietnamese): which models finished, results, crashes, queue.

Reads only files (no GPU): experiments/*/{epochs.csv,attempts.log,summary.json,run_meta*.json},
reports/test_results/*.json, reports/queue_status.json, configs/experiments_registry.json.
Writes reports/STATUS_REPORT.csv and reports/STATUS_REPORT_<YYYY-MM-DD>.docx.
"""
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
REP = ROOT / "reports"
REGISTRY = ROOT / "configs" / "experiments_registry.json"

PAPER = {  # published WACV 2025 numbers (Table 2 row 10, Table 4 row 5) for reference only
    "P_all": {"mIoU": 0.594, "precision": 0.891, "recall": 0.645, "f1": 0.707},
    "BFF": {"mIoU": 0.689}, "AIFM": {"mIoU": 0.498},
}


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def read_csv(p):
    try:
        with open(p, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def experiment_rows(registry, queue):
    rows = []
    running = queue.get("current_exp") if queue else None
    for d in sorted(p for p in EXP.iterdir() if p.is_dir() and not p.name.startswith("_")):
        reg = registry.get(d.name, {})
        epochs = read_csv(d / "epochs.csv")
        if not epochs:  # mmseg runner (FoSp): validation entries of *.log.json, one per evaluation interval
            for lj in sorted(d.glob("*.log.json")):
                for line in lj.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("mode") == "val":
                        epochs.append({"epoch": r.get("iter"), **{k: v for k, v in r.items() if isinstance(v, (int, float))}})
        summary = load_json(d / "summary.json")
        meta = load_json(d / "run_meta.json", {}) or {}
        argv = meta.get("argv", {})
        total = argv.get("epochs") or reg.get("epochs")
        att = (d / "attempts.log").read_text(encoding="utf-8-sig", errors="replace") if (d / "attempts.log").exists() else ""
        exits = [int(c) for c in re.findall(r"exit code (-?\d+)", att)]
        launch_errors = [c for c in exits if c == 2]   # argparse/launcher errors: no training step executed
        stopped = [c for c in exits if c in (-1, 4294967295, 1073807364)]   # pause/manual kill, or host shutdown (0x40010004)
        crashes = [c for c in exits if c not in (0, 2, -1, 4294967295, 1073807364)]
        logs = sorted(d.glob("train*.log"), key=lambda p: p.stat().st_mtime)
        recently_active = bool(logs) and (time.time() - logs[-1].stat().st_mtime) < 1800
        resumes = len(re.findall(r"--resume", att))
        notes = re.findall(r"\] ((?:host reboot|attempt \d+ failed|attempts \d|user approved|PAUSED)[^\n]*)", att)
        if summary:
            status = "DONE"
        elif d.name == running or recently_active:
            status = "RUNNING"
        elif "PAUSED" in att.splitlines()[-1] if att.strip() else False:
            status = "PAUSED"
        elif "crashed" in d.name:
            status = "CRASHED (bá», Ä‘Ã£ cháº¡y láº¡i)"
        elif crashes:
            status = "STOPPED/CRASHED"
        else:
            status = "INCOMPLETE"
        best = None
        if epochs:
            key = reg.get("val_key", "val_iou")
            best = max(epochs, key=lambda r: float(r.get(key) or -1))
        hours = summary.get("train_hours") if summary else (
            round(sum(float(r.get("epoch_sec") or 0) for r in epochs) / 3600, 2) if epochs else None)
        rows.append({
            "experiment": d.name, "method": reg.get("method", ""), "role": reg.get("role", ""),
            "status": status, "epochs_done": len(epochs), "epochs_total": total,
            "best_val_metric": f"{float(best[reg.get('val_key', 'val_iou')]):.4f}" if best else "",
            "best_val_epoch": best["epoch"] if best else "",
            "val_metric_name": reg.get("val_key", "val_iou"),
            "train_hours": hours, "n_crashes": len(crashes), "n_paused_or_killed": len(stopped),
            "n_launch_errors": len(launch_errors),
            "n_resumes": resumes,
            "last_crash_or_note": notes[-1][:220] if notes else "",
            "test_lock": reg.get("test_lock", ""),
        })
    return rows


def test_rows(registry):
    out = []
    for f in sorted((REP / "test_results").glob("*.json")):
        j = load_json(f, {})
        for ds, entry in j.get("datasets", {}).items():
            items = list(entry.get("models", {}).items())
            if entry.get("SAM_pseudo_labels"):
                items.insert(0, ("SAM_pseudo_labels", entry["SAM_pseudo_labels"]))
            expanded = []
            for name, res in items:
                if "by_threshold" in res:
                    for t, agg in res["by_threshold"].items():
                        expanded.append((f"{name} @thr {t}", {"n_images": res.get("n_images"), "independent": agg}))
                else:
                    expanded.append((name, res))
            for name, res in expanded:
                m = res.get("upstream") or {}
                ind = res.get("independent") or {}
                out.append({"lock": j.get("lock_id"), "test_set": ds, "model": name, "n_images": res.get("n_images"),
                            "mIoU": round(m.get("mean_iou", ind.get("mean_iou", float("nan"))), 4),
                            "accuracy": round(m.get("mean_rand", ind.get("mean_accuracy", float("nan"))), 4),
                            "precision": round(m.get("mean_precision", ind.get("mean_precision", float("nan"))), 4),
                            "recall": round(m.get("mean_recall", ind.get("mean_recall", float("nan"))), 4),
                            "f1": round(m.get("mean_f1", ind.get("mean_f1", float("nan"))), 4),
                            "paper_mIoU_PIDNet_S": PAPER.get(ds, {}).get("mIoU", "")})
    return out


def ablation_rows():
    """reports/ablation_results.csv: every PIDNet-family Stage-2 run (baseline seeds, DSA, BRRC grid/seeds):
    configuration, best validation, and locked test results when available (all runs, including poor ones)."""
    tests = {}
    for f in (REP / "test_results").glob("*.json"):
        j = load_json(f, {})
        for ds, entry in j.get("datasets", {}).items():
            for name, res in entry.get("models", {}).items():
                exp, _, ck = name.partition("__")
                if not ck:  # S2 locks name models "<exp-short>_primary..." -> map by prefix below
                    for prefix, e in (("E0IN_m3_", "E0IN_m3_P_pidnet_s_seed42"), ("DSA_m3_", "DSA_m3_P_pidnet_s_seed42")):
                        if name.startswith(prefix):
                            exp, ck = e, name[len(prefix):]
                u = res.get("upstream") or {}
                tests[(exp, ck.split("_")[0], ds)] = (u.get("mean_iou"), u.get("mean_precision"), u.get("mean_recall"),
                                                      j.get("lock_id"))
    rows = []
    for d in sorted(p for p in EXP.iterdir() if p.is_dir() and (p.name.startswith(("BRRC_", "E0IN_m3", "DSA_m3")))):
        meta = load_json(d / "run_meta.json", {}) or {}
        a = meta.get("argv", {})
        ep = read_csv(d / "epochs.csv")
        best = max(ep, key=lambda r: float(r["val_iou"])) if ep else None
        row = {"experiment": d.name, "seed": a.get("seed"), "br_rho": a.get("br_rho", 0), "rc_lambda": a.get("rc_lambda", 0),
               "aux": a.get("aux", "none"), "micro_batch": a.get("micro_batch"), "epochs_done": len(ep),
               "best_val_batch_mIoU": round(float(best["val_iou"]), 4) if best else "",
               "best_val_epoch": best["epoch"] if best else "",
               "status": "DONE" if (d / "summary.json").exists() else "INCOMPLETE"}
        for ck in ("primary", "secondary"):
            for ds in ("P_all", "BFF", "AIFM"):
                t = tests.get((d.name, ck, ds))
                row[f"test_{ds}_{ck}_mIoU"] = round(t[0], 4) if t and t[0] is not None else ""
                if ds == "P_all":
                    row[f"test_P_all_{ck}_recall"] = round(t[2], 4) if t and t[2] is not None else ""
                    row[f"test_lock_{ck}"] = t[3] if t else ""
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_docx(path, exp_rows, t_rows, queue):
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("BÃ¡o cÃ¡o tiáº¿n Ä‘á»™ â€” Wildfire smoke segmentation (Protocol P)", 0)
    doc.add_paragraph(f"Táº¡o lÃºc: {datetime.now():%Y-%m-%d %H:%M}. File mÃ¡y Ä‘á»c: reports/STATUS_REPORT.csv, "
                      f"reports/STATUS_REPORT_tests.csv.")
    done = [r for r in exp_rows if r["status"] == "DONE"]
    crashed = [r for r in exp_rows if r["n_crashes"]]
    doc.add_heading("TÃ³m táº¯t", 1)
    doc.add_paragraph(f"ÄÃ£ xong: {len(done)} thÃ­ nghiá»‡m ({', '.join(r['experiment'] for r in done) or 'chÆ°a cÃ³'}).")
    if queue:
        doc.add_paragraph(f"HÃ ng Ä‘á»£i: Ä‘ang cháº¡y = {queue.get('current_exp') or 'khÃ´ng'}; "
                          f"cÃ²n chá» = {', '.join(queue.get('pending', [])) or 'khÃ´ng'}; "
                          f"cáº­p nháº­t {queue.get('updated', '')}.")
        for e in queue.get("events", [])[-15:]:
            doc.add_paragraph(e, style="List Bullet")
    doc.add_paragraph(f"Sá»‘ thÃ­ nghiá»‡m tá»«ng bá»‹ crash/dá»«ng báº¥t thÆ°á»ng: {len(crashed)} "
                      f"({', '.join(r['experiment'] for r in crashed) or 'khÃ´ng'}).")

    doc.add_heading("Tráº¡ng thÃ¡i tá»«ng thÃ­ nghiá»‡m", 1)
    cols = ["experiment", "method", "status", "epochs_done", "epochs_total", "best_val_metric", "best_val_epoch",
            "train_hours", "n_crashes", "n_resumes"]
    t = doc.add_table(rows=1, cols=len(cols))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(cols):
        t.rows[0].cells[i].text = c
    for r in exp_rows:
        cells = t.add_row().cells
        for i, c in enumerate(cols):
            cells[i].text = str(r.get(c, ""))
    doc.add_paragraph("best_val_metric = mIoU trÃªn validation (nhÃ£n SAM), dÃ¹ng Ä‘á»ƒ chá»n checkpoint â€” khÃ´ng pháº£i Ä‘iá»ƒm test.")

    doc.add_heading("Káº¿t quáº£ test (80 áº£nh mask váº½ tay; má»—i mÃ´ hÃ¬nh cháº¥m 1 láº§n dÆ°á»›i khÃ³a)", 1)
    if t_rows:
        cols2 = ["lock", "test_set", "model", "n_images", "mIoU", "precision", "recall", "f1", "paper_mIoU_PIDNet_S"]
        t2 = doc.add_table(rows=1, cols=len(cols2))
        t2.style = "Light Grid Accent 1"
        for i, c in enumerate(cols2):
            t2.rows[0].cells[i].text = c
        for r in t_rows:
            cells = t2.add_row().cells
            for i, c in enumerate(cols2):
                cells[i].text = str(r.get(c, ""))
        doc.add_paragraph("paper_mIoU_PIDNet_S = sá»‘ cÃ´ng bá»‘ cá»§a WACV 2025 cho PIDNet-S (chá»‰ Ä‘á»ƒ tham chiáº¿u).")
    else:
        doc.add_paragraph("ChÆ°a cÃ³ káº¿t quáº£ test.")

    doc.add_heading("Crash / sá»± cá»‘", 1)
    any_note = False
    for r in exp_rows:
        if r["n_crashes"] or r["last_crash_or_note"]:
            any_note = True
            doc.add_paragraph(f"{r['experiment']}: {r['n_crashes']} láº§n thoÃ¡t lá»—i, {r['n_resumes']} láº§n resume. "
                              f"Ghi chÃº cuá»‘i: {r['last_crash_or_note']}", style="List Bullet")
    if not any_note:
        doc.add_paragraph("KhÃ´ng cÃ³.")
    doc.save(path)


def main():
    registry = load_json(REGISTRY, {}) or {}
    queue = load_json(REP / "queue_status.json")
    exp_rows = experiment_rows(registry, queue)
    t_rows = test_rows(registry)
    write_csv(REP / "STATUS_REPORT.csv", exp_rows)
    write_csv(REP / "STATUS_REPORT_tests.csv", t_rows)
    write_csv(REP / "ablation_results.csv", ablation_rows())
    docx_path = REP / f"STATUS_REPORT_{datetime.now():%Y-%m-%d}.docx"
    write_docx(docx_path, exp_rows, t_rows, queue)
    print(f"wrote {REP / 'STATUS_REPORT.csv'}, {REP / 'STATUS_REPORT_tests.csv'}, {docx_path}")


if __name__ == "__main__":
    main()
