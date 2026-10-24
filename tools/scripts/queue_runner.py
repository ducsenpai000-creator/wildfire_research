"""Unattended experiment queue (runs detached, BelowNormal priority).

For each job in configs/queue.json (in order):
  * skip if already done (training: experiments/<exp>/summary.json; eval: reports/test_results/<lock_id>.json)
  * training: run; on crash retry with --resume (PIDNet/HF) or plain re-run (FoSp auto_resume), max 3 retries;
    after 3 failures mark FAILED, record the error tail, continue with the next job
  * eval: create the test lock (pre-registered checkpoint rule) and run the locked evaluator once
After every event: reports/queue_status.json, status report (CSV + DOCX), git commit.
Pause: create reports/QUEUE_PAUSE (or run research_extension/repro/pause_queue.ps1); the runner stops launching or
retrying and exits; resume_queue.ps1 removes the flag and restarts this runner.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
STATUS = REP / "queue_status.json"
PAUSE = REP / "QUEUE_PAUSE"
QUEUE = ROOT / "configs" / "queue.json"
GIT = ROOT / "tools" / "mingit" / "cmd" / "git.exe"
PY = {"wfseg": ROOT / "envs" / "wfseg_py38" / "Scripts" / "python.exe",
      "cmp": ROOT / "envs" / "cmp_py38" / "Scripts" / "python.exe",
      "docs": ROOT / "envs" / "docs" / "Scripts" / "python.exe"}
BELOW_NORMAL = 0x00004000
MAX_RETRIES = 3


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_status():
    try:
        return json.loads(STATUS.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"events": [], "jobs": {}}


def save_status(st):
    st["updated"] = now()
    STATUS.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")


def event(st, msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    st.setdefault("events", []).append(line)
    save_status(st)


def report_and_commit(msg):
    env = dict(os.environ)
    subprocess.run([str(PY["docs"]), "-I", str(ROOT / "tools" / "scripts" / "status_report.py")],
                   cwd=ROOT, env=env, creationflags=BELOW_NORMAL)
    subprocess.run([str(GIT), "add", "-A"], cwd=ROOT, capture_output=True)
    subprocess.run([str(GIT), "commit", "-q", "-m", f"[queue] {msg}"], cwd=ROOT, capture_output=True)


def run(cmd, log_path, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / "tools" / "hf_cache"),
               HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.update(env_extra or {})
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {now()} START {' '.join(map(str, cmd))}\n")
        log.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=BELOW_NORMAL)
        code = p.wait()
        log.write(f"\n===== {now()} EXIT {code}\n")
    return code


def tail(path, n=25):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def best_val(exp, key="val_iou"):
    import csv
    try:
        with open(ROOT / "experiments" / exp / "epochs.csv", newline="", encoding="utf-8-sig") as f:
            vals = [float(r[key]) for r in csv.DictReader(f) if r.get(key)]
        return max(vals) if vals else None
    except Exception:
        return None


def resolve_select(st, job):
    """Pre-registered selection by validation only: for each placeholder, pick the candidate experiment with the
    highest best val batch-mIoU and substitute its value into the command."""
    cmd = list(job["cmd"])
    for key, cands in job.get("select", {}).items():
        scored = {e: best_val(e) for e in cands}
        avail = {e: v for e, v in scored.items() if v is not None}
        if not avail:
            raise RuntimeError(f"no finished candidate for {key}: {scored}")
        win = max(avail, key=avail.get)
        if cands[win] == "FROM_META":   # rebuild the winning run's BRRC arguments from its recorded argv
            argv = json.loads((ROOT / "experiments" / win / "run_meta.json").read_text(encoding="utf-8"))["argv"]
            cands = dict(cands)
            cands[win] = " ".join(f"--{k.replace('_', '-')} {argv[k]}" for k in
                                  ("br_rho", "br_rmin", "br_rmax", "br_wmin", "rc_lambda", "rc_alpha", "rc_beta"))
        event(st, f"{job['id']}: select {key} -> {win} ({avail[win]:.4f}); candidates {scored}")
        new = []
        for tok in cmd:
            if tok == "{" + key + "}":
                new += cands[win].split()
            else:
                new.append(tok.replace("{" + key + "}", cands[win]))
        cmd = new
    return cmd


def train_done(job):
    return (ROOT / "experiments" / job["exp"] / "summary.json").exists()


def fosp_finalize(job):
    """Official mmseg runner writes no summary; derive one from its json log (best mf_beta on validation)."""
    d = ROOT / "experiments" / job["exp"]
    vals = []
    for lj in sorted(d.glob("*.log.json")):
        for line in lj.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("mode") == "val":
                vals.append(r)
    best = max(vals, key=lambda r: r.get("mf_beta", -1)) if vals else {}
    ckpts = sorted(d.glob("best_mf_beta_iter_*.pth"))
    (d / "summary.json").write_text(json.dumps({"best_val": best, "best_checkpoint": str(ckpts[-1]) if ckpts else None,
                                                "finished": now()}, indent=1), encoding="utf-8")


def do_train(st, job):
    exp = job["exp"]
    d = ROOT / "experiments" / exp
    d.mkdir(parents=True, exist_ok=True)
    log = d / "queue_train.log"
    js = st["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
    while True:
        if PAUSE.exists():
            js["status"] = "paused"
            event(st, f"{job['id']}: paused by QUEUE_PAUSE flag")
            return "paused"
        # Always resume PIDNet/HF jobs: the trainers start from scratch when no <exp>_state.pt exists, and otherwise
        # continue from the last finished epoch (after a crash OR a pause). FoSp resumes via auto_resume in its config.
        resume = job["kind"] != "fosp"
        try:
            base_cmd = resolve_select(st, job) if job.get("select") else job["cmd"]
        except RuntimeError as e:
            js["status"] = "FAILED"
            event(st, f"{job['id']}: selection failed ({e})")
            return "failed"
        cmd = [PY[job["env"]], *base_cmd] + (["--resume"] if resume and "--resume" not in base_cmd else [])
        js["status"] = "running"
        st["current_exp"] = exp
        event(st, f"{job['id']}: start attempt {js['attempts']} ({'resume' if resume else 'fresh/auto-resume'})")
        with open(d / "attempts.log", "a", encoding="utf-8") as al:
            al.write(f"[{now()}] queue attempt {js['attempts']} start args: {' '.join(map(str, cmd))} log={log}\n")
        code = run(cmd, log)
        with open(d / "attempts.log", "a", encoding="utf-8") as al:
            al.write(f"[{now()}] queue attempt {js['attempts']} exit code {code}\n")
        st["current_exp"] = None
        if code == 0:
            if job["kind"] == "fosp":
                fosp_finalize(job)
            js["status"] = "trained"
            event(st, f"{job['id']}: training finished OK")
            report_and_commit(f"{job['id']} trained")
            return "ok"
        if PAUSE.exists():
            js["status"] = "paused"
            event(st, f"{job['id']}: stopped (exit {code}) while QUEUE_PAUSE set -> paused")
            return "paused"
        if code == 1073807364:  # 0x40010004: process terminated by a host shutdown/restart -> not a training failure
            js.setdefault("host_shutdown_exits", []).append({"time": now(), "exit": code})
            event(st, f"{job['id']}: stopped by host shutdown/restart (exit 0x40010004); not counted as a crash")
            js["status"] = "paused"
            return "paused"
        js["attempts"] += 1
        js.setdefault("errors", []).append({"time": now(), "exit": code, "tail": tail(log)})
        event(st, f"{job['id']}: CRASH exit {code} (attempt {js['attempts']}/{MAX_RETRIES}); log tail saved")
        report_and_commit(f"{job['id']} crash {js['attempts']}")
        if js["attempts"] > MAX_RETRIES:
            js["status"] = "FAILED"
            event(st, f"{job['id']}: FAILED after {MAX_RETRIES} retries -> skipping to next job")
            return "failed"
        time.sleep(60)


def do_wait(st, job):
    """Wait (polling, no GPU) until a file produced outside the queue exists, e.g. a running training's summary."""
    js = st["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
    target = ROOT / job["file"]
    deadline = time.time() + job.get("timeout_hours", 12) * 3600
    event(st, f"{job['id']}: waiting for {job['file']}")
    while not target.exists():
        if PAUSE.exists():
            return "paused"
        if time.time() > deadline:
            js["status"] = "FAILED"
            event(st, f"{job['id']}: timeout waiting for {job['file']}")
            return "failed"
        time.sleep(300)
    js["status"] = "done"
    event(st, f"{job['id']}: {job['file']} present")
    return "ok"


def resolve_globs(args):
    """'ckpt=GLOB:<pattern>' -> newest file matching <pattern> (relative to ROOT)."""
    out = []
    for a in args:
        if "GLOB:" in a:
            head, pat = a.split("GLOB:", 1)
            key, rest = (pat.split(",", 1) + [""])[:2]
            hits = sorted(ROOT.glob(key), key=lambda p: p.stat().st_mtime)
            if not hits:
                raise FileNotFoundError(pat)
            a = head + str(hits[-1].relative_to(ROOT)).replace("\\", "/") + ("," + rest if rest else "")
        out.append(a)
    return out


def do_eval(st, job):
    js = st["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
    out = REP / "test_results" / f"{job['lock_id']}.json"
    if out.exists():
        js["status"] = "done"
        return "ok"
    lock = REP / "locks" / f"{job['lock_id']}.json"
    log = REP / "locks" / f"{job['lock_id']}.log"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if not lock.exists():
        try:
            lock_args = resolve_globs(job["lock_args"])
        except FileNotFoundError as e:
            js["status"] = "FAILED"
            event(st, f"{job['id']}: checkpoint not found for lock ({e})")
            return "failed"
        code = run([PY["cmp"], ROOT / "research_extension" / "eval" / "make_lock.py", "--lock-id", job["lock_id"],
                    "--out", lock, *lock_args], log)
        if code != 0:
            js["status"] = "FAILED"
            js.setdefault("errors", []).append({"time": now(), "exit": code, "tail": tail(log)})
            event(st, f"{job['id']}: lock creation FAILED (exit {code})")
            return "failed"
        report_and_commit(f"lock {job['lock_id']} created")
    code = run([PY["cmp"], ROOT / "research_extension" / "eval" / "final_test_eval.py", "--lock", lock], log)
    if code != 0:
        js["status"] = "FAILED"
        js.setdefault("errors", []).append({"time": now(), "exit": code, "tail": tail(log)})
        event(st, f"{job['id']}: locked test evaluation FAILED (exit {code}); lock kept, not re-run automatically")
        return "failed"
    js["status"] = "done"
    event(st, f"{job['id']}: locked test evaluation done -> {out.name}")
    report_and_commit(f"{job['id']} test evaluated")
    return "ok"


def main():
    st = load_status()
    st["runner_pid"] = os.getpid()
    jobs = json.loads(QUEUE.read_text(encoding="utf-8-sig"))["jobs"]
    event(st, f"queue runner started (pid {os.getpid()}), {len(jobs)} jobs")
    seen = set()
    while True:
        # re-read the queue file at every job boundary so jobs appended later are picked up
        jobs = json.loads(QUEUE.read_text(encoding="utf-8-sig"))["jobs"]
        todo = [j for j in jobs if j["id"] not in seen]
        if not todo:
            break
        job = todo[0]
        seen.add(job["id"])
        js = st["jobs"].setdefault(job["id"], {"status": "pending", "attempts": 0})
        st["pending"] = [j["id"] for j in jobs if st["jobs"].get(j["id"], {}).get("status") not in
                         ("done", "trained", "FAILED", "skipped")]
        if js.get("status") in ("done", "FAILED", "skipped"):
            continue
        needs = job.get("needs")
        if needs and not needs.startswith("ext:") and st["jobs"].get(needs, {}).get("status") not in ("trained", "done"):
            js["status"] = "skipped"
            event(st, f"{job['id']}: skipped because prerequisite {needs} did not succeed")
            continue
        if job["kind"] == "wait":
            r = do_wait(st, job)
        elif job["kind"] == "eval":
            if needs and needs.startswith("ext:") and not (ROOT / needs[4:]).exists():
                js["status"] = "skipped"
                event(st, f"{job['id']}: skipped, prerequisite file {needs[4:]} missing")
                continue
            r = do_eval(st, job)
        else:
            if train_done(job):
                js["status"] = "trained"
                continue
            r = do_train(st, job)
        if r == "paused":
            event(st, "queue paused; runner exits")
            report_and_commit("queue paused")
            return
    st["pending"] = []
    event(st, "queue finished")
    report_and_commit("queue finished")


if __name__ == "__main__":
    main()
