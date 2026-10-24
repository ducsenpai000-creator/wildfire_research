"""Shared helpers for reproduction / research runs.

The upstream repository is imported read-only from third_party/. Nothing in it is modified.
"""
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party" / "wildfire-real-time-segmentation"
GIT = ROOT / "tools" / "mingit" / "cmd" / "git.exe"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ALLOWED_TRAINING_SPLITS = ("train", "valid")


def repo_args(**overrides):
    """Namespace with the upstream utils.get_args defaults (parsed with an empty argv)."""
    from utils.get_args import get_args
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        args = get_args()
    finally:
        sys.argv = saved
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def git_commit(path):
    try:
        out = subprocess.run([str(GIT), "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run([str(GIT), "-C", str(path), "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def assert_not_test(mode):
    """Hard guard used by every training/validation entry point (test-set lock)."""
    if mode not in ALLOWED_TRAINING_SPLITS:
        raise PermissionError(f"split '{mode}' is not allowed outside the locked final test evaluator")
