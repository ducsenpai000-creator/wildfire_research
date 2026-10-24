"""Launch the OFFICIAL FoSp training script (third_party/FoSp/tools/train.py, unmodified) on Protocol P.

Only environment set-up happens here: module paths (FoSp root + inpainter_setting, which the official code
expects on PYTHONPATH), a hard VRAM cap, and low CPU thread counts. Resume after a crash is handled by the
official runner via its `--auto-resume` flag (work_dir/latest.pth).

  python run_fosp.py --config configs/fosp_P.py --seed 42 [--max-vram-gb 5] [--smoke N]
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOSP = ROOT / "third_party" / "FoSp"


def patch_mmcv_scatter():
    """Compatibility shim (no numerical effect): mmcv < 1.7.1 passes an int GPU id to
    torch.nn.parallel._functions._get_stream, which PyTorch >= 2.0 expects as torch.device. This mirrors the
    upstream mmcv fix (mmcv 1.7.1 / 2.x: `_get_stream(torch.device('cuda', device))`)."""
    import torch
    import mmcv.parallel._functions as f
    orig = f._get_stream

    def _get_stream(device):
        return orig(torch.device("cuda", device) if isinstance(device, int) else device)
    f._get_stream = _get_stream


def patch_fosp_plugins():
    """Use the official pure-PyTorch reference path of the MAT custom ops (upfirdn2d, bias_act) directly.
    The official `_init()` sets `_inited` only on success, so on Windows (no CUDA compiler) it re-attempts the
    ninja/MSVC build on *every* call before falling back to the very same reference implementation. Marking it as
    initialised (with no plugin) keeps the identical fallback numerics and avoids the repeated build attempts."""
    import torch_utils.ops.bias_act as ba
    import torch_utils.ops.upfirdn2d as up
    for mod in (ba, up):
        mod._inited = True
        mod._plugin = None


def patch_fosp_smokeseg_metric():
    """RECONSTRUCTION of a missing piece of the released FoSp code (documented deviation).
    The released code is inconsistent: the config asks for metric 'SmokeSeg' (accepted by CustomDataset.evaluate),
    the metric helper `total_area_to_metrics` only implements 'smoke_mIoU', and `apis/train.py` selects the best
    checkpoint by a key 'mf_beta' that no released code produces. Minimal reconstruction: treat 'SmokeSeg' as the
    released 'smoke_mIoU' helper and expose its image-averaged smoke F-beta (beta=1, the code's default) as
    'mf_beta'. Nothing else in the evaluation is changed."""
    import torch
    import mmseg.core.evaluation.metrics as M
    orig = M.total_area_to_metrics

    def total_area_to_metrics(pre_eval_results, tai, tau, tapl, tal, miou_by_image, mf1_by_image,
                              metrics=["mIoU"], nan_to_num=None, beta=1):
        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        smoke = "SmokeSeg" in metrics
        metrics = ["smoke_mIoU" if m == "SmokeSeg" else m for m in metrics]
        ret = orig(pre_eval_results, tai, tau, tapl, tal, miou_by_image, mf1_by_image, metrics, nan_to_num, beta)
        if smoke:
            ret["mf_beta"] = torch.tensor([mf1_by_image, mf1_by_image], dtype=torch.float64)
        return ret
    M.total_area_to_metrics = total_area_to_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=2077)  # official tools/train.py default (paper: not stated)
    ap.add_argument("--max-vram-gb", type=float, default=5.0)
    ap.add_argument("--cpu-threads", type=int, default=2)
    ap.add_argument("--smoke", type=int, default=0, help="smoke test: run N optimizer updates into a scratch dir")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.cpu_threads))
    import torch
    torch.set_num_threads(a.cpu_threads)
    if a.max_vram_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, a.max_vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    for p in (FOSP / "inpainter_setting", FOSP):
        sys.path.insert(0, str(p))
    patch_mmcv_scatter()
    patch_fosp_plugins()
    patch_fosp_smokeseg_metric()
    argv = [str(FOSP / "tools" / "train.py"), str(Path(a.config).resolve()), "--seed", str(a.seed)]
    if a.smoke:
        cum = 3
        argv += ["--work-dir", str(ROOT / "experiments" / "_smoke" / "fosp_smoke"),
                 "--cfg-options", f"runner.max_iters={a.smoke * cum}", f"evaluation.interval={a.smoke * cum}",
                 f"checkpoint_config.interval={a.smoke * cum}", "auto_resume=False", "lr_config.warmup_iters=3"]
    else:
        # official train.py overwrites cfg.auto_resume with its --auto-resume flag (default off), so the config
        # value alone is ignored: pass the official flag to resume from work_dir/latest.pth after a pause/crash
        argv.append("--auto-resume")
    sys.argv = argv
    os.chdir(FOSP)
    runpy.run_path(str(FOSP / "tools" / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
