"""Probe: can the official FoSp code (vendored mmseg 0.27 + mmcv 1.7.2 lite) build the model from its config
in the comparison environment? Builds on CPU only (no dataset access)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
FOSP = Path(__file__).resolve().parents[2] / "third_party" / "FoSp"
sys.path.insert(0, str(FOSP))
sys.path.insert(0, str(FOSP / "inpainter_setting"))  # inpainter.py imports `torch_utils` as top-level

import mmcv  # noqa: E402
import mmseg  # noqa: E402
from mmcv import Config  # noqa: E402

print("mmcv", mmcv.__version__, "| mmseg (vendored)", mmseg.__version__)
cfg = Config.fromfile(str(FOSP / "config" / "fosp_config.py"))
cfg.model.pretrained = None  # do not download during the probe
from mmseg.models import build_segmentor  # noqa: E402

try:
    model = build_segmentor(cfg.model)
    n = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"built {type(model).__name__}: params total {n/1e6:.2f}M, trainable {n_train/1e6:.2f}M")
except Exception as e:  # report, do not hide
    import traceback
    traceback.print_exc()
    print("BUILD FAILED:", repr(e))
