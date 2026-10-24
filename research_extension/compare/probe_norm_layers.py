"""Count normalisation layer types in the HF comparison models (decides whether micro-batching is exact)."""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hf import build_model  # noqa: E402

for kind, ckpt in (("segformer", "nvidia/mit-b2"), ("mask2former", "facebook/mask2former-swin-tiny-ade-semantic")):
    m = build_model(kind, ckpt)
    c = collections.Counter(type(x).__name__ for x in m.modules() if "Norm" in type(x).__name__)
    print(kind, dict(c))
