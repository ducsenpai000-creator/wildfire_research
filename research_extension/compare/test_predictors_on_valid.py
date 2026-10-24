"""Exercise the locked-evaluator code path for comparison methods WITHOUT the test set:
make_predictor() + evaluate_predictor() on a few VALIDATION images (SAM pseudo-labels as GT), using smoke-test
checkpoints. Only checks that the path runs and returns sane shapes/values; the numbers are meaningless.

  python test_predictors_on_valid.py kind=...,ckpt=...,[pretrained=...],[config=...] ...
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
sys.path.insert(0, str(HERE))
from evaluator_core import chdir_repo, evaluate_predictor, make_dataset  # noqa: E402
from predictors import make_predictor  # noqa: E402

ROOT = HERE.parents[1]


def main():
    chdir_repo()
    ds, _ = make_dataset(str(ROOT / "data" / "protocol_P"), "valid", manual_masks=False)
    sub_idx = [0, 1, len(ds) // 2, len(ds) - 1]

    class Sub:
        img_dir, img_list = ds.img_dir, [ds.img_list[i] for i in sub_idx]

        def __len__(self):
            return len(sub_idx)

        def __getitem__(self, i):
            return ds[sub_idx[i]]

    for spec in sys.argv[1:]:
        entry = dict(p.split("=", 1) for p in spec.split(","))
        for k in ("ckpt", "config"):
            if k in entry:
                entry["checkpoint" if k == "ckpt" else k] = str((ROOT / entry[k]).resolve())
        predict, n = make_predictor(entry, "cuda")
        res = evaluate_predictor(predict, Sub(), (0.4, 0.5))
        print(entry["kind"], "params", n, json.dumps(res["by_threshold"]["0.5"]))


if __name__ == "__main__":
    main()
