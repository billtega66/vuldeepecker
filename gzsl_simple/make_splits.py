"""Create train/validation/test splits for the two-CWE GZSL setup."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from . import utils_io

DEFAULT_VECTOR_CACHE = Path("data/vectorized_gadgets.pkl")
DEFAULT_OUTPUT_DIR = Path("splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-cache", type=Path, default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--train_cwe", required=True, help="CWE id used for training (e.g., 119)")
    parser.add_argument("--test_cwe", required=True, help="CWE id held out for evaluation (e.g., 399)")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of seen data reserved for validation")
    parser.add_argument(
        "--seen_test_frac", type=float, default=0.2, help="Fraction of seen data held out for seen-CWE evaluation"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def summarize(bundle: utils_io.SampleBundle) -> Dict[str, int]:
    positives = int(bundle.labels.sum())
    total = int(bundle.labels.shape[0])
    return {"total": total, "positives": positives, "negatives": total - positives}


def main() -> None:
    args = parse_args()
    train_cwe = str(args.train_cwe)
    test_cwe = str(args.test_cwe)
    if train_cwe == test_cwe:
        raise SystemExit("train_cwe and test_cwe must differ")

    bundle = utils_io.load_vector_cache(args.vector_cache)
    seen_bundle = utils_io.filter_by_cwe(bundle, train_cwe)
    unseen_bundle = utils_io.filter_by_cwe(bundle, test_cwe)

    if seen_bundle.labels.size == 0:
        raise SystemExit(f"No samples found for CWE-{train_cwe}")
    if unseen_bundle.labels.size == 0:
        raise SystemExit(f"No samples found for CWE-{test_cwe}")

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, seen_idx = utils_io.stratified_three_way(
        seen_bundle.labels, val_frac=args.val_frac, test_frac=args.seen_test_frac, rng=rng
    )

    train_split = seen_bundle.subset(train_idx)
    val_split = seen_bundle.subset(val_idx)
    seen_split = seen_bundle.subset(seen_idx)

    out_dir = Path(args.output_dir)
    utils_io.ensure_dir(out_dir)

    utils_io.save_split(out_dir / "train.npy", train_split, meta={"split": "train", "cwe": train_cwe})
    utils_io.save_split(out_dir / "val.npy", val_split, meta={"split": "val", "cwe": train_cwe})
    utils_io.save_split(out_dir / "seen.npy", seen_split, meta={"split": "seen", "cwe": train_cwe})
    utils_io.save_split(out_dir / "test.npy", unseen_bundle, meta={"split": "test", "cwe": test_cwe})

    meta = {
        "vector_cache": str(args.vector_cache),
        "output_dir": str(out_dir),
        "train_cwe": train_cwe,
        "test_cwe": test_cwe,
        "class_order": [train_cwe, test_cwe],
        "val_fraction": args.val_frac,
        "seen_test_fraction": args.seen_test_frac,
        "seed": args.seed,
        "paths": {
            "train": str(out_dir / "train.npy"),
            "val": str(out_dir / "val.npy"),
            "seen": str(out_dir / "seen.npy"),
            "test": str(out_dir / "test.npy"),
        },
        "splits": {
            "train": summarize(train_split),
            "val": summarize(val_split),
            "seen": summarize(seen_split),
            "test": summarize(unseen_bundle),
        },
    }
    utils_io.write_json(meta, out_dir / "meta.json")

    print("Split summary (positives/negatives):")
    for name in ("train", "val", "seen", "test"):
        stats = meta["splits"][name]
        print(
            f"  {name:>5}: total={stats['total']:5d} pos={stats['positives']:5d} neg={stats['negatives']:5d}"
        )


if __name__ == "__main__":
    main()