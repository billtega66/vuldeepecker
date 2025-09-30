"""Train the ridge projection that maps gadget features to CWE prototypes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge

from . import utils_io

DEFAULT_WEIGHTS = utils_io.ARTIFACTS_DIR / "ridge_W.npy"
DEFAULT_META = utils_io.ARTIFACTS_DIR / "ridge_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--weights_out", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--meta_out", type=Path, default=DEFAULT_META)
    parser.add_argument(
        "--splits_meta",
        type=Path,
        help="Path to the make_splits meta.json (defaults to alongside --train_data)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    splits_meta_path = args.splits_meta
    if splits_meta_path is None:
        candidate = args.train_data.parent / "meta.json"
        splits_meta_path = candidate if candidate.exists() else None
    split_info: Optional[Dict[str, Any]] = None
    if splits_meta_path is not None and splits_meta_path.exists():
        split_info = utils_io.read_json(splits_meta_path)  # type: ignore[attr-defined]
    elif splits_meta_path is not None:
        raise SystemExit(f"Could not find split metadata at {splits_meta_path}")

    train_bundle, _ = utils_io.load_split(args.train_data)
    val_bundle, _ = utils_io.load_split(args.val_data)

    if int(train_bundle.labels.sum()) == 0:
        raise SystemExit("Training data must contain at least one vulnerable gadget")

    if split_info is None:
        train_cwe = str(np.unique(train_bundle.cwe_ids[train_bundle.labels == 1])[0])
        unseen_candidates = np.unique(train_bundle.cwe_ids[train_bundle.labels == 1])
        if unseen_candidates.size != 1:
            raise SystemExit("Training data must contain exactly one CWE when split metadata is absent")
        test_cwe = "399" if train_cwe == "119" else "119"
        class_order = [train_cwe, test_cwe]
    else:
        train_cwe_raw = split_info.get("train_cwe")
        test_cwe_raw = split_info.get("test_cwe")
        if train_cwe_raw is None or test_cwe_raw is None:
            raise SystemExit("split metadata must include train_cwe and test_cwe")
        train_cwe = str(train_cwe_raw)
        test_cwe = str(test_cwe_raw)
        class_order = [str(cwe) for cwe in split_info.get("class_order", [train_cwe, test_cwe])]

    if train_cwe == test_cwe:
        raise SystemExit("train and test CWEs must differ")

    targets = utils_io.make_one_hot_targets(train_bundle, class_order)

    model = Ridge(alpha=args.alpha, fit_intercept=False)
    model.fit(train_bundle.features, targets)
    weights = model.coef_.T.astype(np.float32, copy=False)

    utils_io.ensure_dir(args.weights_out.parent)
    np.save(args.weights_out, weights)

    meta = {
        "alpha": args.alpha,
        "weights_path": str(args.weights_out),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "train_cwe": train_cwe,
        "test_cwe": test_cwe,
        "class_order": class_order,
        "feature_dim": int(train_bundle.features.shape[1]),
        "train_samples": int(train_bundle.labels.shape[0]),
        "val_samples": int(val_bundle.labels.shape[0]),
    }
    if split_info is not None:
        meta["split_meta"] = split_info

    utils_io.write_json(meta, args.meta_out)

    print("Ridge training complete:")
    print(f"  weights saved to {args.weights_out}")
    print(f"  meta saved to {args.meta_out}")


if __name__ == "__main__":
    main()