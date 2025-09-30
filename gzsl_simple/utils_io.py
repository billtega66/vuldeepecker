"""Utility helpers for the simplified two-CWE GZSL baseline."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ARTIFACTS_DIR = Path("artifacts")
EPS = 1e-12


@dataclass
class SampleBundle:
    """Container holding gadget features and annotations."""

    features: np.ndarray
    labels: np.ndarray
    cwe_ids: np.ndarray
    program_ids: np.ndarray

    def subset(self, indices: Sequence[int]) -> "SampleBundle":
        idx = np.asarray(indices, dtype=int)
        return SampleBundle(
            features=self.features[idx],
            labels=self.labels[idx],
            cwe_ids=self.cwe_ids[idx],
            program_ids=self.program_ids[idx],
        )

    def to_payload(self, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "features": self.features,
            "labels": self.labels,
            "cwe_ids": self.cwe_ids,
            "program_ids": self.program_ids,
        }
        if meta:
            payload["meta"] = meta
        return payload

    @property
    def positives(self) -> np.ndarray:
        return np.asarray(self.labels == 1)


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _as_array(data: Any, *, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(data)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, found {arr.ndim}")
    return arr


def load_vector_cache(path: Path) -> SampleBundle:
    """Load the cached VulDeePecker gadget features."""

    with Path(path).open("rb") as f:
        payload = pickle.load(f)

    features: Any
    labels: Any
    cwe_ids: Any = None
    program_ids: Any = None
    meta: Dict[str, Any] = {}

    if isinstance(payload, tuple):
        if len(payload) == 3:
            features, labels, meta = payload
        elif len(payload) == 2:
            features, labels = payload
        else:
            raise ValueError("Expected a tuple of (features, labels[, meta])")
    elif isinstance(payload, dict):
        features = payload.get("features")
        labels = payload.get("labels") or payload.get("y")
        meta = payload.get("meta", {})
        cwe_ids = payload.get("cwe_ids")
        if cwe_ids is None:
            cwe_ids = payload.get("cwe_id")
        program_ids = payload.get("program_ids")
        if program_ids is None:
            program_ids = payload.get("program_id")
    else:
        raise TypeError("Unsupported vector cache format")

    if cwe_ids is None and isinstance(meta, dict):
        for key in ("cwe_id", "cwe_ids"):
            if key in meta:
                cwe_ids = meta[key]
                break
    if program_ids is None and isinstance(meta, dict):
        for key in ("program_id", "program_ids"):
            if key in meta:
                program_ids = meta[key]
                break

    features_arr = _as_array(features, name="features")
    if features_arr.ndim != 2:
        raise ValueError("Feature matrix must be 2-D")
    features_arr = features_arr.astype(np.float32, copy=False)

    labels_arr = _as_array(labels, name="labels").astype(np.int64, copy=False).reshape(-1)
    if labels_arr.shape[0] != features_arr.shape[0]:
        raise ValueError("Feature and label counts do not match")

    if cwe_ids is None:
        raise ValueError("CWE identifiers are required in the vector cache meta")
    cwe_arr = _as_array(cwe_ids, name="cwe_ids").reshape(-1)
    if cwe_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError("CWE identifiers must align with labels")

    if program_ids is None:
        program_ids = np.arange(labels_arr.shape[0])
    prog_arr = _as_array(program_ids, name="program_ids").reshape(-1)
    if prog_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError("Program identifiers must align with labels")

    return SampleBundle(features_arr, labels_arr, cwe_arr.astype(str), prog_arr.astype(str))


def filter_by_cwe(bundle: SampleBundle, cwe_id: str) -> SampleBundle:
    mask = bundle.cwe_ids.astype(str) == str(cwe_id)
    return SampleBundle(
        features=bundle.features[mask],
        labels=bundle.labels[mask],
        cwe_ids=bundle.cwe_ids[mask],
        program_ids=bundle.program_ids[mask],
    )


def stratified_three_way(
    labels: np.ndarray,
    *,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be non-negative and sum to < 1")

    indices = np.arange(labels.shape[0])
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in (0, 1):
        label_idx = indices[labels == label]
        if label_idx.size == 0:
            continue
        shuffled = label_idx.copy()
        rng.shuffle(shuffled)
        n_val = int(round(shuffled.size * val_frac))
        n_test = int(round(shuffled.size * test_frac))

        if shuffled.size >= 2:
            if val_frac > 0 and n_val == 0:
                n_val = 1
            if test_frac > 0 and n_test == 0 and shuffled.size - n_val > 0:
                n_test = 1
        if n_val + n_test > shuffled.size:
            n_val = min(n_val, shuffled.size)
            n_test = min(n_test, shuffled.size - n_val)

        val_parts.append(shuffled[:n_val])
        test_parts.append(shuffled[n_val : n_val + n_test])
        train_parts.append(shuffled[n_val + n_test :])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    val_idx = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def save_split(path: Path, bundle: SampleBundle, *, meta: Optional[Dict[str, Any]] = None) -> None:
    ensure_dir(Path(path).parent)
    np.save(path, bundle.to_payload(meta=meta), allow_pickle=True)


def load_split(path: Path) -> Tuple[SampleBundle, Dict[str, Any]]:
    data = np.load(Path(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        payload = data.item()
    elif isinstance(data, dict):
        payload = data
    else:
        raise ValueError(f"Unsupported split format in {path}")

    features = _as_array(payload["features"], name="features")
    labels = _as_array(payload["labels"], name="labels").astype(np.int64, copy=False).reshape(-1)
    cwe_data = payload.get("cwe_ids")
    if cwe_data is None:
        cwe_data = payload.get("cwe_id")
    if cwe_data is None:
        raise ValueError("split payload missing CWE identifiers")
    cwe_ids = _as_array(cwe_data, name="cwe_ids").reshape(-1)
    prog_data = payload.get("program_ids")
    if prog_data is None:
        prog_data = payload.get("program_id")
    if prog_data is None:
        prog_data = np.arange(labels.shape[0])
    program_ids = _as_array(prog_data, name="program_ids").reshape(-1)

    bundle = SampleBundle(features.astype(np.float32, copy=False), labels, cwe_ids.astype(str), program_ids.astype(str))
    meta = dict(payload.get("meta", {}))
    return bundle, meta


def make_one_hot_targets(bundle: SampleBundle, class_order: Sequence[str]) -> np.ndarray:
    targets = np.zeros((bundle.labels.shape[0], len(class_order)), dtype=np.float32)
    class_lookup = {str(cwe): idx for idx, cwe in enumerate(class_order)}
    for idx, (label, cwe) in enumerate(zip(bundle.labels, bundle.cwe_ids, strict=False)):
        if label != 1:
            continue
        key = str(cwe)
        if key not in class_lookup:
            continue
        targets[idx, class_lookup[key]] = 1.0
    return targets


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return x / norms


def cosine_scores(predictions: np.ndarray, prototypes: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    preds = normalize_rows(predictions)
    protos = normalize_rows(prototypes)
    return preds @ protos.T / float(temperature)


def best_threshold(scores: np.ndarray, labels: np.ndarray, metric: str) -> Tuple[float, float]:
    metric = metric.lower()
    unique = np.unique(scores)
    thresholds = unique if unique.size <= 400 else np.linspace(scores.min(), scores.max(), num=401)

    best_tau = float(thresholds[0]) if thresholds.size else 0.0
    best_score = -np.inf
    for tau in thresholds:
        preds = scores >= tau
        if metric == "f1":
            tp = float(np.logical_and(preds, labels == 1).sum())
            fp = float(np.logical_and(preds, labels == 0).sum())
            fn = float(np.logical_and(~preds, labels == 1).sum())
            score = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        elif metric in {"accuracy", "acc"}:
            score = float((preds == (labels == 1)).mean())
        elif metric in {"recall", "tpr"}:
            positives = max(int((labels == 1).sum()), 1)
            score = float(np.logical_and(preds, labels == 1).sum() / positives)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if score > best_score:
            best_tau = float(tau)
            best_score = float(score)
    return best_tau, best_score


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = scores >= threshold
    metrics: Dict[str, Any] = {"threshold": float(threshold)}
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["aupr"] = float(average_precision_score(labels, scores))

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fp / max(tn + fp, 1)),
            "recall_unseen": float(tp / max(tp + fn, 1)),
        }
    )
    return metrics


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj





def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(serialize(data), f, indent=2, sort_keys=True)