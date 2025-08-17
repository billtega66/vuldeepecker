import os
import json
import pickle
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model
from datetime import datetime
from utils.config import CONFIG

def _predict_stream_arrays(model, X, batch_size=512):
    """Predict on an already-sliced X (arrays) without holding probs for all at once."""
    n = len(X)
    out = []
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        pb = model.predict(xb, verbose=0).ravel()
        out.append(pb)
    return np.concatenate(out, axis=0)


def _predict_stream_indices(model, X_all, y_all, test_idx, batch_size=512):
    """
    Predict using indices into X_all/y_all, streaming in small batches to avoid
    allocating a giant X_test copy.
    Returns y_true (np.array), y_prob (np.array).
    """
    y_true = []
    y_prob = []
    for i in range(0, len(test_idx), batch_size):
        idx = test_idx[i:i+batch_size]
        xb = X_all[idx]
        pb = model.predict(xb, verbose=0).ravel()
        y_true.append(y_all[idx])
        y_prob.append(pb)
    return np.concatenate(y_true, axis=0), np.concatenate(y_prob, axis=0)


def _load_split_or_stream():
    """
    Returns one of:
      ("arrays", X_test, y_test, None)
      ("indices", None, y_all, test_idx, X_all)  # we also return X_all for streaming
    Falls back to generating a stratified split if no cache exists.
    """
    split_cache = CONFIG.get("split_cache_path", None)
    vector_cache = CONFIG["vector_cache"]

    # Helper to ensure y is int 1D
    def _fix_y(y):
        y = np.asarray(y).astype(int).ravel()
        return y

    if split_cache and os.path.exists(split_cache):
        with open(split_cache, "rb") as f:
            obj = pickle.load(f)

        # New format: (X_test, y_test) arrays
        if isinstance(obj, tuple) and len(obj) == 2 and hasattr(obj[0], "shape"):
            X_test, y_test = obj
            return ("arrays", np.asarray(X_test), _fix_y(y_test), None, None)

        # Old format: either (test_idx, y_test) or just test_idx
        if isinstance(obj, tuple) and len(obj) == 2 and not hasattr(obj[0], "shape"):
            test_idx, y_test = obj
            test_idx = np.asarray(test_idx, dtype=np.int64).ravel()
            y_test = _fix_y(y_test)
            with open(vector_cache, "rb") as f2:
                loaded = pickle.load(f2)
            if isinstance(loaded, tuple) and len(loaded) == 3:
                X_all, y_all, _ = loaded
            else:
                X_all, y_all = loaded
            y_all = _fix_y(y_all)
            return ("indices", None, y_all, test_idx, np.asarray(X_all))

        if isinstance(obj, (np.ndarray, list)):
            test_idx = np.asarray(obj, dtype=np.int64).ravel()
            with open(vector_cache, "rb") as f2:
                loaded = pickle.load(f2)
            if isinstance(loaded, tuple) and len(loaded) == 3:
                X_all, y_all, _ = loaded
            else:
                X_all, y_all = loaded
            y_all = _fix_y(y_all)
            return ("indices", None, y_all, test_idx, np.asarray(X_all))

    # No cache → fall back to a small stratified split (keeps memory modest)
    from sklearn.model_selection import train_test_split
    with open(vector_cache, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        X_all, y_all, _ = loaded
    else:
        X_all, y_all = loaded
    X_all = np.asarray(X_all)
    y_all = _fix_y(y_all)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all,
        test_size=CONFIG.get("test_size", 0.10),
        random_state=42,
        stratify=y_all
    )
    return ("arrays", X_test, _fix_y(y_test), None, None)

def _paper_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr, "Precision": prec, "F1": f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN
    }


def evaluate(return_dict: bool = False):
    # --- Load vectors (+ meta if present) ---
    with open(CONFIG["vector_cache"], "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        X_all, y_all, meta = loaded
    else:
        X_all, y_all = loaded
        meta = {
            "program_id": np.zeros(len(y_all), dtype=object),
            "cwe_id": np.full(len(y_all), -1, dtype=int),
        }
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all).astype(int).ravel()
    programs = np.asarray(meta.get("program_id"))
    cwes     = np.asarray(meta.get("cwe_id"))

    # --- Use cached program-level test indices if available; else recompute by program ---
    # --- Choose test set without building giant arrays; predict in streams ---
    model_path = CONFIG["model_save_path"].replace(".pt", ".h5")
    model = load_model(model_path)
    eval_bs = int(CONFIG.get("eval_batch_size", CONFIG.get("batch_size", 256)))

    # Try to load via the streaming-aware loader you already defined
    mode, X_test_arr, y_all_or_test, test_idx, X_all_arr = _load_split_or_stream()

    # Work out cwes for the test set
    cwes_test = None
    split_cache = CONFIG.get("split_cache_path", None)
    if mode == "indices":
        # We have indices into the full arrays
        test_idx = np.asarray(test_idx)
        cwes_test = cwes[test_idx]
        # Stream predictions using indices (no giant X_test allocation)
        y_test, y_prob = _predict_stream_indices(
            model, np.asarray(X_all), y_all_or_test, test_idx, batch_size=eval_bs
        )
    else:
        # mode == "arrays": cache stored (X_test, y_test) arrays. Try to grab cwes_test if present.
        if split_cache and os.path.exists(split_cache):
            try:
                with open(split_cache, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, tuple) and len(obj) >= 3 and hasattr(obj[0], "shape"):
                    cwes_test = np.asarray(obj[2])
            except Exception:
                cwes_test = None
        if cwes_test is None:
            # Best-effort: recompute test_idx by program to align cwes; same RNG as training.
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=float(CONFIG.get("test_size", 0.20)), random_state=42)
            _, test_idx = next(gss.split(X_all, y_all, programs))
            cwes_test = cwes[test_idx]
        # Stream predictions directly on the cached test arrays
        y_test = y_all_or_test
        y_prob = _predict_stream_arrays(model, X_test_arr, batch_size=eval_bs)

    # Final labels from probs
    y_pred = (y_prob >= 0.5).astype(int)


    # --- Paper metrics (HY-ALL, BE-ALL, RM-ALL) ---
    hy = _paper_metrics(y_test, y_pred)
    print(f"HY-ALL (all):  FPR={hy['FPR']*100:.1f}%  FNR={hy['FNR']*100:.1f}%  "
          f"TPR={hy['TPR']*100:.1f}%  P={hy['Precision']*100:.1f}%  F1={hy['F1']*100:.1f}%")

    be_mask = (cwes_test == 119)
    rm_mask = (cwes_test == 399)
    be = _paper_metrics(y_test[be_mask], y_pred[be_mask]) if be_mask.any() else None
    rm = _paper_metrics(y_test[rm_mask], y_pred[rm_mask]) if rm_mask.any() else None
    if be is not None:
        print(f"BE-ALL (119): FPR={be['FPR']*100:.1f}%  FNR={be['FNR']*100:.1f}%  "
              f"TPR={be['TPR']*100:.1f}%  P={be['Precision']*100:.1f}%  F1={be['F1']*100:.1f}%")
    if rm is not None:
        print(f"RM-ALL (399): FPR={rm['FPR']*100:.1f}%  FNR={rm['FNR']*100:.1f}%  "
              f"TPR={rm['TPR']*100:.1f}%  P={rm['Precision']*100:.1f}%  F1={rm['F1']*100:.1f}%")

    # --- Your global metrics (kept) ---
    labels = [0, 1]
    target_names = ["Safe", "Vulnerable"]

    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = None
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    # --- Pretty per-CWE block (positive class metrics per CWE) ---
    cwe_names = {
        119: "Buffer errors",
        399: "Resource management errors",
    }
    lines = []
    per_cwe_stats = {}
    for cid, human in cwe_names.items():
        mask = (cwes_test == cid)
        if not np.any(mask):
            continue
        yt = y_test[mask]
        yp = y_pred[mask]
        p, r, f1_c, _ = precision_recall_fscore_support(
            yt, yp, labels=[1], zero_division=0
        )
        p, r, f1_c = float(p[0]), float(r[0]), float(f1_c[0])
        per_cwe_stats[cid] = {"precision": p, "recall": r, "f1": f1_c}
        lines.append(f"CWE-{cid} ({human})\n")
        lines.append(f"F1-score: {f1_c:.2f}\n")
        lines.append(f"Precision: {p:.2f}\n")
        lines.append(f"Recall: {r:.2f}\n")

    if per_cwe_stats:
        avg_p = np.mean([v["precision"] for v in per_cwe_stats.values()])
        avg_r = np.mean([v["recall"] for v in per_cwe_stats.values()])
        avg_f = np.mean([v["f1"] for v in per_cwe_stats.values()])
        lines.append("\n2. Average results across datasets\n")
        lines.append(f"Average Precision: {avg_p:.3f}\n")
        lines.append(f"Average Recall: {avg_r:.3f}\n")
        lines.append(f"Average F1-score: {avg_f:.3f}\n")

    # Print your formatted block
    if lines:
        print("\n" + "\n".join(lines).strip())

    # Keep the detailed global print
    print("\n=== Classification Report ===")
    for cls_name in target_names:
        row = report.get(cls_name, {})
        print(f"{cls_name:11s}  precision={row.get('precision', 0):.4f}  "
              f"recall={row.get('recall', 0):.4f}  f1={row.get('f1-score', 0):.4f}  "
              f"support={int(row.get('support', 0))}")
    avg = report.get("weighted avg", {})
    print(f"\nWeighted Avg  precision={avg.get('precision', 0):.4f}  "
          f"recall={avg.get('recall', 0):.4f}  f1={avg.get('f1-score', 0):.4f}")
    if auc is not None:
        print(f"AUC={auc:.4f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(np.array(cm))

    # --- Save metrics JSON (keep your structure, add paper + per-CWE) ---


    payload = {
        "accuracy": acc,
        "auc": auc,
        "precision_per_class": dict(zip(target_names, [float(x) for x in prec])),
        "recall_per_class": dict(zip(target_names, [float(x) for x in rec])),
        "f1_per_class": dict(zip(target_names, [float(x) for x in f1])),
        "confusion_matrix": cm,
        "report": report,
        "paper_metrics": {
            "HY-ALL": hy,
            "BE-ALL": be,
            "RM-ALL": rm
        },
        # per-CWE (positive class) as a simple dict keyed by numeric CWE
        "per_cwe": {f"CWE-{k}": v for k, v in per_cwe_stats.items()},
        "per_cwe_pretty": lines,
    }

    out_dir = os.path.dirname(model_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    
    metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    hist_path = os.path.join(runs_dir, f"eval_metrics_{run_ts}.json")
    with open(hist_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Eval] Metrics saved to {metrics_path}")
    print(f"[Eval] (also copied to {hist_path})")