import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from utils.tokenizer import CodeTokenizer
from representation.symbolic_transform import normalize_tokens  # keep if you use it in generate_gadgets
from preprocessing.generate_code_gadgets import generate_gadgets
from utils.config import CONFIG

def _ensure_parent(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def vectorize_and_save():
    # 1) Generate tokens + labels (+ paths)
    gadgets = generate_gadgets(CONFIG["code_dir"])  # must return (tokens, label, file_path)
    token_sequences = [tokens for tokens, _, _ in gadgets]
    y = np.array([label for _, label, _ in gadgets], dtype=np.int32)

    # Guard: require both classes
    u, c = np.unique(y, return_counts=True)
    dist = dict(zip(u.tolist(), c.tolist()))
    print("[Vectorize] class_dist (raw):", dist)
    assert 0 in dist and 1 in dist and dist[0] > 0 and dist[1] > 0, \
        "Dataset must contain both classes (0=Safe, 1=Vulnerable)."

    # -------- META (program_id / cwe_id) -----------
    # Put these helpers INSIDE the function so they can see CONFIG and gadgets
    def _relpath(p: str) -> str:
        try:
            return os.path.relpath(p, CONFIG["code_dir"]).replace("\\", "/")
        except Exception:
            return os.path.basename(p)

    def _program_id_from_path(p: str) -> str:
        rp = _relpath(p)
        first = rp.split("/")[0]
        return first or "UNKNOWN"

    def _cwe_id_from_path(p: str) -> int:
        low = (p or "").lower()
        if "cwe-119" in low or "cwe119" in low or "/cwe_119" in low:
            return 119
        if "cwe-399" in low or "cwe399" in low or "/cwe_399" in low:
            return 399
        return -1  # unknown/other

    program_ids = np.array([_program_id_from_path(fp) for _, _, fp in gadgets], dtype=object)
    cwe_ids     = np.array([_cwe_id_from_path(fp)     for _, _, fp in gadgets], dtype=np.int32)
    # -----------------------------------------------

    # 2) Make a stratified split to fit tokenizer ONLY on train subset (avoid leakage)
    test_size = float(CONFIG.get("test_size", 0.10))
    val_size  = float(CONFIG.get("val_size", 0.10))
    X_idx = np.arange(len(token_sequences))

    # first carve out test
    trainval_idx, test_idx, y_trainval, y_test = train_test_split(
        X_idx, y, test_size=test_size, random_state=42, stratify=y
    )
    # then carve out val from trainval
    val_frac_of_trainval = val_size / (1.0 - test_size)
    train_idx, val_idx, y_train, y_val = train_test_split(
        trainval_idx, y_trainval, test_size=val_frac_of_trainval,
        random_state=42, stratify=y_trainval
    )

    # 3) Fit tokenizer on TRAIN ONLY, then transform ALL
    tok = CodeTokenizer()
    tok.fit([token_sequences[i] for i in train_idx])

    # Transform every sequence with the train-fitted tokenizer
    X = tok.transform(token_sequences, CONFIG["max_seq_length"])

    # 4) Save artifacts (NOW include meta)
    _ensure_parent(CONFIG["vector_cache"])
    _ensure_parent(CONFIG["tokenizer_path"])

    meta = {"program_id": program_ids, "cwe_id": cwe_ids}
    with open(CONFIG["vector_cache"], "wb") as f:
        pickle.dump((X, y, meta), f)  # <-- changed to 3-tuple
    tok.save(CONFIG["tokenizer_path"])
    print("Saved vectorized inputs, meta, and tokenizer to disk.")

    # (Optional) DO NOT overwrite the program-level split cache that training will write.
    # If you want to keep this preview split, write it to a different file:
    preview_split = os.path.join(os.path.dirname(CONFIG["vector_cache"]), "split_cache_preview.pkl")
    _ensure_parent(preview_split)
    with open(preview_split, "wb") as f:
        pickle.dump((test_idx, y_test), f)
    print(f"[Vectorize] Preview test indices saved to {preview_split} (not used by training).")

if __name__ == "__main__":
    vectorize_and_save()