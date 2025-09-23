# -*- coding: utf-8 -*-
import os
import pickle
import random
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from gensim.models import Word2Vec

from utils.config import CONFIG
from utils.tokenizer import CodeTokenizer


# --------------------------
# Utilities
# --------------------------
def _make_group_splits(
    X, y, groups,
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
    attempts: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust group-aware split (mặc định 80/10/10). Bảo đảm có cả 2 lớp nếu có thể.
    Nếu không tạo được, fallback sang stratified sample split.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int).ravel()
    groups = np.asarray(groups)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    n_g = len(uniq)

    # Ít group quá -> fallback
    if n_g < 3:
        print(f"[GroupSplit] Too few groups (n={n_g}); falling back to stratified sample split.")
        X_idx = np.arange(len(y))
        X_trainval_idx, X_test_idx, y_trainval, y_test = train_test_split(
            X_idx, y, test_size=test_size, random_state=seed, stratify=y
        )
        val_frac_of_trainval = val_size / (1.0 - test_size)
        X_train_idx, X_val_idx, _, _ = train_test_split(
            X_trainval_idx, y_trainval,
            test_size=val_frac_of_trainval, random_state=seed, stratify=y_trainval
        )
        return np.array(X_train_idx), np.array(X_val_idx), np.array(X_test_idx)

    n_test = max(1, int(round(test_size * n_g)))
    n_test = min(n_test, n_g - 2)
    n_val  = max(1, int(round(val_size  * n_g)))
    n_val  = min(n_val,  n_g - 1)

    def both_classes(idx):
        if idx.size == 0: return False
        yy = y[idx]
        return (yy == 0).any() and (yy == 1).any()

    for _ in range(attempts):
        perm = rng.permutation(uniq)
        test_groups = set(perm[:n_test])
        rem_groups  = perm[n_test:]
        n_val_eff = min(n_val, len(rem_groups) - 1) if len(rem_groups) > 1 else 0
        if n_val_eff <= 0:
            continue
        val_groups   = set(rem_groups[:n_val_eff])
        train_groups = set(rem_groups[n_val_eff:])

        test_idx  = np.where(np.isin(groups, list(test_groups)))[0]
        val_idx   = np.where(np.isin(groups, list(val_groups)))[0]
        train_idx = np.where(np.isin(groups, list(train_groups)))[0]

        if both_classes(train_idx) and both_classes(val_idx) and both_classes(test_idx):
            print(f"[GroupSplit] Using group split: groups n={n_g} -> "
                  f"train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
            return train_idx, val_idx, test_idx

    print("[GroupSplit] Could not make class-balanced group splits; falling back to stratified sample split.")
    X_idx = np.arange(len(y))
    X_trainval_idx, X_test_idx, y_trainval, y_test = train_test_split(
        X_idx, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac_of_trainval = val_size / (1.0 - test_size)
    X_train_idx, X_val_idx, _, _ = train_test_split(
        X_trainval_idx, y_trainval,
        test_size=val_frac_of_trainval, random_state=seed, stratify=y_trainval
    )
    return np.array(X_train_idx), np.array(X_val_idx), np.array(X_test_idx)


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------
# Word2Vec helpers
# --------------------------
def build_w2v_embedding(tokenizer, embedding_dim: int, *X_splits):
    """Fit Word2Vec using only the provided splits and build the embedding matrix.

    Parameters
    ----------
    tokenizer : CodeTokenizer
        Tokenizer providing ``index_word`` / ``word_index`` mappings.
    embedding_dim : int
        Target dimensionality for the Word2Vec vectors.
    *X_splits : np.ndarray
        One or more matrices of token ids (train first, optionally val). Only these
        splits are used for Word2Vec so test gadgets never leak into the embedding.
    """
    if not X_splits:
        raise ValueError("At least one split of token ids must be provided to build_w2v_embedding().")
    index_word = tokenizer.tokenizer.index_word  # {idx: token}, idx bắt đầu từ 1

    # Tạo sentences (bỏ PAD=0)
    sentences = []
    for split in X_splits:
        for row in np.asarray(split):
            sentence = [index_word.get(int(t), None) for t in row if int(t) != 0]
            sentence = [tok for tok in sentence if tok]
            if sentence:
                sentences.append(sentence)

    print(f"[W2V] Training Word2Vec on {len(sentences)} sequences...")
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,           # skip-gram
        negative=10,
        epochs=10,
        seed=42
    )

    vocab_size = len(tokenizer.tokenizer.word_index) + 1  # +1 cho PAD=0
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)  # row 0 = PAD

    # UNK vector = trung bình
    all_vecs = np.stack([w2v.wv[w] for w in w2v.wv.index_to_key], axis=0)
    unk_vec = all_vecs.mean(axis=0)

    # Điền vector theo index_word
    for idx, tok in index_word.items():
        if tok in w2v.wv:
            embedding_matrix[idx] = w2v.wv[tok]
        else:
            embedding_matrix[idx] = unk_vec

    print(f"[W2V] Built embedding matrix: {embedding_matrix.shape}")
    return embedding_matrix


# --------------------------
# Train
# --------------------------
def train():
    _set_seed(42)

    # GPU soft-growth
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # --------------------------
    # Load data + tokenizer
    # --------------------------
    with open(CONFIG["vector_cache"], "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 3:
        X, y, meta = loaded
    else:
        X, y = loaded
        meta = {"program_id": np.zeros(len(y), dtype=object)}  # fallback

    cwes   = np.asarray(meta.get("cwe_id", np.full(len(y), -1, dtype=int)))
    groups = np.asarray(meta.get("program_id", np.zeros(len(y), dtype=object)))

    tokenizer = CodeTokenizer()
    tokenizer.load(CONFIG["tokenizer_path"])

    # --------------------------
    # Splits
    # --------------------------
    test_size = CONFIG.get("test_size", 0.20)
    val_size  = CONFIG.get("val_size", 0.10)
    train_idx, val_idx, test_idx = _make_group_splits(
        X, y, groups, test_size=test_size, val_size=val_size, seed=42
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    def _dist(arr):
        u, c = np.unique(arr, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))
    print(f"[Split] train: {X_train.shape}, class_dist={_dist(y_train)}")
    print(f"[Split]   val: {X_val.shape}, class_dist={_dist(y_val)}")
    print(f"[Split]  test: {X_test.shape}, class_dist={_dist(y_test)}")

    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        u = np.unique(yy)
        if len(u) < 2:
            print(f"[WARN] {name} split has only class {u.tolist()}.")

    # Cache test split nếu cần
    split_cache = CONFIG.get("split_cache_path", None)
    if split_cache:
        os.makedirs(os.path.dirname(split_cache), exist_ok=True)
        with open(split_cache, "wb") as f:
            pickle.dump((X_test, y_test, cwes[test_idx]), f)
        print(f"[Split] Test split cached to {split_cache}")

    # --------------------------
    # Word2Vec Embedding
    # --------------------------
    embedding_dim = CONFIG["embedding_dim"]
    embedding_matrix = build_w2v_embedding(tokenizer, embedding_dim, X_train, X_val)
    vocab_size = embedding_matrix.shape[0]
    input_len  = X.shape[1]
    expected_len = CONFIG["max_len"]
    assert input_len == expected_len, (
        f"Vectorized sequence length {input_len} does not match CONFIG['max_len']={expected_len}."
    )

    # --------------------------
    # Model (3-layer BLSTM như bạn đang dùng)
    # --------------------------
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=CONFIG["embedding_dim"],
                  input_length=expected_len,
                  mask_zero=True,
                  weights=[embedding_matrix],
                  trainable=False), 
        Bidirectional(LSTM(CONFIG["hidden_dim"], return_sequences=True)),
        Bidirectional(LSTM(CONFIG["hidden_dim"], return_sequences=True)),
        Bidirectional(LSTM(CONFIG["hidden_dim"], return_sequences=False)),
        Dropout(0.5),
        Dense(CONFIG.get("dense_dim", 300), activation=None),  
        Dense(2, activation="softmax"),   
    ])

    # Nhãn y_* là 0/1 -> dùng sparse_categorical_crossentropy cho 2-class softmax
    model.compile(
        optimizer=Adamax(learning_rate=CONFIG["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = CONFIG["model_save_path"].replace(".pt", ".h5")

    # Class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_dict = {int(k): float(v) for k, v in zip(classes, class_weights)}
    print(f"[ClassWeights] {class_weight_dict}")

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ]

    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    print(f"Using device: {device}")

    with tf.device(device):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG["num_epochs"],
            batch_size=CONFIG["batch_size"],
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )


    model.save(checkpoint_path)
    print(f"Model trained and saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
