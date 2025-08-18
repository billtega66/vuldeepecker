# VulDeePecker (Rebuild) — BLSTM + Word2Vec

This README explains how to **train** the BLSTM model initialized with **Word2Vec** embeddings, following the paper-style pipeline you’re reproducing.

> TL;DR
> 1) Prepare `CONFIG`, `vector_cache.pkl`, and a saved `CodeTokenizer`.  
> 2) `python train_blstm.py`  
> 3) The model (`.h5`) and the cached test split (`.pkl`) are written to `artifacts/` (or your paths).

---

## 1) Environment

- **Python**: 3.9+  
- **Required packages**: `tensorflow`, `gensim`, `numpy`, `scikit-learn`
- **(Optional GPU)**: Make sure your TensorFlow build matches your local CUDA/cuDNN. If unsure, you can use the CPU build of TensorFlow.

Install:
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install tensorflow gensim numpy scikit-learn
```

---

## 2) Project layout (expected)

```
.
├── train_blstm.py
├── utils/
│   ├── config.py           # provides CONFIG (dict)
│   └── tokenizer.py        # CodeTokenizer class with .load() / .save()
├── artifacts/              # (recommended) for models & cached splits
└── data/                   # where you keep your vector_cache, etc.
```

---

## 3) What inputs the trainer expects

### 3.1 `CONFIG` (in `utils/config.py`)

An example you can start with:
```python
CONFIG = {
    "code_dir": os.path.join(BASE_DIR, "data"),
    "tokenizer_path": os.path.join(BASE_DIR, "utils", "tokenizer.pkl"),
    "embedding_dim": 300,
    "hidden_dim": 300,
    "num_layers": 2,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "max_seq_length": 300,
    "model_save_path": os.path.join(BASE_DIR, "model", "blstm_model.pt"),
    "vector_cache": os.path.join(BASE_DIR, "data", "vectorized_gadgets.pkl"),
    "test_size": 0.20,  # program-level 80/20 split as in the paper
    "val_size": 0.10,   # optional 10% of train for validation
    "split_cache_path": os.path.join(BASE_DIR, "model", "split_cache.pkl"),
    "labels_manifest": os.path.join(BASE_DIR, "data", "labels.json"),  # file we'll create next
"label_unknown_policy": "manifest",  # enforce ground-truth for all files
    "metrics_dir": os.path.join(BASE_DIR, "metrics_history"),
    "ignore_split_cache": True,
    "max_len": 50,
    "dropout": 0.5
}

```

> You can tweak paths and numbers, but keep `τ = 50` (max sequence length) in your data prep to mirror the paper.

### 3.2 `vector_cache.pkl`

- Either a tuple `(X, y, meta)` or `(X, y)` created by your **vectorization pipeline**.
- **X**: `np.ndarray[int32]` of shape `(N, 50)` — token IDs per gadget, **PAD=0**.
- **y**: `np.ndarray[int32]` of shape `(N,)` with labels **0 or 1**.
- **meta** *(optional but recommended)*: `dict` with keys:
  - `program_id`: array-like (len N) to group samples for **group-aware split** (avoid leakage),
  - `cwe_id`: array-like (len N) with CWE ids (for analysis/reporting).

> Important: Ensure the `CodeTokenizer` used to produce `X` is **the same** one you save at `CONFIG["tokenizer_path"]` so that token ID ↔ token string mapping is consistent.

### 3.3 `CodeTokenizer`

Your `utils/tokenizer.py` should implement `CodeTokenizer` with:
- `load(path)` to load the tokenizer state (so `tokenizer.tokenizer.word_index` and `tokenizer.tokenizer.index_word` are available),
- `save(path)` if you need to export it earlier in your pipeline.

---

## 4) How training works (summary)

`train_blstm.py` will:
1. Load `(X, y, meta)` and the `CodeTokenizer`.
2. Create **group-aware** train/val/test splits using `program_id` (fallback: stratified).
3. Train **Word2Vec (skip-gram)** on the token sequences reconstructed from `X` and the tokenizer mapping, then build the **Embedding matrix** (PAD=0; UNK ≈ mean vector).
4. Build a **2–3 layer BLSTM** stack + **1 dense (linear)** + **softmax(2)** output, with `dropout=0.5`.
5. Train with **Adamax (lr=1.0)** for **4 epochs**, `batch_size=64`, class weights balanced.
6. Save the best model (`.h5`) and cache the test split (`.pkl`).

> By default, the Embedding layer is **frozen** at first for stability. You can unfreeze it later if desired (already scaffolded in code as comments).

---

## 5) Run

From the project root (where `train_blstm.py` lives):
```bash
python train_blstm.py
```
Outputs:
- `artifacts/vuldee_blstm.h5` — the trained model (best on val loss).
- `artifacts/test_split.pkl` — `(X_test, y_test, cwe_test)` for downstream evaluation.

---

## 6) Evaluate on cached test split (example)

Create a small script or a notebook cell like this:

```python
from tensorflow.keras.models import load_model
import numpy as np, pickle
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("artifacts/vuldee_blstm.h5")
with open("artifacts/test_split.pkl", "rb") as f:
    X_test, y_test, cwe_test = pickle.load(f)

y_prob = model.predict(X_test, verbose=0)
y_pred = y_prob.argmax(axis=1)

print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\\n", confusion_matrix(y_test, y_pred))
```

> If you want **HY-ALL / BE-ALL / RM-ALL** style reporting, slice your test indices by the corresponding **function/API lists** and compute metrics per slice, then average (macro) where needed.

---

## 7) Reproducibility notes (to match the paper closer)

- **Sequence length τ = 50**. Use the same **pad/truncate rules** you defined for forward/backward slices so that the **API call sits at the expected end** of the backward gadget.
- **Optimizer**: Adamax with **learning rate ≈ 1.0**, **epochs = 4**, **batch = 64**.
- **BLSTM layers**: 2–3. (The provided script uses 3. You can set 2 if you want to mirror their “best” with fewer layers.)
- **Class weights**: enabled to balance 0/1.
- **Group-aware split**: use `program_id` to avoid near-duplicate leakage across splits.
- **Random seeds** are set (Python/NumPy/TF) for stability.

---

## 8) Troubleshooting

- **Shape mismatch** at Embedding or index errors: The tokenizer and `X` must be produced by the **same** vocabulary. Rebuild the tokenizer or the vectors to align.
- **Loss/label shape error**: This setup uses `Dense(2, softmax)` + `sparse_categorical_crossentropy`. Ensure `y` is an **integer array** of 0/1 (not one-hot).
- **Very low precision**: Double-check pad/truncate rules for backward slices and confirm your **function/API lists** for BE/RM are correct. Also try unfreezing the Embedding after 1–2 epochs.
- **OOM on GPU**: Reduce `batch_size` (e.g., 32 or 16) or switch to CPU for testing.

---

## 9) Notes

- The script saves in **Keras `.h5`** format for portability. If you prefer the native Keras format, change the extension to `.keras` and Keras will save in the newer format.
- If you maintain multiple dataset variants (e.g., `BE-ALL`, `RM-ALL`, `HY-ALL`), you can automate runs by looping over the config and swapping the function/API filter upstream of `vector_cache.pkl` creation.

---

**Good luck & happy reproducing!**
