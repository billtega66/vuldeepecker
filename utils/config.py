import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
