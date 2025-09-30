"""Create normalized CWE prototype vectors from textual descriptions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils_io import ARTIFACTS_DIR, ensure_dir, l2_normalize, save_numpy, write_json

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_texts(path: Path) -> List[Tuple[str, str]]:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = [(str(entry["cwe"]), entry.get("text", "")) for entry in payload]
    else:
        raise ValueError("cwe_texts.json must be a dict or list of objects")
    cleaned = []
    for cwe, text in items:
        if text is None:
            text = ""
        cleaned.append((str(cwe), str(text)))
    cleaned.sort(key=lambda x: x[0])
    return cleaned


def embed_sbert(texts: Sequence[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "sentence-transformers is required unless --use_tfidf is set"
        ) from exc
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_tfidf(texts: Sequence[str], max_features: Optional[int] = None) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix.astype(np.float32).toarray()


def nearest_neighbor_report(cwes: Sequence[str], prototypes: np.ndarray, top_k: int = 3) -> List[str]:
    sims = prototypes @ prototypes.T
    lines: List[str] = []
    for row, cwe in enumerate(cwes[: min(len(cwes), 5)]):
        order = np.argsort(-sims[row])
        neighbors = [cwes[idx] for idx in order if idx != row][:top_k]
        joined = ", ".join(neighbors) if neighbors else "<none>"
        lines.append(f"{cwe}: {joined}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwe-texts", type=Path, default=Path("data/cwe_texts.json"))
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--use-tfidf", action="store_true", help="fallback to TF-IDF embeddings")
    parser.add_argument("--tfidf-max-features", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_texts(args.cwe_texts)
    if not pairs:
        raise SystemExit("No CWE descriptions found")
    cwes, texts = zip(*pairs)
    if args.use_tfidf:
        print("Embedding descriptions with TF-IDF ...")
        matrix = embed_tfidf(texts, args.tfidf_max_features)
    else:
        print(f"Embedding descriptions with SBERT ({args.model}) ...")
        matrix = embed_sbert(texts, args.model)
    matrix = l2_normalize(np.asarray(matrix, dtype=np.float32), axis=1)

    artifacts = Path(args.artifacts_dir)
    ensure_dir(artifacts)
    save_numpy(artifacts / "prototypes.npy", matrix)
    mapping: Dict[str, int] = {cwe: idx for idx, cwe in enumerate(cwes)}
    write_json(mapping, artifacts / "cwe_to_row.json")

    print(f"Saved prototypes with shape {matrix.shape} to {artifacts / 'prototypes.npy'}")
    print(f"Stored CWE to row mapping at {artifacts / 'cwe_to_row.json'}")

    print("Nearest-neighbor sanity check (top-3):")
    for line in nearest_neighbor_report(cwes, matrix):
        print("  ", line)


if __name__ == "__main__":
    main()