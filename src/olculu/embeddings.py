"""Dish vectors from name and main ingredients.

Backends, selected with --embeddings:
  tfidf:    character n-gram TF-IDF (offline, no download)
  berturk:  BERTurk fine-tuned for Turkish sentence similarity (sentence-transformers; needs the optional extra)
  minilm:   multilingual MiniLM paraphrase model (sentence-transformers; needs the optional extra)
  category: the dish group alone (one-hot), the "manual category" baseline
`st` is kept as an alias of `minilm` for older commands.
"""

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ST_MODELS = {
    "berturk": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
BACKENDS = ["tfidf", "berturk", "minilm", "category", "st"]


def dish_text(dish: dict) -> str:
    return f"{dish['name']}: {', '.join(dish['ingredients'])}"


def embed(dishes: list[dict], backend: str = "tfidf") -> np.ndarray:
    texts = [dish_text(d) for d in dishes]
    backend = "minilm" if backend == "st" else backend
    if backend in ST_MODELS:
        from sentence_transformers import SentenceTransformer

        vecs = SentenceTransformer(ST_MODELS[backend], device="cpu").encode(texts, normalize_embeddings=True)
        return np.asarray(vecs)
    if backend == "category":
        groups = sorted({d["group"] for d in dishes})
        return np.array([[1.0 if d["group"] == g else 0.0 for g in groups] for d in dishes])
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(texts)
    return normalize(tfidf).toarray()


def reduce(vecs: np.ndarray, dims: int, seed: int = 0) -> np.ndarray:
    dims = min(dims, vecs.shape[0] - 1, vecs.shape[1])
    if vecs.shape[1] > 500:
        return TruncatedSVD(dims, random_state=seed).fit_transform(vecs)
    return PCA(dims, random_state=seed).fit_transform(vecs)


def nearest(vecs: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k most similar other dishes (cosine) for each dish."""
    v = normalize(vecs)
    sim = v @ v.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


def project_2d(vecs: np.ndarray, seed: int = 0) -> np.ndarray:
    return PCA(2, random_state=seed).fit_transform(vecs)
