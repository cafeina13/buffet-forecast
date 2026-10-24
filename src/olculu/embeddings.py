"""Dish vectors from name and main ingredients.

Two backends, selected with --embeddings:
  st:    sentence-transformers, paraphrase-multilingual-MiniLM-L12-v2 (needs the optional extra)
  tfidf: character n-gram TF-IDF (offline fallback)
"""

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def dish_text(dish: dict) -> str:
    return f"{dish['name']}: {', '.join(dish['ingredients'])}"


def embed(dishes: list[dict], backend: str = "tfidf") -> np.ndarray:
    texts = [dish_text(d) for d in dishes]
    if backend == "st":
        from sentence_transformers import SentenceTransformer

        vecs = SentenceTransformer(ST_MODEL).encode(texts, normalize_embeddings=True)
        return np.asarray(vecs)
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
    return np.argsort(-sim, axis=1)[:, :k]


def project_2d(vecs: np.ndarray, seed: int = 0) -> np.ndarray:
    return PCA(2, random_state=seed).fit_transform(vecs)
