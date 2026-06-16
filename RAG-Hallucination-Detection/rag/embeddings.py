"""
rag/embeddings.py
-----------------
Thin wrapper around SentenceTransformer that provides:
  - encode_documents()  →  embed a list of text chunks
  - encode_query()      →  embed a single query string

The model is loaded once (singleton pattern) to avoid
reloading the ~90 MB weights on every call.
"""

import hashlib
import numpy as np
from config import EMBEDDING_DIM, EMBEDDING_MODEL
from utils.logger import get_logger
from utils.helpers import timeit

_HAS_SENTENCE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _HAS_SENTENCE = False

logger = get_logger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────────
_model_instance = None


def _get_model() -> "SentenceTransformer | None":
    global _model_instance
    if _model_instance is None and _HAS_SENTENCE:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model_instance


class EmbeddingEngine:
    """Generate sentence embeddings for documents and queries."""

    def __init__(self):
        # If sentence-transformers is available we use it, otherwise fall back
        # to a lightweight deterministic CPU-only embedding generator.
        self._model = _get_model()

    @timeit
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of document chunks.

        Returns
        -------
        np.ndarray
            Shape (n_chunks, embedding_dim), dtype float32.
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")

        logger.info(f"Encoding {len(texts)} document chunks…")
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype("float32")

        # Fallback deterministic embedding: seed a RNG from text hash
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype="float32")
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(EMBEDDING_DIM, dtype="float32")
            v /= (np.linalg.norm(v) + 1e-12)
            out[i] = v
        return out

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns
        -------
        np.ndarray
            Shape (1, embedding_dim), dtype float32.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if self._model is not None:
            embedding = self._model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.astype("float32")

        seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBEDDING_DIM, dtype="float32")
        v /= (np.linalg.norm(v) + 1e-12)
        return v.reshape(1, -1)
