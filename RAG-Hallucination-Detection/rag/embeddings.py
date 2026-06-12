"""
rag/embeddings.py
-----------------
Thin wrapper around SentenceTransformer that provides:
  - encode_documents()  →  embed a list of text chunks
  - encode_query()      →  embed a single query string

The model is loaded once (singleton pattern) to avoid
reloading the ~90 MB weights on every call.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from utils.logger import get_logger
from utils.helpers import timeit

logger = get_logger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────────
_model_instance: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model_instance


class EmbeddingEngine:
    """Generate sentence embeddings for documents and queries."""

    def __init__(self):
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
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit-length → cosine ≡ dot product
        )
        return embeddings.astype("float32")

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

        embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")
