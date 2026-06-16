"""
rag/vector_store.py
-------------------
In-memory FAISS vector store.

Responsibilities:
  - Build an index from document embeddings
  - Persist the original chunk texts alongside the index
  - Search by query embedding and return top-k chunks
"""

import numpy as np
from config import TOP_K, EMBEDDING_DIM
from utils.logger import get_logger
from utils.helpers import timeit

_HAS_FAISS = True
try:
    import faiss
except Exception:
    _HAS_FAISS = False

logger = get_logger(__name__)


class VectorStore:
    """
    FAISS-backed vector store for document chunks.

    Usage
    -----
    store = VectorStore()
    store.build(chunks, embeddings)
    results = store.search(query_embedding, top_k=4)
    """

    def __init__(self):
        self._index = None
        # If faiss is unavailable, we keep embeddings in memory
        self._embeddings: np.ndarray | None = None
        self._chunks: list[str]         = []

    # ── Build ─────────────────────────────────────────────────────────────────

    @timeit
    def build(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """
        Create a flat inner-product index.

        Because embeddings are L2-normalised in EmbeddingEngine,
        inner product == cosine similarity, so IndexFlatIP gives
        cosine nearest-neighbour search.

        Parameters
        ----------
        chunks : list[str]
            Original text chunks (same order as embeddings).
        embeddings : np.ndarray
            Shape (n, dim), dtype float32.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        dim = embeddings.shape[1]
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings)
            self._embeddings = None
        else:
            # keep embeddings in memory for numpy-based search
            self._index = None
            self._embeddings = embeddings.copy()
        self._chunks = list(chunks)

        logger.info(
            f"FAISS index built: {self._index.ntotal} vectors, dim={dim}"
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Return the top-k most similar chunks for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape (1, dim), dtype float32.
        top_k : int
            Number of results.

        Returns
        -------
        list[dict]
            Each dict has keys: "chunk" (str), "score" (float), "index" (int).
            Sorted by score descending.
        """
        if self._index is None or not self._chunks:
            raise RuntimeError("Vector store is empty. Call build() first.")

        k = min(top_k, len(self._chunks))
        results = []
        if _HAS_FAISS and self._index is not None:
            scores, indices = self._index.search(query_embedding, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                results.append({
                    "chunk": self._chunks[idx],
                    "score": float(score),
                    "index": int(idx),
                })
        else:
            # numpy fallback: embeddings are L2-normalised so dot==cosine
            q = query_embedding.reshape(-1)
            sims = (self._embeddings @ q)
            idxs = np.argsort(-sims)[:k]
            for idx in idxs:
                results.append({
                    "chunk": self._chunks[int(idx)],
                    "score": float(sims[int(idx)]),
                    "index": int(idx),
                })

        logger.debug(
            f"FAISS search returned {len(results)} results "
            f"(scores: {[round(r['score'],4) for r in results]})"
        )
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return (self._index is not None or self._embeddings is not None) and len(self._chunks) > 0

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
