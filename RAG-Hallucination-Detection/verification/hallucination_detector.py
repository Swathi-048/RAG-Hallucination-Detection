"""
verification/hallucination_detector.py
---------------------------------------
Core hallucination detection logic.

Algorithm
---------
1. Embed the generated answer.
2. Embed each retrieved context chunk.
3. Compute cosine similarity between the answer and every chunk.
   (Embeddings are L2-normalised, so cosine similarity = dot product.)
4. Return per-chunk similarities plus aggregate statistics.

This module is pure computation — it does NOT decide what to do
with the scores (that's trust_score.py and response_validator.py).
"""

import numpy as np

from config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Try to use SentenceTransformer if available; otherwise fall back to the
# project's EmbeddingEngine which provides compatible embeddings.
_HAS_SENTENCE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    _HAS_SENTENCE = False

_embedder = None
if not _HAS_SENTENCE:
    from rag.embeddings import EmbeddingEngine
    _embedder = EmbeddingEngine()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D float32 vectors (already normalised)."""
    # dot product of unit vectors == cosine similarity
    return float(np.dot(a, b))


def compute_similarities(answer: str, context_chunks: list[str]) -> dict:
    """
    Compute cosine similarities between the answer and each context chunk.

    Parameters
    ----------
    answer : str
        The LLM-generated answer.
    context_chunks : list[str]
        The chunks retrieved from the document.

    Returns
    -------
    dict
        {
          "per_chunk": list[float],   # similarity for each chunk
          "max":       float,
          "mean":      float,
          "min":       float,
        }
    """
    if not answer.strip():
        logger.warning("Empty answer passed to compute_similarities.")
        return {"per_chunk": [], "max": 0.0, "mean": 0.0, "min": 0.0}

    if not context_chunks:
        logger.warning("No context chunks passed to compute_similarities.")
        return {"per_chunk": [], "max": 0.0, "mean": 0.0, "min": 0.0}

    # Encode and normalise. Use SentenceTransformer when available for
    # performance/compatibility; otherwise use the EmbeddingEngine fallback.
    if _HAS_SENTENCE:
        model = SentenceTransformer(EMBEDDING_MODEL)
        answer_emb = model.encode([answer], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
        chunk_embs = model.encode(context_chunks, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    else:
        answer_emb = _embedder.encode_query(answer)[0].astype("float32")
        chunk_embs = _embedder.encode_documents(context_chunks).astype("float32")

    per_chunk = [_cosine_similarity(answer_emb, c) for c in chunk_embs]

    result = {
        "per_chunk": [round(s, 6) for s in per_chunk],
        "max":        round(max(per_chunk),              6),
        "mean":       round(float(np.mean(per_chunk)),   6),
        "min":        round(min(per_chunk),              6),
    }

    logger.debug(
        f"Similarity stats → max={result['max']:.4f}  "
        f"mean={result['mean']:.4f}  min={result['min']:.4f}"
    )
    return result
