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
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Reuse the module-level singleton from rag/embeddings.py if already loaded,
# otherwise instantiate here (SentenceTransformer caches internally).
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.debug(f"Loading embedding model for hallucination detection: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


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

    model = _get_model()

    # Encode and normalise
    answer_emb  = model.encode(
        [answer], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype("float32")

    chunk_embs  = model.encode(
        context_chunks, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

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
