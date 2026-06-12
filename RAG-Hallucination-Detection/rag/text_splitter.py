"""
rag/text_splitter.py
--------------------
Splits a long document string into overlapping fixed-size character chunks.

Why overlapping?
  The answer to a question may straddle two adjacent chunks.
  Overlap ensures no critical sentence is split at a boundary and lost.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN
from utils.logger import get_logger
from utils.helpers import timeit

logger = get_logger(__name__)


class TextSplitter:
    """
    Sliding-window character-level text splitter.

    Parameters
    ----------
    chunk_size : int
        Maximum characters per chunk (default from config).
    chunk_overlap : int
        Characters shared between consecutive chunks (default from config).
    min_chunk_len : int
        Chunks shorter than this are discarded (default from config).
    """

    def __init__(
        self,
        chunk_size: int   = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_len: int = MIN_CHUNK_LEN,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    @timeit
    def split(self, text: str) -> list[str]:
        """
        Split *text* into overlapping chunks.

        Returns
        -------
        list[str]
            Non-empty chunks, each at most `chunk_size` characters.
        """
        if not text:
            logger.warning("split() called with empty text — returning []")
            return []

        chunks: list[str] = []
        step   = self.chunk_size - self.chunk_overlap
        start  = 0

        while start < len(text):
            end   = start + self.chunk_size
            chunk = text[start:end].strip()

            if len(chunk) >= self.min_chunk_len:
                chunks.append(chunk)

            start += step

        logger.info(
            f"Text split: {len(text)} chars → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
