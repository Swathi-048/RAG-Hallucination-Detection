"""
utils/helpers.py
----------------
Shared utility functions used across the project.
"""

import re
import time
import functools
from typing import Callable, Any
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw extracted text:
      - Collapse multiple blank lines → single newline
      - Remove non-printable characters
      - Strip leading/trailing whitespace
    """
    # Remove non-printable chars (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse repeated spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """Safely truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + suffix


# ── Timing Decorator ──────────────────────────────────────────────────────────

def timeit(func: Callable) -> Callable:
    """Decorator: logs execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__qualname__} completed in {elapsed:.3f}s")
        return result
    return wrapper


# ── Validation Helpers ────────────────────────────────────────────────────────

def validate_api_key(key: str) -> bool:
    """Basic sanity-check that the API key is not empty / placeholder."""
    return bool(key) and key != "your_gemini_api_key_here"


def validate_question(question: str, min_len: int = 5) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    A question is valid if it is non-empty and long enough to be meaningful.
    """
    q = question.strip()
    if not q:
        return False, "Question cannot be empty."
    if len(q) < min_len:
        return False, f"Question is too short (minimum {min_len} characters)."
    return True, ""


# ── Trust Score Helpers ───────────────────────────────────────────────────────

def score_to_color(score: int) -> str:
    """Map a 0-100 trust score to a hex colour for UI display."""
    if score >= 70:
        return "#4ade80"   # green
    if score >= 40:
        return "#fbbf24"   # amber
    return "#f87171"       # red


def score_to_label(score: int) -> str:
    """Return a human-readable label for a trust score."""
    if score >= 70:
        return "High Trust"
    if score >= 40:
        return "Medium Trust"
    return "Low Trust – Possible Hallucination"
