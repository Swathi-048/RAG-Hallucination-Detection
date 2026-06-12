"""
verification/trust_score.py
----------------------------
Converts raw similarity statistics into a human-readable trust report.

Trust Score Formula
-------------------
  trust = (HIGH_W × max_similarity + LOW_W × mean_similarity) × 100

Rationale:
  - max_similarity captures the BEST match between answer and any chunk.
    An answer grounded in just one chunk should still score well.
  - mean_similarity penalises answers that diverge from most context.
  - Weighting 70/30 ensures a well-grounded answer is rewarded even if
    some chunks are topically distant.

Thresholds (from config.py)
---------------------------
  ≥ 70  →  High Trust
  40-69 →  Medium Trust
  < 40  →  Low Trust  (triggers correction)
"""

from config import HIGH_TRUST_THRESHOLD, LOW_TRUST_THRESHOLD, HIGH_W, LOW_W
from utils.logger  import get_logger
from utils.helpers import score_to_color, score_to_label

logger = get_logger(__name__)


def build_trust_report(similarities: dict) -> dict:
    """
    Build a full trust report from similarity statistics.

    Parameters
    ----------
    similarities : dict
        Output of hallucination_detector.compute_similarities().

    Returns
    -------
    dict
        {
          "trust_score":      int,   (0-100)
          "label":            str,
          "color":            str,   (hex colour for UI)
          "needs_correction": bool,
          "details":          str,   (human-readable explanation)
          "max_sim":          float,
          "mean_sim":         float,
          "per_chunk":        list[float],
        }
    """
    max_sim  = similarities.get("max",  0.0)
    mean_sim = similarities.get("mean", 0.0)
    per_chunk = similarities.get("per_chunk", [])

    # Core formula
    raw_score   = HIGH_W * max_sim + LOW_W * mean_sim
    trust_score = min(100, max(0, round(raw_score * 100)))

    label             = score_to_label(trust_score)
    color             = score_to_color(trust_score)
    needs_correction  = trust_score < LOW_TRUST_THRESHOLD

    # Human-readable explanation
    if trust_score >= HIGH_TRUST_THRESHOLD:
        details = (
            f"The answer is closely aligned with the source document "
            f"(best chunk match: {max_sim:.2f}). No hallucination detected."
        )
    elif trust_score >= LOW_TRUST_THRESHOLD:
        details = (
            f"Partial overlap with the source document "
            f"(best chunk match: {max_sim:.2f}). "
            "The answer may contain minor extrapolations beyond the document."
        )
    else:
        details = (
            f"Low semantic overlap with the source document "
            f"(best chunk match: {max_sim:.2f}). "
            "The answer likely contains information not found in the document. "
            "Auto-correction has been triggered."
        )

    report = {
        "trust_score":      trust_score,
        "label":            label,
        "color":            color,
        "needs_correction": needs_correction,
        "details":          details,
        "max_sim":          round(max_sim,  4),
        "mean_sim":         round(mean_sim, 4),
        "per_chunk":        per_chunk,
    }

    logger.info(
        f"Trust report → score={trust_score}%  label='{label}'  "
        f"correction={'yes' if needs_correction else 'no'}"
    )
    return report
