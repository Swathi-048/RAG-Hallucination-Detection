"""
verification/response_validator.py
------------------------------------
Public API for the verification subsystem.

ResponseValidator.validate() takes an answer + context chunks and returns
a complete verification result — including a corrected answer if needed.

This is the only verification file that app.py needs to import.
"""

from verification.hallucination_detector import compute_similarities
from verification.trust_score            import build_trust_report
from utils.logger                        import get_logger

logger = get_logger(__name__)


class ResponseValidator:
    """
    Ties together hallucination detection, trust scoring, and correction.

    Parameters
    ----------
    pipeline : RAGPipeline
        The active pipeline — needed to call corrected_answer() when
        the trust score is below threshold.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def validate(self, question: str, answer: str, retrieved_chunks: list[str]) -> dict:
        """
        Full validation cycle.

        Steps
        -----
        1. Compute cosine similarities (answer vs each chunk).
        2. Build a trust report.
        3. If trust < threshold → regenerate with stricter prompt.
        4. Return the complete result bundle.

        Parameters
        ----------
        question : str
        answer   : str   Initial Gemini answer.
        retrieved_chunks : list[str]

        Returns
        -------
        dict
            {
              "initial_answer":  str,
              "final_answer":    str,
              "was_corrected":   bool,
              "trust":           dict,  ← from build_trust_report()
            }
        """
        logger.info("Starting response validation…")

        # Step 1 & 2: similarity → trust
        sims   = compute_similarities(answer, retrieved_chunks)
        report = build_trust_report(sims)

        initial_answer = answer
        final_answer   = answer
        was_corrected  = False

        # Step 3: correction if needed
        if report["needs_correction"]:
            logger.info(
                f"Trust score {report['trust_score']}% is below threshold "
                f"({report['trust_score']} < threshold). Triggering correction."
            )
            corrected = self._pipeline.corrected_answer(question)
            final_answer = corrected["answer"]
            was_corrected = True

            # Re-evaluate the corrected answer
            sims2   = compute_similarities(final_answer, retrieved_chunks)
            report  = build_trust_report(sims2)

            logger.info(
                f"Post-correction trust score: {report['trust_score']}%"
            )

        return {
            "initial_answer": initial_answer,
            "final_answer":   final_answer,
            "was_corrected":  was_corrected,
            "trust":          report,
        }
