"""
rag/pdf_loader.py
-----------------
Responsible for reading a PDF (supplied as raw bytes or a file path)
and returning cleaned plain text.

Uses PyMuPDF (fitz) — fast, reliable, no Java dependency.
"""

import fitz  # PyMuPDF
from utils.logger import get_logger
from utils.helpers import clean_text, timeit

logger = get_logger(__name__)


class PDFLoader:
    """Load and extract text from a PDF document."""

    @timeit
    def load_from_bytes(self, pdf_bytes: bytes, filename: str = "upload") -> str:
        """
        Extract all text from PDF bytes (e.g. from Streamlit file_uploader).

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF file content.
        filename : str
            Used only for logging.

        Returns
        -------
        str
            Cleaned, concatenated text from all pages.

        Raises
        ------
        ValueError
            If the PDF contains no extractable text.
        """
        logger.info(f"Loading PDF: '{filename}'")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF '{filename}': {e}")
            raise ValueError(f"Could not open PDF file: {e}") from e

        pages_text = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages_text.append(text)
                logger.debug(f"  Page {page_num}: extracted {len(text)} chars")
            else:
                logger.warning(f"  Page {page_num}: no text found (may be scanned image)")

        doc.close()

        if not pages_text:
            raise ValueError(
                "No extractable text found in this PDF. "
                "The document may be a scanned image — OCR is not supported."
            )

        raw_text = "\n\n".join(pages_text)
        cleaned  = clean_text(raw_text)

        logger.info(
            f"PDF loaded: {len(pages_text)} pages, "
            f"{len(raw_text)} raw chars → {len(cleaned)} cleaned chars"
        )
        return cleaned

    @timeit
    def load_from_path(self, path: str) -> str:
        """
        Extract text from a PDF file on disk.

        Parameters
        ----------
        path : str
            Absolute or relative path to a .pdf file.
        """
        logger.info(f"Loading PDF from path: {path}")
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        return self.load_from_bytes(pdf_bytes, filename=path)
