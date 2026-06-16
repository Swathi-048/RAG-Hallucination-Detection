"""
rag/rag_pipeline.py
-------------------
Orchestrates the full RAG pipeline:
  1. Ingest a document (PDF bytes → chunks → embeddings → FAISS index)
  2. Answer a question  (query → retrieve → Gemini → answer)
  3. Re-answer with a stricter grounding prompt (correction path)

This module is the only place that talks to the Gemini API.
"""

_HAS_GENAI = True
try:
    import google.generativeai as genai
except Exception:
    genai = None
    _HAS_GENAI = False

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    TOP_K,
)
from rag.pdf_loader    import PDFLoader
from rag.text_splitter import TextSplitter
from rag.embeddings    import EmbeddingEngine
from rag.vector_store  import VectorStore
from utils.logger      import get_logger
from utils.helpers     import timeit, validate_api_key

logger = get_logger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

_ANSWER_PROMPT = """\
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say exactly:
"I could not find the answer in the provided document."

Context:
{context}

Question: {question}

Answer:"""

_CORRECTED_ANSWER_PROMPT = """\
You are a strict fact-checker. Answer ONLY using the provided context.
Do NOT use any external knowledge, assumptions, or information not found in the context.
If the answer is not in the context, say exactly:
"I could not find the answer in the provided document."

Context:
{context}

Question: {question}

Grounded Answer:"""

_OPEN_ANSWER_PROMPT = """\
You are a helpful assistant. Use the provided context to support your answer.
You may also use your general knowledge to answer the question if it is not fully covered by the document.
If the question can be answered from the document, prefer that information.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Typical usage
    -------------
    pipeline = RAGPipeline()
    pipeline.ingest(pdf_bytes, filename="policy.pdf")
    result = pipeline.answer("How many leave days do employees get?")
    """

    def __init__(self):
        # If the official Google GenAI client is available, require an API key
        # and configure the real Gemini model. Otherwise fall back to a
        # deterministic mock LLM so the app remains usable offline.
        if _HAS_GENAI:
            if not validate_api_key(GOOGLE_API_KEY):
                raise EnvironmentError(
                    "GOOGLE_API_KEY is missing or not set. "
                    "Copy .env.example → .env and add your key."
                )

            genai.configure(api_key=GOOGLE_API_KEY)
            self._llm = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS,
                ),
            )
            self._mock_llm = False
        else:
            logger.warning(
                "google.generativeai not installed — running with a mock LLM. "
                "Install the google-generativeai package and set GOOGLE_API_KEY "
                "to enable Gemini integration."
            )
            # Simple mock LLM that returns a context-aware snippet
            class _MockResponse:
                def __init__(self, text: str):
                    self.text = text

            class _MockLLM:
                def generate_content(self, prompt: str):
                    # Attempt to extract the Context: section from the prompt
                    marker = "Context:\n"
                    if marker in prompt:
                        ctx = prompt.split(marker, 1)[1]
                        # Use the first 400 chars of the context as a mocked answer
                        snippet = ctx.strip()[:400]
                        if snippet:
                            return _MockResponse(snippet + "\n\n(MOCKED LLM response — Gemini unavailable)")
                    # Default fallback
                    return _MockResponse("I could not find the answer in the provided document.")

            self._llm = _MockLLM()
            self._mock_llm = True

        self._loader    = PDFLoader()
        self._splitter  = TextSplitter()
        self._embedder  = EmbeddingEngine()
        self._store     = VectorStore()

        self.doc_name: str  = ""
        self.chunk_count: int = 0

        logger.info("RAGPipeline initialised.")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    @timeit
    def ingest(self, pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
        """
        Process a PDF and build the vector store.

        Returns
        -------
        dict
            {"filename": str, "chunks": int, "characters": int}
        """
        logger.info(f"Ingesting document: {filename}")

        text    = self._loader.load_from_bytes(pdf_bytes, filename)
        chunks  = self._splitter.split(text)

        if not chunks:
            raise ValueError("Document produced no usable text chunks.")

        embeddings = self._embedder.encode_documents(chunks)
        self._store.build(chunks, embeddings)

        self.doc_name    = filename
        self.chunk_count = len(chunks)

        logger.info(f"Ingestion complete: {len(chunks)} chunks indexed.")
        return {
            "filename":   filename,
            "chunks":     len(chunks),
            "characters": len(text),
        }

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Returns
        -------
        list[dict]
            Each dict: {"chunk": str, "score": float, "index": int}
        """
        if not self._store.is_ready:
            raise RuntimeError("No document ingested. Call ingest() first.")

        q_emb = self._embedder.encode_query(question)
        return self._store.search(q_emb, top_k=top_k)

    # ── Generation ────────────────────────────────────────────────────────────

    @timeit
    def answer(self, question: str, top_k: int = TOP_K, allow_external_knowledge: bool = False) -> dict:
        """
        Full RAG answer: retrieve → generate.

        Returns
        -------
        dict
            {
              "question":        str,
              "answer":          str,
              "retrieved_chunks": list[dict],   # [{chunk, score, index}, …]
              "context":         str,           # joined chunks sent to LLM
            }
        """
        retrieved = self.retrieve(question, top_k=top_k)
        context   = self._build_context(retrieved)
        prompt    = _OPEN_ANSWER_PROMPT if allow_external_knowledge else _ANSWER_PROMPT
        answer    = self._call_gemini(prompt, question, context)

        logger.info(f"Answer generated ({len(answer)} chars).")
        return {
            "question":        question,
            "answer":          answer,
            "retrieved_chunks": retrieved,
            "context":         context,
        }

    @timeit
    def corrected_answer(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Stricter re-generation for low-trust answers.
        Uses the grounding prompt that forbids external knowledge.
        """
        retrieved = self.retrieve(question, top_k=top_k)
        context   = self._build_context(retrieved)
        answer    = self._call_gemini(_CORRECTED_ANSWER_PROMPT, question, context)

        logger.info(f"Corrected answer generated ({len(answer)} chars).")
        return {
            "question":        question,
            "answer":          answer,
            "retrieved_chunks": retrieved,
            "context":         context,
            "corrected":       True,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context(retrieved: list[dict]) -> str:
        """Join retrieved chunks with a clear separator."""
        return "\n\n---\n\n".join(r["chunk"] for r in retrieved)

    def _call_gemini(self, prompt_template: str, question: str, context: str) -> str:
        """Format prompt and call Gemini; return stripped text."""
        prompt = prompt_template.format(context=context, question=question)
        try:
            response = self._llm.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS,
                ),
            )

            # Response object may expose `_result` in this client version.
            result = getattr(response, "_result", None)
            if result is None:
                raise RuntimeError("Gemini response missing _result payload")

            candidates = getattr(result, "candidates", None)
            if not candidates:
                raise RuntimeError("Gemini response contained no candidates")

            return candidates[0].content.parts[0].text.strip()
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    @property
    def is_ready(self) -> bool:
        """True once a document has been successfully ingested."""
        return self._store.is_ready
