# 🛡️ RAG Hallucination Detection & Trust Verification

> A final-year B.Tech CSE project demonstrating how Retrieval-Augmented Generation (RAG)
> can be made more reliable through semantic hallucination detection and trust-based
> response verification.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5--Flash-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-0467DF)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Example Usage & Testing](#example-usage--testing)
- [Trust Score Formula](#trust-score-formula)
- [Future Enhancements](#future-enhancements)

---

## Project Overview

Standard RAG systems retrieve relevant document chunks and feed them to an LLM to
generate answers. However, LLMs can still "hallucinate" — producing confident answers
that are not grounded in the retrieved context.

This project adds a **verification layer** on top of RAG:
1. Every generated answer is semantically compared against the retrieved context.
2. A **Trust Score (0-100%)** quantifies how grounded the answer is.
3. Answers with low trust scores are **automatically re-generated** with a stricter prompt.

---

## Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Upload any PDF; text is extracted, cleaned, and indexed |
| 🔍 RAG Q&A | Questions answered using top-4 retrieved document chunks |
| 🧠 Hallucination Detection | Cosine similarity between answer and context embeddings |
| 📊 Trust Score | 0-100% score with colour-coded status |
| 🔄 Auto-Correction | Low-trust answers re-generated with a grounding-only prompt |
| 💬 Q&A History | Full session history with per-question trust details |
| 📈 Similarity Breakdown | Per-chunk similarity visualised in the UI |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│              (PDF Upload · Question Input · Results)        │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       RAG Pipeline         │
              │                            │
              │  PDF → PyMuPDF → Text      │
              │     ↓                      │
              │  TextSplitter (500/50)     │
              │     ↓                      │
              │  SentenceTransformer       │
              │  (all-MiniLM-L6-v2)        │
              │     ↓                      │
              │  FAISS Index               │
              │     ↓  (query time)        │
              │  Top-K Retrieval           │
              │     ↓                      │
              │  Gemini 1.5 Flash → Answer │
              └─────────────┬──────────────┘
                            │
         ┌──────────────────▼──────────────────┐
         │       Verification Layer            │
         │                                     │
         │  Embed(Answer) vs Embed(Chunks)     │
         │         ↓                           │
         │  Cosine Similarity per chunk        │
         │         ↓                           │
         │  Trust Score Formula                │
         │  = (0.7×max + 0.3×mean) × 100      │
         │         ↓                           │
         │  ≥70%: ✅ High Trust → display      │
         │  40-69%: ⚠️ Medium Trust → display  │
         │  <40%: 🚨 Low Trust → Re-prompt     │
         │              ↓                      │
         │         Corrected Answer            │
         └─────────────────────────────────────┘
```

---

## Repository Structure

```
RAG-Hallucination-Detection/
│
├── app.py                          # Streamlit UI entry point
├── config.py                       # All tuneable parameters
├── requirements.txt
├── .env.example                    # API key template
├── .gitignore
├── README.md
│
├── rag/
│   ├── __init__.py
│   ├── pdf_loader.py               # PyMuPDF text extraction
│   ├── text_splitter.py            # Overlapping character chunking
│   ├── embeddings.py               # SentenceTransformer wrapper
│   ├── vector_store.py             # FAISS index (build + search)
│   └── rag_pipeline.py             # Orchestrator: ingest + answer
│
├── verification/
│   ├── __init__.py
│   ├── hallucination_detector.py   # Cosine similarity computation
│   ├── trust_score.py              # Formula + trust report builder
│   └── response_validator.py       # Public API: validate() + correct
│
├── utils/
│   ├── logger.py                   # Centralised logging
│   └── helpers.py                  # Clean text, validate, colour map
│
└── data/
    └── sample_policy.txt           # Sample document for testing
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- A free Google Gemini API key — get one at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/RAG-Hallucination-Detection.git
cd RAG-Hallucination-Detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Open .env and replace: GOOGLE_API_KEY=your_actual_key_here
```

> The app uses `gemini-2.5-flash` by default. If you want to switch models, update `GEMINI_MODEL` in `config.py`.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## How It Works

### Step 1 — Document Ingestion
1. User uploads a PDF via the sidebar.
2. `PDFLoader` extracts text from all pages using PyMuPDF.
3. `TextSplitter` divides the text into 500-character chunks with 50-character overlap.
4. `EmbeddingEngine` encodes each chunk using `all-MiniLM-L6-v2` (384-dim vectors, L2-normalised).
5. `VectorStore` builds an in-memory FAISS `IndexFlatIP` (cosine similarity via dot product).

### Step 2 — Question Answering
6. The user's question is encoded into a query embedding.
7. FAISS returns the top-4 most similar chunks.
8. Chunks are assembled into a context string and sent to **Gemini 1.5 Flash** with a grounding prompt.
9. Gemini returns a natural-language answer.

### Step 3 — Verification
10. `HallucinationDetector` computes cosine similarity between the answer embedding and each chunk embedding.
11. `TrustScore` applies the formula: `(0.7 × max_sim + 0.3 × mean_sim) × 100`.
12. If score < 40%, `ResponseValidator` re-calls Gemini with a strict "use only this context" prompt.
13. The corrected answer is re-scored and displayed alongside the original.

---

## Example Usage & Testing

Use the content in `data/sample_policy.txt` — print it to PDF and upload it.

### High-trust questions (expected score ≥ 70%)

| Question | Expected Answer |
|---|---|
| How many leave days do full-time employees get? | 15 days |
| What are standard office hours? | 9 AM – 6 PM, Monday–Friday |
| How many days can I work remotely per week? | Up to 2 days |
| What percentage of health insurance does the company cover? | 80% |
| What is the annual learning budget per employee? | ₹25,000 |

### Low-trust questions (expected score < 40% → triggers correction)

| Question | Why it hallucinate-risks |
|---|---|
| Does the company offer a 401k plan? | Not in document |
| What is the CEO's name? | Not in document |
| How many vacation days do contractors get? | Not in document |
| What is the company's stock price? | Not in document |

### Hallucination demo

Ask: **"How many leave days does the company provide?"**

If Gemini says "30 days" (hallucination):
- Similarity between "30 days" and "15 days" context → low cosine score
- Trust score drops below 40%
- Correction triggered → Gemini re-reads context strictly → returns "15 days"

---

## Trust Score Formula

```
similarity_i  = cosine_similarity(embed(answer), embed(chunk_i))

trust_score = round(
    (0.70 × max(similarity_i) + 0.30 × mean(similarity_i)) × 100
)
```

| Score | Status | UI Colour | Action |
|---|---|---|---|
| 70 – 100 | ✅ High Trust | Green | Display as-is |
| 40 – 69 | ⚠️ Medium Trust | Amber | Display with warning |
| 0 – 39 | 🚨 Low Trust | Red | Auto-correct with grounding prompt |

**Why this formula?**
- `max_similarity` (70% weight): if the answer matches *any* chunk well, it's likely grounded.
- `mean_similarity` (30% weight): penalises answers that broadly diverge from the retrieved context.

---

## Future Enhancements

- **NLI-based verification** — use a Natural Language Inference model (e.g. `cross-encoder/nli-deberta-v3-base`) for sentence-level entailment checking
- **Persistent FAISS storage** — save and reload indexes without re-embedding
- **Multi-document support** — index and query across multiple PDFs simultaneously
- **Citation highlighting** — highlight the exact sentence in the source chunk that supports the answer
- **Confidence calibration** — calibrate trust scores against human-labelled ground truth
- **REST API** — wrap the pipeline in FastAPI for programmatic access
- **OCR support** — handle scanned PDFs using Tesseract or Google Vision

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit 1.35 |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (CPU, in-memory) |
| PDF Parsing | PyMuPDF (fitz) |
| Language | Python 3.10+ |

---

## License

MIT — free to use, modify, and distribute with attribution.

---

*Built as a final-year B.Tech CSE project — 2024*
