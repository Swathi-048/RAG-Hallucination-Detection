"""
config.py
---------
Central configuration for the entire RAG pipeline.
All tuneable parameters live here — never scattered across files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────────────────────
GEMINI_MODEL: str       = "gemini-1.5-flash"
GEMINI_TEMPERATURE: float = 0.2       # lower = more factual/deterministic
GEMINI_MAX_TOKENS: int  = 1024

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int   = 384             # output dimension of MiniLM-L6-v2

# ── Text Splitting ────────────────────────────────────────────────────────────
CHUNK_SIZE: int    = 500               # characters per chunk
CHUNK_OVERLAP: int = 50                # overlap between adjacent chunks
MIN_CHUNK_LEN: int = 30                # discard chunks shorter than this

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 4                         # number of chunks to retrieve per query

# ── Trust / Hallucination Thresholds ─────────────────────────────────────────
HIGH_TRUST_THRESHOLD: int  = 70        # ≥70 → High trust
LOW_TRUST_THRESHOLD: int   = 40        # <40 → trigger correction
# Weight formula: trust = HIGH_W*max_sim + LOW_W*mean_sim (then ×100)
HIGH_W: float = 0.70
LOW_W: float  = 0.30

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str  = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"
