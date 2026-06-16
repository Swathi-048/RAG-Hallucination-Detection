"""
app.py
------
Streamlit entry point for the RAG Hallucination Detection system.

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag.rag_pipeline              import RAGPipeline
from verification.response_validator import ResponseValidator
from utils.helpers                   import truncate_text, validate_question, validate_api_key
from utils.logger                    import get_logger
from config                          import GOOGLE_API_KEY, HIGH_TRUST_THRESHOLD, LOW_TRUST_THRESHOLD

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Hallucination Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — dark research-tool aesthetic
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0d14; color: #d1d5db; }

/* ── Cards ─────────────────────────────────────────────────────── */
.card {
    background: #13151f;
    border: 1px solid #1f2235;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 16px;
}
.card-accent-blue   { border-left: 4px solid #6366f1; }
.card-accent-green  { border-left: 4px solid #4ade80; }
.card-accent-amber  { border-left: 4px solid #fbbf24; }
.card-accent-red    { border-left: 4px solid #f87171; }

/* ── Trust bar ─────────────────────────────────────────────────── */
.trust-track {
    background: #1f2235;
    border-radius: 999px;
    height: 12px;
    margin: 10px 0 6px;
    overflow: hidden;
}
.trust-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── Badges ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-high   { background:#14532d; color:#4ade80; }
.badge-medium { background:#713f12; color:#fbbf24; }
.badge-low    { background:#7f1d1d; color:#f87171; }
.badge-corrected { background:#1e1b4b; color:#a5b4fc; }

/* ── Labels ────────────────────────────────────────────────────── */
.eyebrow {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 4px;
    font-weight: 600;
}

/* ── Chunk pills ───────────────────────────────────────────────── */
.chunk {
    background: #0f1120;
    border-left: 3px solid #4338ca;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.80rem;
    color: #9ca3af;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.chunk-score {
    font-size: 0.70rem;
    color: #4b5563;
    margin-bottom: 4px;
    font-family: 'Inter', sans-serif;
}

/* ── Answer text ───────────────────────────────────────────────── */
.answer-text {
    font-size: 1.0rem;
    line-height: 1.75;
    color: #e2e8f0;
}

/* ── Correction notice ─────────────────────────────────────────── */
.correction-notice {
    background: #1e1b4b;
    border: 1px solid #4338ca;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.88rem;
    color: #a5b4fc;
    margin-bottom: 16px;
}

/* ── Similarity bars ───────────────────────────────────────────── */
.sim-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
    font-size: 0.82rem;
    color: #9ca3af;
}
.sim-bar-bg { flex:1; background:#1f2235; border-radius:999px; height:8px; }
.sim-bar    { height:100%; border-radius:999px; background:#6366f1; }
.sim-val    { width: 46px; text-align: right; font-family: 'JetBrains Mono', monospace; }

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0f1120; border-right: 1px solid #1f2235; }

/* ── Hide Streamlit chrome ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "pipeline":    None,
        "validator":   None,
        "doc_name":    "",
        "chunk_count": 0,
        "history":     [],   # list of result dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
# Helper: initialise pipeline (cached per session)
# ══════════════════════════════════════════════════════════════════════════════

def _get_pipeline() -> RAGPipeline:
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline  = RAGPipeline()
            st.session_state.validator = ResponseValidator(st.session_state.pipeline)
        except EnvironmentError as e:
            st.error(str(e))
            logger.error(f"Pipeline initialization failed: {e}")
            return None
    return st.session_state.pipeline

# ══════════════════════════════════════════════════════════════════════════════
# Helper: render one Q&A result card
# ══════════════════════════════════════════════════════════════════════════════

def _render_result(r: dict, idx: int):
    trust = r["trust"]
    score = trust["trust_score"]
    color = trust["color"]

    # Pick badge class
    if score >= HIGH_TRUST_THRESHOLD:
        badge_cls, badge_emoji = "badge-high",   "✅"
    elif score >= LOW_TRUST_THRESHOLD:
        badge_cls, badge_emoji = "badge-medium", "⚠️"
    else:
        badge_cls, badge_emoji = "badge-low",    "🚨"

    # ── Trust score card ──────────────────────────────────────────
    st.markdown(f"""
    <div class='card card-accent-blue'>
      <div style='display:flex; justify-content:space-between; align-items:center;'>
        <span class='eyebrow'>Trust Score</span>
        <span class='badge {badge_cls}'>{badge_emoji} {score}% — {trust["label"]}</span>
      </div>
      <div class='trust-track'>
        <div class='trust-fill' style='width:{score}%; background:{color};'></div>
      </div>
      <div style='color:#6b7280; font-size:0.83rem; margin-top:6px;'>{trust["details"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Correction notice ─────────────────────────────────────────
    if r["was_corrected"]:
        st.markdown("""
        <div class='correction-notice'>
            🔄 <strong>Auto-corrected</strong> — Low trust score triggered a strict
            re-generation. The answer below is grounded exclusively in the document.
        </div>
        """, unsafe_allow_html=True)

    # ── Answer card ───────────────────────────────────────────────
    accent = "green" if score >= HIGH_TRUST_THRESHOLD else ("amber" if score >= LOW_TRUST_THRESHOLD else "red")
    st.markdown(f"""
    <div class='card card-accent-{accent}'>
      <div class='eyebrow'>Answer</div>
      <div class='answer-text' style='margin-top:8px;'>{r["final_answer"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Expandable details ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("📊 Similarity breakdown"):
            st.markdown(
                f"**Max:** `{trust['max_sim']:.4f}` &nbsp;|&nbsp; "
                f"**Mean:** `{trust['mean_sim']:.4f}`"
            )
            if trust.get("per_chunk"):
                st.markdown("<br>", unsafe_allow_html=True)
                for ci, s in enumerate(trust["per_chunk"], 1):
                    pct = int(s * 100)
                    st.markdown(f"""
                    <div class='sim-row'>
                        <span>Chunk {ci}</span>
                        <div class='sim-bar-bg'><div class='sim-bar' style='width:{pct}%;'></div></div>
                        <span class='sim-val'>{s:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

    with col_b:
        with st.expander("📄 Retrieved context chunks"):
            chunks  = r["retrieved_chunks"]
            scores  = [rc["score"] for rc in chunks]
            for ci, rc in enumerate(chunks, 1):
                st.markdown(
                    f"<div class='chunk-score'>Chunk {ci} · retrieval score: {scores[ci-1]:.4f}</div>"
                    f"<div class='chunk'>{rc['chunk']}</div>",
                    unsafe_allow_html=True,
                )

    if r["was_corrected"]:
        with st.expander("🔍 Initial answer (before correction)"):
            st.markdown(
                f"<div class='chunk'>{r['initial_answer']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#1f2235; margin:24px 0 8px;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:1.4rem; font-weight:800; color:#a5b4fc;'>🛡️ RAG Guard</div>
        <div style='color:#4b5563; font-size:0.80rem; margin-top:4px;'>
            Hallucination Detection & Trust Verification
        </div>
    </div>
    <hr style='border-color:#1f2235; margin:10px 0 18px;'>
    """, unsafe_allow_html=True)

    # ── API Key check ─────────────────────────────────────────────
    if not validate_api_key(GOOGLE_API_KEY):
        st.error("⚠️ `GOOGLE_API_KEY` not found.\nAdd it to your `.env` file and restart.")

    # ── Upload section ────────────────────────────────────────────
    st.markdown("<div class='eyebrow'>Document</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name != st.session_state.doc_name:
            with st.spinner("Indexing document…"):
                try:
                    pipeline = _get_pipeline()
                    if pipeline is None:
                        # _get_pipeline already displayed an error message
                        st.error("Pipeline unavailable. Check API key and restart the app.")
                    else:
                        info = pipeline.ingest(uploaded.read(), filename=uploaded.name)
                        st.session_state.doc_name    = info["filename"]
                        st.session_state.chunk_count = info["chunks"]
                        st.session_state.history     = []
                        st.success(f"✅ Ready — {info['chunks']} chunks indexed")
                        logger.info(f"Document ingested via UI: {info}")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    logger.error(f"Ingest error: {e}")

    # ── Document info ─────────────────────────────────────────────
    if st.session_state.doc_name:
        st.markdown(f"""
        <div style='background:#0f1120; border:1px solid #1f2235; border-radius:10px;
                    padding:12px 14px; margin-top:12px;'>
            <div class='eyebrow'>Loaded</div>
            <div style='color:#a5b4fc; font-weight:600; word-break:break-word; margin-top:4px;'>
                📄 {st.session_state.doc_name}
            </div>
            <div style='color:#4b5563; font-size:0.80rem; margin-top:6px;'>
                {st.session_state.chunk_count} chunks indexed
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ How it works"):
        st.markdown("""
**Pipeline steps**

1. PDF → text extraction (PyMuPDF)
2. Text → 500-char overlapping chunks
3. Chunks → embeddings (MiniLM-L6-v2)
4. Embeddings → FAISS index

**Per question**

5. Query → embedding → FAISS search → top-4 chunks
6. Context + question → Gemini 1.5 Flash
7. Answer + chunks → cosine similarity
8. Similarity → Trust Score (0-100 %)
9. If score < 40 % → strict re-prompt → corrected answer

**Trust Score Formula**

`score = (0.7 × max_sim + 0.3 × mean_sim) × 100`
        """)

    # ── Clear history ─────────────────────────────────────────────
    if st.session_state.history:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Main panel
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:28px 0 4px;'>
    <h1 style='font-size:2rem; font-weight:800; color:#a5b4fc; letter-spacing:-0.5px; margin:0;'>
        RAG Hallucination Detection
    </h1>
    <p style='color:#4b5563; font-size:0.92rem; margin-top:6px;'>
        Upload a document · ask questions · every answer is verified against the source
    </p>
</div>
<hr style='border-color:#1f2235; margin:12px 0 20px;'>
""", unsafe_allow_html=True)

# ── Question input ────────────────────────────────────────────────────────────
pipeline_ready = (
    st.session_state.pipeline is not None
    and st.session_state.pipeline.is_ready
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the company's leave policy?",
    disabled=not pipeline_ready,
    label_visibility="collapsed",
)

mode = st.radio(
    "Answer mode",
    options=["Grounded only", "Open AI-style"],
    help="Grounded only answers stay tied to the document; open mode may use broader knowledge.",
    index=0,
)
allow_external = mode == "Open AI-style"

ask_col, _ = st.columns([1, 5])
with ask_col:
    ask = st.button(
        "Get Answer →",
        disabled=(not pipeline_ready or not question.strip()),
        use_container_width=True,
    )

if not pipeline_ready:
    st.markdown("""
    <div class='card' style='text-align:center; padding:40px 20px; color:#4b5563;'>
        ⬅️ &nbsp; Upload a PDF in the sidebar to get started
    </div>
    """, unsafe_allow_html=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if ask and question.strip() and pipeline_ready:
    is_valid, err = validate_question(question)
    if not is_valid:
        st.warning(err)
    else:
        with st.spinner("Retrieving context and verifying answer…"):
            try:
                pipeline  = _get_pipeline()
                validator = st.session_state.validator

                # 1. RAG answer
                rag_result = pipeline.answer(question, allow_external_knowledge=allow_external)

                # 2. Validate + possibly correct
                verification = validator.validate(
                    question          = question,
                    answer            = rag_result["answer"],
                    retrieved_chunks  = [r["chunk"] for r in rag_result["retrieved_chunks"]],
                )

                # Bundle for display
                result = {
                    "question":        question,
                    "initial_answer":  verification["initial_answer"],
                    "final_answer":    verification["final_answer"],
                    "was_corrected":   verification["was_corrected"],
                    "trust":           verification["trust"],
                    "retrieved_chunks": rag_result["retrieved_chunks"],
                }

                st.session_state.history.insert(0, result)
                logger.info(
                    f"Q: {question[:80]} | "
                    f"Trust: {result['trust']['trust_score']}% | "
                    f"Corrected: {result['was_corrected']}"
                )
            except Exception as e:
                st.error(f"Error: {e}")
                logger.error(f"Pipeline error: {e}", exc_info=True)

# ── Render history ────────────────────────────────────────────────────────────
for i, result in enumerate(st.session_state.history):
    st.markdown(f"""
    <div style='margin-bottom:6px;'>
        <span class='eyebrow'>Q{len(st.session_state.history) - i}</span>
        <span style='color:#e2e8f0; font-size:1.0rem; font-weight:600; margin-left:8px;'>
            {result["question"]}
        </span>
    </div>
    """, unsafe_allow_html=True)
    _render_result(result, i)
