"""app.py — AI Resume Screener  (run: streamlit run app.py)"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.resume_parser import extract_text
from src.vector_store import ResumeVectorStore
from src.screener_chain import build_screener_chain, build_screening_prompt
from src.config import TOP_K_RESULTS, QWEN_MODEL


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ---------- Global ---------- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p { color: #cbd5e1 !important; }

/* ---------- Header banner ---------- */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0ea5e9 100%);
    border-radius: 14px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
}
.hero-banner h1 { font-size: 2rem; font-weight: 800; margin: 0; }
.hero-banner p  { font-size: 1rem; opacity: .85; margin: 6px 0 0; }

/* ---------- Metric cards ---------- */
.metric-row { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #0ea5e9; }
.metric-card .lbl { font-size: .8rem; color: #64748b; margin-top: 4px; }

/* ---------- Chat bubbles ---------- */
.chat-user {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
}
.chat-assistant {
    background: #f0fdf4; border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
}
.chat-label { font-size: .72rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px; }

/* ---------- Status badges ---------- */
.badge-hire  { background:#dcfce7; color:#166534; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.badge-maybe { background:#fef9c3; color:#854d0e; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.badge-no    { background:#fee2e2; color:#991b1b; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }

/* ---------- Section dividers ---------- */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #1e293b;
    border-bottom: 2px solid #0ea5e9; padding-bottom: 6px;
    margin: 24px 0 16px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = ResumeVectorStore()
    if "screener_chain" not in st.session_state:
        st.session_state.screener_chain = build_screener_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []          # list of {"role","content"}
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []
    if "last_screening_result" not in st.session_state:
        st.session_state.last_screening_result = None
    if "last_search_results" not in st.session_state:
        st.session_state.last_search_results = []

_init_state()
vs: ResumeVectorStore = st.session_state.vector_store
chain = st.session_state.screener_chain


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 AI Resume Screener")
    st.caption(f"Model: `{QWEN_MODEL}`")
    st.divider()

    # ── Upload resumes ────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        help="Upload one or more candidate resumes.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            key = uf.name
            if key not in st.session_state.uploaded_names:
                with st.spinner(f"Parsing {uf.name}…"):
                    text = extract_text(uf, uf.name)
                    candidate_name = uf.name.rsplit(".", 1)[0].replace("_", " ").title()
                    vs.add_resume(candidate_name, uf.name, text)
                    st.session_state.uploaded_names.append(key)
                st.success(f"✅ {uf.name} indexed")

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────
    count = vs.candidate_count()
    st.metric("Candidates in DB", count)

    if count > 0:
        names = vs.list_candidates()
        with st.expander("📋 Indexed candidates"):
            for n in names:
                st.write(f"• {n}")

    st.divider()

    # ── Clear DB ──────────────────────────────────────────────────────────
    if st.button("🗑 Clear all resumes", use_container_width=True):
        vs.clear()
        st.session_state.uploaded_names = []
        st.session_state.last_screening_result = None
        st.session_state.last_search_results = []
        st.rerun()

    # ── Clear chat ────────────────────────────────────────────────────────
    if st.button("💬 Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.screener_chain = build_screener_chain()
        chain = st.session_state.screener_chain
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown(
    """
<div class="hero-banner">
  <h1>🧠 AI Resume Screener</h1>
  <p>Upload resumes → paste a job description → let Qwen rank your candidates.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Quick metrics
col1, col2, col3 = st.columns(3)
col1.metric("📄 Resumes Indexed", vs.candidate_count())
col2.metric("💬 Chat Turns", len(st.session_state.chat_history) // 2)
col3.metric("🤖 LLM", QWEN_MODEL)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_screen, tab_chat, tab_about = st.tabs(
    ["🔍 Screen Candidates", "💬 HR Chat Assistant", "ℹ️ About"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCREEN CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────
with tab_screen:
    st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)

    job_desc = st.text_area(
        "Paste the full job description here",
        height=220,
        placeholder=(
            "e.g.  We are looking for a Senior Python Developer with 5+ years experience "
            "in FastAPI, PostgreSQL, and cloud deployments (AWS/GCP). Strong ML/LLM knowledge preferred…"
        ),
    )

    top_k = st.slider("How many candidates to retrieve?", 1, min(10, max(vs.candidate_count(), 1)), TOP_K_RESULTS)

    screen_btn = st.button(
        "🚀 Screen Candidates",
        use_container_width=True,
        disabled=(vs.candidate_count() == 0 or not job_desc.strip()),
    )

    if vs.candidate_count() == 0:
        st.info("⬅️ Upload at least one resume in the sidebar to get started.")

    if screen_btn and job_desc.strip():
        with st.spinner("🔍 Searching vector store…"):
            results = vs.search(job_desc, k=top_k)

        if not results:
            st.warning("No candidates found in the database.")
        else:
            # Cache results so the chart reuses them without a second FAISS search
            st.session_state.last_search_results = results
            prompt = build_screening_prompt(job_desc, results)

            st.markdown('<div class="section-title">Screening Results</div>', unsafe_allow_html=True)
            response_placeholder = st.empty()
            full_response = ""

            # Stream tokens live so user sees output immediately
            for chunk in chain.llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Save to memory manually since we bypassed chain.predict
            chain.memory.chat_memory.add_user_message("Screen candidates for the provided job description.")
            chain.memory.chat_memory.add_ai_message(full_response)

            st.session_state.last_screening_result = full_response
            st.session_state.chat_history.append({"role": "user", "content": "Screen candidates for the provided job description."})
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    if st.session_state.last_screening_result and not screen_btn:
        st.markdown('<div class="section-title">Screening Results</div>', unsafe_allow_html=True)
        st.markdown(st.session_state.last_screening_result)

        # ── similarity score chart — reuse cached results, no second FAISS search ──
        results_for_chart = st.session_state.get("last_search_results", [])
        if results_for_chart:
            df = pd.DataFrame([
                {
                    "Candidate": r["candidate_name"],
                    "Similarity": round(1 / (1 + r["similarity_score"]), 3),
                }
                for r in results_for_chart
            ]).sort_values("Similarity", ascending=False)

            fig = px.bar(
                df,
                x="Similarity",
                y="Candidate",
                orientation="h",
                title="Vector Similarity to Job Description",
                color="Similarity",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=max(300, 50 * len(df)), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — HR CHAT ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<div class="section-title">Ask anything about the screened candidates</div>', unsafe_allow_html=True)
    st.caption("The assistant remembers the full conversation (ConversationBufferMemory).")

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user"><div class="chat-label">👤 YOU</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant"><div class="chat-label">🤖 QWEN</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. Which candidate has the strongest cloud experience?",
        )
        send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Optionally enrich with FAISS context
        rag_context = ""
        if vs.candidate_count() > 0:
            hits = vs.search(user_input, k=3)
            if hits:
                snippets = "\n\n".join(
                    f"[{h['candidate_name']}]: {h['content'][:600]}" for h in hits
                )
                rag_context = f"\n\nRelevant resume context:\n{snippets}\n\n"

        full_input = rag_context + user_input

        chat_placeholder = st.empty()
        full_reply = ""

        # Stream tokens live
        for chunk in chain.llm.stream(full_input):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_reply += token
            chat_placeholder.markdown(
                f'<div class="chat-assistant"><div class="chat-label">🤖 QWEN</div>{full_reply}▌</div>',
                unsafe_allow_html=True,
            )
        chat_placeholder.empty()

        # Save to memory manually
        chain.memory.chat_memory.add_user_message(full_input)
        chain.memory.chat_memory.add_ai_message(full_reply)

        st.session_state.chat_history.append({"role": "assistant", "content": full_reply})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
## 🧠 AI Resume Screener — Architecture

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Qwen (via OpenAI-compatible endpoint) |
| Orchestration | LangChain `ConversationChain` |
| Memory | `ConversationBufferMemory` (last N turns) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | FAISS (persisted to disk) |
| Resume parsing | PyMuPDF (PDF) · python-docx (DOCX) |
| Charts | Plotly |

### How it works
1. **Upload** — resumes are parsed to plain text, embedded, and stored in a local FAISS index.
2. **Screen** — a job description is embedded and used to retrieve the most relevant candidates from FAISS.
3. **Rank** — retrieved candidates are passed to Qwen with a structured scoring prompt.
4. **Chat** — the HR chat assistant uses `ConversationBufferMemory` to maintain context across turns, and optionally injects live FAISS snippets as RAG context.

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env file and fill in your settings
cp .env.example .env

# 3. Make sure Qwen is running (e.g. via Ollama)
ollama pull qwen2:7b
ollama serve

# 4. Run
streamlit run app.py
```
"""
    )
