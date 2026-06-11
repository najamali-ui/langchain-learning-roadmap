"""
app.py — AI Research Assistant
A premium Streamlit UI powered by LangChain agents, Ollama (Qwen), SerpAPI Search, and SQLite memory.
"""

import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# ─── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Main area ── */
[data-testid="stMainBlockContainer"] {
    padding: 1rem 2rem !important;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8 !important;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Chat messages ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1rem;
}
.msg-bubble {
    display: flex;
    gap: 0.75rem;
    animation: fadeSlideIn 0.3s ease;
}
.msg-bubble.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.avatar-user { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.avatar-ai   { background: linear-gradient(135deg, #0ea5e9, #6366f1); }

.msg-content {
    max-width: 78%;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    line-height: 1.65;
    font-size: 0.9rem;
}
.content-user {
    background: linear-gradient(135deg, #312e81, #4338ca);
    border: 1px solid rgba(99,102,241,0.4);
    border-top-right-radius: 4px;
    color: #e0e7ff !important;
    text-align: right;
}
.content-ai {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.15);
    border-top-left-radius: 4px;
    color: #e2e8f0 !important;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Status badges ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.9rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    animation: pulse 1.5s ease-in-out infinite;
}
.status-search {
    background: rgba(234, 179, 8, 0.15);
    border: 1px solid rgba(234, 179, 8, 0.4);
    color: #fbbf24;
}
.status-analyze {
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.4);
    color: #a78bfa;
}
.status-write {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #4ade80;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* ── Source cards ── */
.sources-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin: 0.75rem 0 0.4rem;
}
.source-chip {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-radius: 6px;
    font-size: 0.72rem;
    color: #7dd3fc;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}

/* ── Query chips ── */
.query-chip {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    background: rgba(234, 179, 8, 0.08);
    border: 1px solid rgba(234, 179, 8, 0.2);
    border-radius: 6px;
    font-size: 0.72rem;
    color: #fde68a;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input area ── */
[data-testid="stChatInput"] textarea {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── Sidebar components ── */
.sidebar-section {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #64748b !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.session-item {
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid transparent;
    margin-bottom: 0.3rem;
}
.session-item:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.2);
}
.session-active {
    background: rgba(99,102,241,0.15) !important;
    border-color: rgba(99,102,241,0.3) !important;
}

/* ── Expander override ── */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}

/* ── Text inputs in sidebar ── */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 1rem;
    padding: 0.6rem 1rem;
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.78rem;
    color: #64748b;
}
.stat-item { display: flex; align-items: center; gap: 0.35rem; }
.stat-val { color: #818cf8; font-weight: 600; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #475569;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.3rem; font-weight: 700; color: #64748b; margin-bottom: 0.5rem; }
.empty-desc  { font-size: 0.85rem; line-height: 1.6; }
.suggestion-chips { margin-top: 1.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center; }
.suggestion {
    padding: 0.45rem 0.9rem;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 99px;
    font-size: 0.78rem;
    color: #818cf8;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ─── Imports (after page config) ──────────────────────────────────────────────
from database.db import (
    init_db, create_session, list_sessions, delete_session,
    get_session_messages, update_session_title
)
from agent.research_agent import run_research

# Ensure DB is ready
init_db()


# ─── Session State Initialization ─────────────────────────────────────────────
def init_state():
    defaults = {
        "current_session_id": None,
        "messages": [],          # list of {"role", "content", "sources", "queries"}
        "thinking": False,
        "serpapi_key": os.getenv("SERPAPI_API_KEY", ""),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "temperature": float(os.getenv("AGENT_TEMPERATURE", "0.3")),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Helper functions ──────────────────────────────────────────────────────────

def start_new_session():
    session_id = create_session("New Research Session")
    st.session_state.current_session_id = session_id
    st.session_state.messages = []


def load_session(session_id: str):
    st.session_state.current_session_id = session_id
    rows = get_session_messages(session_id)
    msgs = []
    for i, row in enumerate(rows):
        role = "user" if row["role"] == "human" else "assistant"
        content = row["content"]
        msgs.append({"role": role, "content": content, "sources": [], "queries": []})
    st.session_state.messages = msgs


def format_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return iso_str


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">🔬 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── API Configuration ──
    st.markdown('<div class="sidebar-label">⚙️ Configuration</div>', unsafe_allow_html=True)
    with st.container():
        serpapi_key = st.text_input(
            "SerpAPI Key",
            value=st.session_state.serpapi_key,
            type="password",
            placeholder="Paste your SerpAPI key here…",
            help="Get your free key at serpapi.com",
            key="serpapi_key_input"
        )
        if serpapi_key:
            st.session_state.serpapi_key = serpapi_key

        # Fix 3: removed qwen:latest — Qwen 1.x has no tool-calling support.
        # qwen2 / qwen2.5 both support the tools API required by LangGraph ReAct.
        QWEN_MODELS = [
            "qwen2.5:7b",       # recommended default — fast, tool-capable
            "qwen2.5:latest",
            "qwen2.5:14b",
            "qwen2.5:32b",
            "qwen2:latest",
            "qwen2:7b",
        ]
        current_model = st.session_state.ollama_model
        model_idx = QWEN_MODELS.index(current_model) if current_model in QWEN_MODELS else 0  # defaults to qwen2.5:7b
        model = st.selectbox(
            "Ollama Model (Qwen)",
            QWEN_MODELS,
            index=model_idx,
            help="Select the Qwen model pulled in Ollama",
            key="model_select"
        )
        st.session_state.ollama_model = model

        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher = more creative; Lower = more factual",
            key="temp_slider"
        )
        st.session_state.temperature = temperature

        st.markdown(
            '<div style="font-size:0.72rem;color:#475569;margin-top:0.4rem">'
            '🤖 LLM runs locally via <b style="color:#818cf8">Ollama</b> — no cloud costs!'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Session Management ──
    st.markdown('<div class="sidebar-label">💬 Sessions</div>', unsafe_allow_html=True)

    col_new, col_refresh = st.columns([3, 1])
    with col_new:
        if st.button("＋ New Session", key="btn_new_session", use_container_width=True):
            start_new_session()
            st.rerun()

    sessions = list_sessions()

    if sessions:
        st.markdown(f'<div style="font-size:0.72rem;color:#475569;margin:0.3rem 0;">Recent Sessions ({len(sessions)})</div>', unsafe_allow_html=True)
        for s in sessions[:15]:
            is_active = s["id"] == st.session_state.current_session_id
            label = s["title"][:28] + "…" if len(s["title"]) > 28 else s["title"]
            badge = "🟢 " if is_active else ""
            col_s, col_d = st.columns([5, 1])
            with col_s:
                if st.button(
                    f"{badge}{label}",
                    key=f"sess_{s['id']}",
                    use_container_width=True,
                    help=f"Last updated: {format_time(s['updated_at'])}"
                ):
                    load_session(s["id"])
                    st.rerun()
            with col_d:
                if st.button("🗑", key=f"del_{s['id']}", help="Delete session"):
                    if s["id"] == st.session_state.current_session_id:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    delete_session(s["id"])
                    st.rerun()
    else:
        st.caption("No sessions yet. Start a new one!")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem;color:#334155;text-align:center">'
        'Powered by LangChain · Ollama (Qwen) · SerpAPI · SQLite'
        '</div>',
        unsafe_allow_html=True
    )


# ─── Main Content ──────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔬 AI Research Assistant</div>
    <div class="hero-subtitle">
        LangChain Agents · Ollama Qwen · SerpAPI Search · Persistent Memory
    </div>
</div>
""", unsafe_allow_html=True)

# Stats bar
if st.session_state.current_session_id:
    n_msgs = len(st.session_state.messages)
    n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
    all_sources = [s for m in st.session_state.messages for s in m.get("sources", [])]
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-item">💬 <span class="stat-val">{n_user}</span> questions</div>
        <div class="stat-item">📨 <span class="stat-val">{n_msgs}</span> messages</div>
        <div class="stat-item">🌐 <span class="stat-val">{len(all_sources)}</span> sources found</div>
    </div>
    """, unsafe_allow_html=True)

# Guard: no session yet
if not st.session_state.current_session_id:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔬</div>
        <div class="empty-title">Start Your Research</div>
        <div class="empty-desc">
            Ask anything — the agent will search the web in real-time,<br>
            synthesize information, and remember your conversation.
        </div>
        <div class="suggestion-chips">
            <span class="suggestion">🧬 Latest AI breakthroughs in 2025</span>
            <span class="suggestion">🌍 Climate change solutions</span>
            <span class="suggestion">💊 CRISPR gene editing progress</span>
            <span class="suggestion">🚀 SpaceX Starship updates</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start New Session", key="btn_start_hero"):
        start_new_session()
        st.rerun()
    st.stop()


# ─── Chat Display ──────────────────────────────────────────────────────────────
chat_area = st.container()

with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💡</div>
            <div class="empty-title">Session Ready</div>
            <div class="empty-desc">
                Type your research question below to begin.<br>
                The agent will search, analyze, and summarize information for you.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])
            queries = msg.get("queries", [])

            if role == "user":
                st.markdown(f"""
                <div class="msg-bubble user">
                    <div class="msg-avatar avatar-user">👤</div>
                    <div class="msg-content content-user">{content}</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="msg-bubble">
                    <div class="msg-avatar avatar-ai">🤖</div>
                    <div class="msg-content content-ai">
                        {content.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Sources and queries
                if queries or sources:
                    with st.expander("🔍 Research Details", expanded=False):
                        if queries:
                            st.markdown('<div class="sources-header">Search Queries Used</div>', unsafe_allow_html=True)
                            chips = "".join(f'<span class="query-chip">🔎 {q}</span>' for q in queries)
                            st.markdown(chips, unsafe_allow_html=True)
                        if sources:
                            st.markdown('<div class="sources-header">Sources</div>', unsafe_allow_html=True)
                            chips = "".join(f'<span class="source-chip">🔗 {s[:60]}{"…" if len(s)>60 else ""}</span>' for s in sources)
                            st.markdown(chips, unsafe_allow_html=True)


# ─── Chat Input ────────────────────────────────────────────────────────────────
if not st.session_state.serpapi_key:
    st.warning("⚠️ Please enter your **SerpAPI Key** in the sidebar to enable web search.", icon="🔑")
    st.stop()

if question := st.chat_input("Ask a research question... (e.g. 'What are the latest AI developments in 2025?')"):

    # Add user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
        "queries": [],
    })

    # Auto-title session from first question
    if len(st.session_state.messages) == 1:
        title = question[:50] + ("…" if len(question) > 50 else "")
        update_session_title(st.session_state.current_session_id, title)

    # Show status indicators while processing
    with st.status("🔬 Researching...", expanded=True) as status_box:
        st.write("🔍 Searching the web for relevant information...")
        time.sleep(0.5)
        st.write("🧠 Analyzing and synthesizing findings...")

        response = run_research(
            question=question,
            session_id=st.session_state.current_session_id,
            serpapi_key=st.session_state.serpapi_key,
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
        )

        st.write("✍️ Generating structured response...")
        time.sleep(0.2)

        if response.has_error:
            status_box.update(label="❌ Research Failed", state="error")
        else:
            status_box.update(label="✅ Research Complete!", state="complete")

    if response.has_error:
        # Fix 4: show a friendly user-facing message; hide raw error behind expander
        # to avoid leaking internal details (tracebacks, URLs, ports) into the chat.
        friendly_error = (
            "⚠️ **Research failed.** "
            f"There was a problem running the agent with model `{st.session_state.ollama_model}`.\n\n"
            "**Quick checklist:**\n"
            "- Ollama is running → `ollama serve`\n"
            f"- Model is pulled → `ollama pull {st.session_state.ollama_model}`\n"
            "- SerpAPI key is set in the sidebar\n"
            "- Internet connection is available\n"
        )
        with st.expander("🔎 Error details (for debugging)", expanded=False):
            st.code(response.error, language="")
        st.session_state.messages.append({
            "role": "assistant",
            "content": friendly_error,
            "sources": [],
            "queries": [],
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "queries": response.search_queries,
        })

    st.rerun()
