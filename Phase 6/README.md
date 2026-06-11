# 🧠 AI Resume Screener

> **Phase 6 Mastery Project** — LangChain + Qwen + FAISS + Streamlit

---

## ✨ Features

| Feature | Detail |
|---|---|
| Resume upload | PDF, DOCX, TXT |
| Vector search | FAISS + HuggingFace embeddings (local, no API key) |
| LLM ranking | Qwen via OpenAI-compatible endpoint |
| Memory | LangChain `ConversationBufferMemory` |
| HR Chat | Ask follow-up questions with full context |
| Charts | Plotly similarity bar chart |
| Persistence | FAISS index saved to disk between sessions |

---

## 🗂 Project Structure

```
ai_resume_screener/
├── app.py                  ← Streamlit entry point  (only file you run)
├── requirements.txt
├── .env.example            ← copy to .env and fill in
├── data/
│   └── resumes/            ← optional: store sample resumes here
│   └── faiss_index/        ← auto-created when you index resumes
├── src/
│   ├── __init__.py
│   ├── config.py           ← reads .env, central settings
│   ├── vector_store.py     ← FAISS wrapper (add / search / persist)
│   └── screener_chain.py   ← LangChain chain + Qwen LLM + buffer memory
└── utils/
    ├── __init__.py
    └── resume_parser.py    ← PDF / DOCX / TXT extraction
```

---

## ⚡ Quick Start

### Option A — Qwen locally via Ollama (recommended, free)

```bash
# 1. Install Ollama  →  https://ollama.com
ollama pull qwen2:7b      # or qwen2:1.5b for low-RAM machines
ollama serve              # starts on http://localhost:11434

# 2. Clone / copy this folder, then:
cd ai_resume_screener
pip install -r requirements.txt

# 3. Create .env
cp .env.example .env
# defaults already point to Ollama, no edits needed for local use

# 4. Launch
streamlit run app.py
```

### Option B — Alibaba DashScope (cloud Qwen)

```bash
# In .env set:
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-your-key-here
QWEN_MODEL=qwen-turbo          # or qwen-plus / qwen-max
```

---

## 🖥 Usage

1. **Sidebar → Upload Resumes**: drag-drop PDF/DOCX files. Each is parsed and embedded into FAISS automatically.
2. **Screen Candidates tab**: paste a job description → click **Screen Candidates**. Qwen returns a scored, ranked list.
3. **HR Chat tab**: ask follow-up questions ("Who has the most AWS experience?", "Compare candidates 1 and 3").
4. **Clear buttons** in sidebar reset the vector DB or chat history independently.

---

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_BASE` | `http://localhost:11434/v1` | Ollama or DashScope base URL |
| `OPENAI_API_KEY` | `ollama` | API key (use `ollama` for local) |
| `QWEN_MODEL` | `qwen2:7b` | Model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Where FAISS index is saved |
| `MEMORY_BUFFER_SIZE` | `10` | Chat turns kept in buffer memory |

---

## 📦 Key Dependencies

```
streamlit          — UI
langchain          — LLM orchestration + memory
langchain-community — ChatOpenAI, FAISS, HuggingFaceEmbeddings
faiss-cpu          — vector similarity search
sentence-transformers — local embeddings (no API key)
PyMuPDF            — PDF text extraction
python-docx        — DOCX text extraction
plotly             — similarity chart
```
