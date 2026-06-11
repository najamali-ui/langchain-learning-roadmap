import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="AI-Powered PDF Search Assistant", layout="wide")

# ── Constants ────────────────────────────────────────────────
PDF_FILE_PATH  = "RDF-Annual-Report-2023-72.pdf"
FAISS_INDEX_PATH = "faiss_index"

# FIX 1 ─ Embedding model consistency
# ──────────────────────────────────────────────────────────────
# The FAISS index was built (and is stored on disk) using
# OllamaEmbeddings(model="nomic-embed-text")  — vector dim 768.
#
# app.py was using OllamaEmbeddings(model="qwen") — vector dim 2560.
#
# When the query embedding (2560-d) was compared against stored
# index vectors (768-d) the dimensions didn't match, causing either
# a runtime crash or FAISS returning completely wrong neighbours.
# Fix: use the SAME "nomic-embed-text" model that was used to build
# the index.  If you ever rebuild the index, keep this the same.
EMBEDDING_MODEL = "nomic-embed-text"

# FIX 3 ─ How many chunks to retrieve
# ──────────────────────────────────────────────────────────────
# The original k=3 meant only 3 chunks were fed to the LLM.
# A question like "how much safe drinking water access do poor
# people have" draws facts scattered across at least 7 chunks
# (pages 22, 25-28).  With k=3 most of the relevant numbers
# were simply never sent to the LLM, so it either hallucinated
# or said "I don't know".
# Fix: raise k to 6 so the LLM sees enough context.
RETRIEVER_K = 6

st.title("📄 AI-Powered PDF Search Assistant")
st.write(f"Querying: `{PDF_FILE_PATH}`")


# ── Build / load the vector store (cached across reruns) ─────
@st.cache_resource
def load_and_index_document(pdf_path, index_path):
    """
    Load PDF, split into chunks, embed with nomic-embed-text, store in FAISS.

    FIX 1: embedding model is now nomic-embed-text (matches the stored index).
    FIX 2: junk/cover-page chunks are filtered out before embedding.
    FIX 4: chunk_size raised 1000→1500 and overlap raised 200→300 so
            numbers and their context land in the same chunk more often.
    """
    # FIX 1: use the correct embedding model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Re-use existing index if available
    if os.path.exists(index_path):
        st.info("✅ Loading existing vector store…")
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    # ── Step 1: Load PDF ─────────────────────────────────────
    progress_bar = st.progress(0, text="Step 1/3 — Loading PDF…")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    progress_bar.progress(10, text=f"Step 1/3 — Loaded {len(docs)} pages from PDF")

    # ── Step 2: Split into chunks ────────────────────────────
    progress_bar.progress(15, text="Step 2/3 — Splitting into chunks…")

    # FIX 4: larger chunks keep stat + context together; more overlap
    # prevents a sentence from being split right at the chunk boundary.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = splitter.split_documents(docs)

    # FIX 2: drop near-empty chunks (cover/divider pages like "2023\n2023")
    # They pollute the vector space and can surface as false positives.
    MIN_CHUNK_CHARS = 50
    splits = [s for s in splits if len(s.page_content.strip()) >= MIN_CHUNK_CHARS]

    total_chunks = len(splits)
    progress_bar.progress(20, text=f"Step 2/3 — Created {total_chunks} chunks (after filtering)")

    # ── Step 3: Embed chunks in batches ──────────────────────
    BATCH_SIZE = 10
    vector_store = None

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = splits[i : i + BATCH_SIZE]
        pct = 20 + int(((i + len(batch)) / total_chunks) * 75)
        progress_bar.progress(
            min(pct, 95),
            text=f"Step 3/3 — Embedding chunks {i + 1}–{min(i + BATCH_SIZE, total_chunks)} of {total_chunks}…",
        )
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vector_store.merge_from(batch_store)

    # ── Save to disk ─────────────────────────────────────────
    progress_bar.progress(98, text="Saving vector store to disk…")
    vector_store.save_local(index_path)
    progress_bar.progress(100, text="✅ Vector store ready!")
    st.success(f"Indexed {total_chunks} chunks from {len(docs)} pages.")

    return vector_store


# ── Helper: format retrieved docs into a single string ───────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Main app logic ───────────────────────────────────────────
try:
    if not os.path.exists(PDF_FILE_PATH):
        st.error(f"Error: `{PDF_FILE_PATH}` not found in the current directory.")
        st.stop()

    vectorstore = load_and_index_document(PDF_FILE_PATH, FAISS_INDEX_PATH)

    # FIX 3: k=6 instead of k=3 — fetch more chunks so scattered facts are covered
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # LLM — keep qwen for generation (that's fine; only embeddings needed fixing)
    llm = ChatOllama(model="qwen", temperature=0)

    # FIX 5 ─ Better system prompt
    # ──────────────────────────────────────────────────────────
    # The original prompt said "three sentences max" which forced the
    # LLM to omit numbers or aggregate stats even when they existed in
    # the context.  For an annual-report assistant users expect precise
    # figures.  New prompt: answer fully with all numbers from context,
    # still stay concise, but do NOT truncate facts.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant for answering questions about the RDF Annual Report 2023. "
                "Use ONLY the provided context to answer. "
                "Include all relevant numbers, statistics, and figures you find in the context — "
                "do not omit data. "
                "If the context contains partial information, share what is available and note any gaps. "
                "If the answer is not in the context at all, say so clearly. "
                "Be thorough but concise.\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── User input ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Ask a Question")

    # FIX 6 ─ Note about financial pages
    # The financial statement pages (44-46) are scanned images embedded
    # inside the PDF — PyPDFLoader cannot extract text from them.
    # Surface this as an informational note so users aren't confused.
    with st.expander("ℹ️ Document notes", expanded=False):
        st.info(
            "**Financial statements (pages 44–46)** are scanned images in the PDF and "
            "cannot be searched — only the text-based programme pages are indexed.\n\n"
            f"**Embedding model:** `{EMBEDDING_MODEL}` &nbsp;|&nbsp; "
            f"**Chunks retrieved per query:** {RETRIEVER_K}"
        )

    user_query = st.text_input("Enter your question here:")

    if user_query:
        import time

        # Step 1: Retrieve relevant docs
        t0 = time.time()
        source_docs = retriever.invoke(user_query)
        t_retrieval = time.time() - t0

        context_text = format_docs(source_docs)

        # Step 2: Build the prompt with retrieved context
        messages = prompt.invoke({"context": context_text, "question": user_query})

        # Step 3: Stream the LLM response token-by-token
        st.markdown("### Answer")
        t1 = time.time()
        response_container = st.empty()
        full_response = ""
        for chunk in llm.stream(messages):
            full_response += chunk.content
            response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
        t_generation = time.time() - t1

        st.caption(
            f"⏱ Retrieval: {t_retrieval:.2f}s  |  Generation: {t_generation:.2f}s  "
            f"|  Total: {t_retrieval + t_generation:.2f}s  |  "
            f"Chunks used: {len(source_docs)}"
        )

        # Show the source chunks
        with st.expander("Show Sources"):
            for i, doc in enumerate(source_docs):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i + 1}** (Page {page}):")
                st.write(doc.page_content)

except Exception as e:
    st.error(f"An error occurred: {e}")
