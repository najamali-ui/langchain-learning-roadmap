"""
test.py — Diagnostic script to verify FAISS retrieval quality.

Run this before launching the Streamlit app to confirm:
  1. The embedding model matches the stored index
  2. The top-k results for sample queries look sensible

Usage:
    python test.py
"""

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# FIX 1: must match the model used when the index was originally created
# The stored index has vector dim=768, which is nomic-embed-text.
# Using "qwen" (dim=2560) causes wrong or crashed retrieval.
EMBEDDING_MODEL = "nomic-embed-text"
FAISS_PATH = "faiss_index"
TOP_K = 6  # FIX 3: retrieve more chunks to cover scattered facts

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

# Sample queries to verify retrieval quality
queries = [
    "What is the rate of poor people having safe drinking water?",
    "How many individuals benefited from safe drinking water programs?",
    "What are the WASH achievements in Sanghar?",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)

    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n  --- Chunk {i} (score={score:.4f}, page={doc.metadata.get('page','?')}) ---")
        print(f"  {doc.page_content[:300]}")
