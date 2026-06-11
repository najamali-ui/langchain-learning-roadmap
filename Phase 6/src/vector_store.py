"""src/vector_store.py — FAISS-backed resume store with HuggingFace embeddings."""
from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.config import EMBEDDING_MODEL, FAISS_INDEX_PATH


# ── singleton embedding model ───────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── FAISS helpers ───────────────────────────────────────────────────────────
class ResumeVectorStore:
    """Wraps a FAISS index to store and retrieve candidate resumes."""

    def __init__(self):
        self._store: Optional[FAISS] = None
        self._metadata: Dict[str, Dict[str, Any]] = {}   # doc_id → meta
        self._load_if_exists()

    # ── persistence ────────────────────────────────────────────────────────
    def _index_dir(self) -> Path:
        return Path(FAISS_INDEX_PATH)

    def _load_if_exists(self):
        idx_dir = self._index_dir()
        meta_path = idx_dir / "metadata.pkl"
        if idx_dir.exists() and (idx_dir / "index.faiss").exists():
            try:
                self._store = FAISS.load_local(
                    str(idx_dir),
                    get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                if meta_path.exists():
                    with open(meta_path, "rb") as f:
                        self._metadata = pickle.load(f)
            except Exception:
                self._store = None
                self._metadata = {}

    def _save(self):
        idx_dir = self._index_dir()
        idx_dir.mkdir(parents=True, exist_ok=True)
        if self._store:
            self._store.save_local(str(idx_dir))
        with open(idx_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

    # ── public API ──────────────────────────────────────────────────────────
    def add_resume(self, candidate_name: str, filename: str, text: str) -> str:
        """Embed and store one resume. Returns a doc_id."""
        doc_id = f"{candidate_name}::{filename}"

        doc = Document(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "candidate_name": candidate_name,
                "filename": filename,
            },
        )

        if self._store is None:
            self._store = FAISS.from_documents([doc], get_embeddings())
        else:
            self._store.add_documents([doc])

        self._metadata[doc_id] = {
            "candidate_name": candidate_name,
            "filename": filename,
            "text_preview": text[:400],
        }
        self._save()
        return doc_id

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k matching resume chunks with metadata."""
        if self._store is None:
            return []
        docs_scores = self._store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in docs_scores:
            meta = doc.metadata.copy()
            meta["similarity_score"] = float(score)
            meta["content"] = doc.page_content
            results.append(meta)
        return results

    def list_candidates(self) -> List[str]:
        return [m["candidate_name"] for m in self._metadata.values()]

    def candidate_count(self) -> int:
        return len(self._metadata)

    def clear(self):
        self._store = None
        self._metadata = {}
        idx_dir = self._index_dir()
        for f in idx_dir.glob("*"):
            f.unlink()
        self._save()
