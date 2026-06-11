"""utils/resume_parser.py — extract raw text from PDF / DOCX resumes."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union


# ── PDF ────────────────────────────────────────────────────────────────────
def _parse_pdf(file: Union[bytes, io.BytesIO]) -> str:
    try:
        import fitz  # PyMuPDF
        if isinstance(file, bytes):
            doc = fitz.open(stream=file, filetype="pdf")
        else:
            doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        return f"[PDF parse error: {e}]"


# ── DOCX ───────────────────────────────────────────────────────────────────
def _parse_docx(file: Union[bytes, io.BytesIO]) -> str:
    try:
        from docx import Document
        if isinstance(file, bytes):
            file = io.BytesIO(file)
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX parse error: {e}]"


# ── Plain text ─────────────────────────────────────────────────────────────
def _parse_txt(file: Union[bytes, io.BytesIO]) -> str:
    if isinstance(file, io.BytesIO):
        return file.read().decode("utf-8", errors="replace")
    return file.decode("utf-8", errors="replace")


# ── Public API ─────────────────────────────────────────────────────────────
def extract_text(file_obj, filename: str) -> str:
    """
    Accept a Streamlit UploadedFile (or raw bytes) and return clean text.
    """
    ext = Path(filename).suffix.lower()
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj

    if ext == ".pdf":
        text = _parse_pdf(raw_bytes)
    elif ext in (".docx", ".doc"):
        text = _parse_docx(io.BytesIO(raw_bytes))
    elif ext == ".txt":
        text = _parse_txt(raw_bytes)
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    return _clean(text)


def _clean(text: str) -> str:
    """Remove excessive whitespace while preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
