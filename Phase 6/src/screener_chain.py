"""src/screener_chain.py — LangChain chain using Qwen + ConversationBufferMemory."""
from __future__ import annotations

from typing import List, Dict, Any

from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

from src.config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    QWEN_MODEL,
    MEMORY_BUFFER_SIZE,
)


# ── LLM ────────────────────────────────────────────────────────────────────
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=QWEN_MODEL,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3,
        max_tokens=1024,
        streaming=True,
    )


# ── Memory (Buffer) ─────────────────────────────────────────────────────────
def build_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        k=MEMORY_BUFFER_SIZE,
    )


# ── Screening prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Recruitment Assistant specialising in resume screening and candidate evaluation.

Your responsibilities:
1. Analyse resumes against job requirements with structured, objective scoring.
2. Rank candidates clearly, stating strengths, gaps, and a final recommendation.
3. Answer follow-up HR questions using context from the conversation history.
4. Always be concise, professional, and fair — avoid bias based on names or demographics.

When scoring candidates use this rubric:
- Skills match (0–40 pts)
- Experience relevance (0–30 pts)
- Education fit (0–20 pts)
- Overall presentation / clarity (0–10 pts)

Respond with structured Markdown when ranking or scoring."""


def build_screener_chain() -> ConversationChain:
    llm = build_llm()
    memory = build_memory()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False,
    )
    return chain


# ── Helper: build screening query ──────────────────────────────────────────
def build_screening_prompt(
    job_description: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """Turn FAISS search results into a structured prompt for the LLM."""
    candidate_blocks = []
    for i, c in enumerate(candidates, 1):
        block = (
            f"### Candidate {i}: {c.get('candidate_name', 'Unknown')}\n"
            f"**File:** {c.get('filename', '')}\n\n"
            f"{c.get('content', '')[:800]}\n"
            f"---"
        )
        candidate_blocks.append(block)

    return (
        f"## Job Description\n{job_description}\n\n"
        f"## Candidates Retrieved (top {len(candidates)})\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\n"
        "Please **score and rank** each candidate against the job description "
        "using the rubric in your system instructions. "
        "For each candidate provide: overall score, brief rationale, "
        "top 3 strengths, top 2 gaps, and a hire/maybe/no recommendation. "
        "End with a final ranked summary table."
    )
