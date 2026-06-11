"""
agent/research_agent.py — LangChain/LangGraph ReAct agent for AI Research Assistant.

Architecture:
  - Uses Ollama (Qwen2.5+) as the local reasoning LLM — no OpenAI required
  - Equipped with SerpAPI (Google Search) + webpage fetcher tools
  - Maintains conversation memory via SQLite-backed history
  - Returns a structured ResearchResponse with answer + sources
"""

from dataclasses import dataclass, field
from typing import Optional
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import config
from agent.tools import get_search_tool, fetch_webpage
from memory.sqlite_memory import SQLiteChatMessageHistory


# ─── Models that support tool/function calling ─────────────────────────────────
# qwen:latest (Qwen 1.x) does NOT support tools — it returns HTTP 400.
# Qwen2 and Qwen2.5 both support the tools API.
TOOLS_CAPABLE_PREFIXES = ("qwen2", "qwen2.5")


def _supports_tools(model: str) -> bool:
    """Return True if the model name indicates tool-call support."""
    normalized = model.lower().replace(" ", "")
    return any(normalized.startswith(p) for p in TOOLS_CAPABLE_PREFIXES)


# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI Research Assistant with access to real-time web search.

Your core capabilities:
- Search the web for up-to-date, accurate information
- Synthesize information from multiple sources into clear, structured answers
- Remember the conversation context to provide relevant follow-up answers
- Cite your sources transparently

Instructions:
1. For any factual question, ALWAYS use the web_search tool to get current information
2. If a search result mentions a URL that seems highly relevant, use fetch_webpage to get deeper details
3. Structure your final answer clearly with sections where appropriate
4. Always end with a "**Sources:**" section listing URLs or search queries used
5. Be concise but comprehensive
6. Use the conversation history for context when answering follow-up questions"""


# ─── Response Model ────────────────────────────────────────────────────────────

@dataclass
class ResearchResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


# ─── Source Extraction ─────────────────────────────────────────────────────────

def _extract_sources(text: str) -> list[str]:
    """Extract URLs and source references from agent output."""
    urls = re.findall(r'https?://[^\s\)\]"<>,]+', text)
    sources_section = re.search(r'\*{0,2}Sources:?\*{0,2}(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    bullets = []
    if sources_section:
        bullets = re.findall(r'[-*•]\s*(.+)', sources_section.group(1))
    return list(dict.fromkeys(urls + bullets))


# ─── Main Research Function ────────────────────────────────────────────────────

def run_research(
    question: str,
    session_id: str,
    serpapi_key: str,
    model: str = "qwen2.5:7b",
    temperature: float = 0.3,
) -> ResearchResponse:
    """
    Run the research agent for a given question with persistent memory.

    Args:
        question:     User's research question
        session_id:   Unique session ID for memory isolation
        serpapi_key:  SerpAPI key for Google search
        model:        Ollama model name — must support tool calling (qwen2 / qwen2.5)
        temperature:  LLM temperature

    Returns:
        ResearchResponse with answer, sources, and search queries used
    """
    # Fix 3: Validate tool support before attempting agent creation.
    # qwen:latest (Qwen 1.x) rejects tool specs with HTTP 400.
    if not _supports_tools(model):
        return ResearchResponse(
            answer="",
            error=(
                f"Model '{model}' does not support tool/function calling.\n\n"
                f"Please select a tool-capable model such as:\n"
                f"  • qwen2.5:7b  (recommended)\n"
                f"  • qwen2.5:latest\n"
                f"  • qwen2:latest\n\n"
                f"Run: ollama pull qwen2.5:7b"
            ),
        )

    try:
        llm = ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

        tools = [get_search_tool(serpapi_key), fetch_webpage]

        # LangGraph ReAct agent — replaces deprecated AgentExecutor
        agent = create_react_agent(
            model=llm,
            tools=tools,
        )

        # Load chat history from SQLite and build message list
        history = SQLiteChatMessageHistory(session_id=session_id)
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history.messages:
            messages.append(msg)
        messages.append(HumanMessage(content=question))

        # Invoke agent
        result = agent.invoke({"messages": messages})

        # The last message in result["messages"] is the final AI answer
        all_messages = result.get("messages", [])
        answer_msg = None
        search_queries = []

        for msg in reversed(all_messages):
            cls_name = type(msg).__name__
            if cls_name == "AIMessage" and answer_msg is None:
                # Skip tool-call messages (they have tool_calls populated)
                if not getattr(msg, "tool_calls", None):
                    answer_msg = msg

        # Collect search queries from tool messages
        for msg in all_messages:
            cls_name = type(msg).__name__
            if cls_name == "AIMessage" and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if "search" in tc.get("name", "").lower():
                        args = tc.get("args", {})
                        query = args.get("query", args.get("__arg1", str(args)))
                        search_queries.append(query)

        answer = answer_msg.content if answer_msg else "No answer returned."

        # Extract sources from answer text
        sources = _extract_sources(answer)

        # Persist the exchange to SQLite
        history.add_user_message(question)
        history.add_ai_message(answer)

        return ResearchResponse(
            answer=answer,
            sources=sources,
            search_queries=search_queries,
        )

    except Exception as e:
        return ResearchResponse(
            answer="",
            error=str(e),
        )
