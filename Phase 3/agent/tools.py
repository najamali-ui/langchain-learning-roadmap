"""
agent/tools.py — LangChain tools for the AI Research Assistant.

Tools available:
  - SerpAPISearchTool: Searches Google via SerpAPI (requires API key)
  - WebContentFetcherTool: Fetches and cleans content from a specific URL
"""

import re
import urllib.request
from html.parser import HTMLParser

from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import Tool
from config import config


# ─── SerpAPI Search Tool ──────────────────────────────────────────────────────

def get_search_tool(serpapi_key: str) -> Tool:
    """
    Returns a configured SerpAPI search tool using the provided API key.
    Searches Google for real-time results.
    """
    search = SerpAPIWrapper(
        serpapi_api_key=serpapi_key,
        params={
            "engine": "google",
            "num": config.MAX_SEARCH_RESULTS,
            "hl": "en",
            "gl": "us",
        },
    )
    return Tool(
        name="web_search",
        func=search.run,
        description=(
            "Search the web (Google) for current, real-time information. "
            "Use this whenever you need up-to-date facts, news, research, or any information "
            "you are not confident about. "
            "Input should be a clear, concise search query. "
            "Returns a summary of the top search results."
        ),
    )


# ─── HTML Stripper (Fix 5: handle nested skip-tags correctly) ─────────────────

class _HTMLStripper(HTMLParser):
    """HTML tag stripper that correctly handles nested skip-tag depth."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        # Fix 5: use a depth counter per skip-tag instead of a single
        # _current_tag string. The old approach reset _current_tag to ""
        # on ANY closing tag, so closing a nested <div> inside <script>
        # would re-enable data capture prematurely.
        self._skip_depth: int = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._text.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._text)


# ─── Web Content Fetcher Tool ─────────────────────────────────────────────────

@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and extract readable text content from a specific webpage URL.
    Use this when you have a specific URL and need its full content for deeper research.
    Input must be a valid HTTP/HTTPS URL.
    Returns the cleaned text content of the page (first 3000 characters).
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")

        stripper = _HTMLStripper()
        stripper.feed(html)
        text = stripper.get_text()

        # Clean up excessive whitespace
        text = re.sub(r"\s{3,}", "  ", text)
        return text[:3000] if len(text) > 3000 else text

    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"
