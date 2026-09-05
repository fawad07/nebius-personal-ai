"""
src/tools/web_search.py

A `web_search` tool backed by the Tavily API — the one tool that leaves the
machine. Tavily is an LLM-oriented search API: it returns a synthesized answer
plus ranked sources, which is exactly the shape a voice assistant wants to speak
back.

Exposed as a voice-agent ``Tool`` (dict in, short string out) and registered
only when a ``TAVILY_API_KEY`` is present, so the assistant degrades gracefully
without one. This is a real runtime call to the Tavily API (hackathon "Best Use
of Tavily" bonus).
"""

from __future__ import annotations

import logging

import requests

from src.tools.registry import Tool

logger = logging.getLogger("tools")

_TAVILY_URL = "https://api.tavily.com/search"


def _make_tavily_search(api_key: str, max_results: int = 5, timeout: float = 15.0):
    def search(args: dict) -> str:
        query = str((args or {}).get("query", "")).strip()
        if not query:
            return "(no query given for web search)"
        try:
            resp = requests.post(
                _TAVILY_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "basic",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Tavily search failed: %s", e)
            return f"(web search failed: {e})"

        results = data.get("results") or []
        answer = (data.get("answer") or "").strip()
        if answer:
            if results:
                sources = "; ".join(r.get("title", "") for r in results[:3] if r.get("title"))
                if sources:
                    return f"{answer} (Sources: {sources})"
            return answer
        if not results:
            return f"No web results found for '{query}'."
        # No synthesized answer: hand back the top snippets for the model to phrase.
        lines = [
            f"{r.get('title', 'result')}: {str(r.get('content', '')).strip()[:200]}"
            for r in results[:3]
        ]
        return " | ".join(lines)

    return search


def build_web_search_tool(api_key: str, max_results: int = 5) -> Tool:
    return Tool(
        name="web_search",
        description=(
            "Search the web for current or factual information (news, prices, "
            "facts, anything the model may not know). Use ONLY when the user "
            "explicitly asks to look something up online."
        ),
        params='{"query": str}',
        func=_make_tavily_search(api_key, max_results=max_results),
    )
