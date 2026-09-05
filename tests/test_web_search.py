"""Tests for the Tavily-backed web_search tool (HTTP mocked)."""

from unittest.mock import patch, MagicMock

from src.tools.web_search import build_web_search_tool


def _resp(payload):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = payload
    return m


def test_returns_answer_with_sources():
    tool = build_web_search_tool("k")
    payload = {
        "answer": "Paris is the capital of France.",
        "results": [{"title": "Wikipedia: France"}, {"title": "Britannica"}],
    }
    with patch("src.tools.web_search.requests.post", return_value=_resp(payload)) as post:
        out = tool.func({"query": "capital of France"})
    assert "Paris is the capital of France." in out
    assert "Sources:" in out
    # It made a real POST to Tavily with the query and key.
    _, kwargs = post.call_args
    assert kwargs["json"]["query"] == "capital of France"
    assert kwargs["json"]["api_key"] == "k"


def test_falls_back_to_snippets_without_answer():
    tool = build_web_search_tool("k")
    payload = {"answer": "", "results": [{"title": "T", "content": "some content"}]}
    with patch("src.tools.web_search.requests.post", return_value=_resp(payload)):
        out = tool.func({"query": "x"})
    assert "some content" in out


def test_empty_query():
    tool = build_web_search_tool("k")
    assert "no query" in tool.func({"query": "  "}).lower()


def test_network_error_is_caught():
    import requests
    tool = build_web_search_tool("k")
    with patch("src.tools.web_search.requests.post", side_effect=requests.RequestException("boom")):
        out = tool.func({"query": "x"})
    assert "web search failed" in out.lower()
