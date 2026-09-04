import pytest

from exceptions.llm_err import LLMAuthError
from src.llm.llm_client import LLMClient, extract_json


def test_extract_json_plain_object():
    assert extract_json('{"reply": "hi", "mood": "calm"}') == {"reply": "hi", "mood": "calm"}


def test_extract_json_from_code_fence():
    text = 'Here you go:\n```json\n{"reply": "hi"}\n```\nHope that helps!'
    assert extract_json(text) == {"reply": "hi"}


def test_extract_json_embedded_in_prose():
    text = 'Sure! {"mood": "sad", "reply": "I hear you."} Let me know.'
    assert extract_json(text) == {"mood": "sad", "reply": "I hear you."}


def test_extract_json_returns_none_when_absent():
    assert extract_json("no json here at all") is None
    assert extract_json("") is None


def test_extract_json_strips_reasoning_block_with_braces():
    # A reasoning model (e.g. Nemotron) emits <think>...</think> — which may
    # itself contain braces — before the real structured reply. The tool call
    # must survive; a naive first-{-to-last-} scan would drop it.
    text = (
        "<think>The user asked the time. I should call {get_current_time}.</think>\n"
        '{"mood":"neutral","intent":"task_request","sensitivity":"low",'
        '"reply":"Sure.","action":{"name":"get_current_time","args":{}}}'
    )
    out = extract_json(text)
    assert out is not None
    assert out["action"]["name"] == "get_current_time"
    assert out["reply"] == "Sure."


def test_extract_json_handles_unclosed_think():
    # Model hit the token cap mid-thought: an unclosed <think> and no JSON.
    assert extract_json("<think>hmm let me consider {this}") is None


def test_extract_json_prefers_structured_reply_over_decoy():
    # When prose contains a decoy object plus the real reply object, pick the
    # one that looks like our structured reply.
    text = 'note {"foo": 1} then {"reply": "hello", "mood": "calm"}'
    assert extract_json(text)["reply"] == "hello"


def test_local_provider_needs_no_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(
        model="llama3.2",
        base_url="http://localhost:11434/v1",
        require_key=False,
        provider="ollama",
    )
    assert client.base_url == "http://localhost:11434/v1"
    assert client.provider == "ollama"
    assert client.model == "llama3.2"


def test_hosted_provider_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMAuthError):
        LLMClient(model="gpt-4o-mini")


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(model="gpt-4o-mini", api_key="sk-explicit")
    assert client.api_key == "sk-explicit"
    assert client.base_url is None
