import json
import os
import re
import threading

from openai import AuthenticationError, BadRequestError, OpenAI

from exceptions.llm_err import LLMAuthError, LLMResponseError
from src.llm.rate_limiter import LLMUsageGuard


# Reasoning models (e.g. NVIDIA Nemotron) prepend a chain-of-thought block,
# typically wrapped in <think>...</think>, before the actual answer. That block
# often contains its own braces, which would defeat a naive "first { to last }"
# scan and make us miss the real JSON object (silently dropping the tool call).
# Strip such blocks before extraction.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    # An unclosed <think> (model hit the token cap mid-thought) — drop to end.
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def _iter_json_objects(text: str):
    """Yield every top-level balanced JSON object found in ``text``, in order."""
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    yield obj
                    i = end
                    continue
            except json.JSONDecodeError:
                pass
        i += 1


def extract_json(text: str) -> dict | None:
    """
    Best-effort extraction of a JSON object from a model response. Handles the
    common ways instruct/reasoning models deviate from strict JSON mode: a clean
    object, an object wrapped in ```json ... ``` fences, an object embedded in
    surrounding prose, and a reasoning preamble (<think>...</think>) before the
    object. Returns the parsed dict, or None if nothing parses.
    """
    if not text:
        return None

    cleaned = _strip_reasoning(text)

    # 1) Fast path: the (cleaned) response is exactly one JSON object.
    for candidate in (cleaned, text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 2) Fenced ```json ... ``` block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3) Scan for all balanced objects and pick the most likely structured
    #    reply: prefer one carrying our expected keys ("reply"/"action"), then
    #    the richest (most keys). This survives braces in surrounding prose.
    objs = list(_iter_json_objects(cleaned))
    if not objs:
        return None
    objs.sort(key=lambda o: (("reply" in o) or ("action" in o), len(o)))
    return objs[-1]


class LLMClient:
    """
    Wrapper around any OpenAI-compatible Chat Completions API.
    Supports:
    - plain text
    - JSON
    - streaming tokens

    Works with hosted OpenAI *and* local, OpenAI-compatible servers such as
    Ollama (http://localhost:11434/v1) or LM Studio -- point ``base_url`` at
    the local endpoint and set ``require_key=False`` for a fully offline LLM
    leg. The same request/streaming/JSON code path serves both.

    Enforces two local safety nets on every call: a sliding-window
    call-rate limit and a cumulative token budget (see LLMUsageGuard).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        require_key: bool = True,
        provider: str = "openai",
        max_calls_per_minute: int = 30,
        max_tokens_total: int | None = 200_000,
        max_tokens: int | None = None,
        system_preamble: str = "",
    ):
        self.model = model
        self.provider = provider
        self.base_url = base_url
        # Per-response output cap (keeps latency + cost bounded); None = model default.
        self.max_tokens = max_tokens
        # Prepended to every system prompt. For NVIDIA Nemotron reasoning models,
        # "detailed thinking off" disables chain-of-thought — turning ~50s voice
        # turns into ~1s. Harmless plain text for models that don't recognize it.
        self.system_preamble = (system_preamble or "").strip()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if require_key and not self.api_key:
            raise LLMAuthError("OPENAI_API_KEY not set.")
        # Local servers ignore the key but the SDK still requires a non-empty
        # placeholder string.
        self.client = OpenAI(api_key=self.api_key or "local", base_url=base_url)
        self.usage_guard = LLMUsageGuard(
            max_calls_per_minute=max_calls_per_minute,
            max_tokens_total=max_tokens_total,
        )

    def validate_api_key(self):
        """
        Cheap startup connectivity/auth check: list models. For hosted OpenAI
        this fails fast on a bad/revoked key; for a local endpoint it confirms
        the server (e.g. Ollama) is actually up and reachable. Listing models
        carries no token cost.
        """
        try:
            self.client.models.list()
        except AuthenticationError as e:
            raise LLMAuthError(f"API key was rejected by the LLM endpoint: {e}") from e
        except Exception as e:
            if self.base_url:
                raise LLMAuthError(
                    f"Could not reach local LLM endpoint at {self.base_url} "
                    f"({self.provider}). Is the server running? Details: {e}"
                ) from e
            raise LLMAuthError(f"Failed to validate LLM credentials: {e}") from e

    def warmup(self) -> None:
        """Fire a tiny completion so a serverless model cold-starts NOW.

        Nebius (and most serverless inference) spins a model up on the first
        request after it goes idle, which can add ~30-60s. Doing that here — at
        startup — keeps it off the user's first spoken turn. Best-effort: any
        failure is swallowed (a failed warmup must never block startup).
        """
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
        except Exception:
            pass

    def warmup_async(self) -> None:
        """Warm the model in a background daemon thread (non-blocking)."""
        threading.Thread(target=self.warmup, daemon=True).start()

    def _guard_before_call(self):
        self.usage_guard.check_budget()
        self.usage_guard.check_and_record_call()

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """Build the chat messages, prepending the configured system preamble."""
        system = system_prompt
        if self.system_preamble:
            system = f"{self.system_preamble}\n\n{system_prompt}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

    def _complete(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """
        Run one chat completion and return the raw content string. When
        ``json_mode`` is requested but the endpoint rejects the
        ``response_format`` parameter (common on local OpenAI-compatible
        servers), transparently retry without it -- lenient JSON extraction
        downstream still recovers the object.
        """
        kwargs = {"model": self.model, "messages": self._messages(system_prompt, user_prompt)}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not json_mode:
                raise
            kwargs.pop("response_format", None)
            resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage_guard.record_usage(resp.usage.total_tokens)
        return resp.choices[0].message.content or ""

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        self._guard_before_call()
        return self._complete(system_prompt, user_prompt, json_mode=False).strip()

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        self._guard_before_call()
        content = self._complete(system_prompt, user_prompt, json_mode=True)
        parsed = extract_json(content)
        if parsed is None:
            raise LLMResponseError(f"Model did not return parseable JSON: {content[:200]!r}")
        return parsed

    def stream_text(self, system_prompt: str, user_prompt: str):
        """
        Yield tokens as they are generated. Usage is only available on the
        final chunk (stream_options.include_usage), so it's recorded after
        the stream is exhausted.
        """
        self._guard_before_call()
        stream_kwargs = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": self._messages(system_prompt, user_prompt),
        }
        if self.max_tokens is not None:
            stream_kwargs["max_tokens"] = self.max_tokens
        stream = self.client.chat.completions.create(**stream_kwargs)
        for chunk in stream:
            if chunk.usage:
                self.usage_guard.record_usage(chunk.usage.total_tokens)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
