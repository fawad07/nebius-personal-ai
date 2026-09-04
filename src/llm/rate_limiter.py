import threading
import time

from exceptions.llm_err import LLMBudgetExceededError, LLMRateLimitExceeded


class LLMUsageGuard:
    """
    Local, in-process guard against runaway LLM usage:
      - a sliding-window request-rate limit (calls per minute)
      - a cumulative token budget for the process/session lifetime

    These are soft, self-imposed caps -- not a substitute for provider-side
    billing alerts -- but they turn a bug (e.g. a retry storm, a runaway
    loop) into an immediate, clear exception instead of a silent runaway
    bill. Thread-safe since LLMClient calls happen from worker threads via
    asyncio.to_thread.
    """

    def __init__(self, max_calls_per_minute: int = 30, max_tokens_total: int | None = 200_000):
        self.max_calls_per_minute = max_calls_per_minute
        self.max_tokens_total = max_tokens_total
        self._call_times: list[float] = []
        self._tokens_used = 0
        self._lock = threading.Lock()

    def check_and_record_call(self):
        """Raise if the call-rate limit is exceeded, otherwise record this call."""
        with self._lock:
            now = time.monotonic()
            window_start = now - 60
            self._call_times = [t for t in self._call_times if t >= window_start]
            if len(self._call_times) >= self.max_calls_per_minute:
                raise LLMRateLimitExceeded(
                    f"LLM call rate exceeded: {self.max_calls_per_minute} calls/min limit."
                )
            self._call_times.append(now)

    def check_budget(self):
        """Raise if the cumulative token budget has already been exhausted."""
        with self._lock:
            if self.max_tokens_total is not None and self._tokens_used >= self.max_tokens_total:
                raise LLMBudgetExceededError(
                    f"LLM token budget exceeded: {self._tokens_used}/{self.max_tokens_total} tokens used."
                )

    def record_usage(self, total_tokens: int):
        if not total_tokens:
            return
        with self._lock:
            self._tokens_used += total_tokens

    @property
    def tokens_used(self) -> int:
        with self._lock:
            return self._tokens_used
