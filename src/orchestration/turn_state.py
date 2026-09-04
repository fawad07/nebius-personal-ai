import logging
import time
from enum import Enum


class TurnState(str, Enum):
    """The phase a conversation turn is currently in."""

    IDLE = "idle"              # not yet started / between runs
    LISTENING = "listening"    # capturing + endpointing user speech
    THINKING = "thinking"      # perception (STT/speaker/SER) + LLM
    SPEAKING = "speaking"      # synthesizing and playing the reply
    INTERRUPTED = "interrupted"  # user barged in during playback


# Expected forward transitions. Enforcement is *soft* (unexpected moves are
# logged, not raised) so a state bug can never crash a live conversation.
_ALLOWED = {
    TurnState.IDLE: {TurnState.LISTENING, TurnState.IDLE},
    TurnState.LISTENING: {TurnState.THINKING, TurnState.LISTENING, TurnState.IDLE},
    TurnState.THINKING: {TurnState.SPEAKING, TurnState.LISTENING, TurnState.IDLE},
    TurnState.SPEAKING: {TurnState.INTERRUPTED, TurnState.LISTENING, TurnState.SPEAKING, TurnState.IDLE},
    TurnState.INTERRUPTED: {TurnState.LISTENING, TurnState.THINKING, TurnState.IDLE},
}


class TurnStateMachine:
    """
    Tracks the turn lifecycle as first-class, observable transitions.

    An optional ``on_change(old, new, elapsed_ms)`` callback is invoked on each
    transition, where ``elapsed_ms`` is the time spent in the previous state --
    a natural hook for a UI to reflect status, or for per-phase latency
    tracing. The callback is best-effort and never allowed to break the loop.
    """

    def __init__(self, on_change=None, logger=None):
        self.state = TurnState.IDLE
        self._on_change = on_change
        self._logger = logger or logging.getLogger("session_manager")
        self._since = time.monotonic()

    def transition(self, to: TurnState) -> TurnState:
        old = self.state
        now = time.monotonic()
        elapsed_ms = (now - self._since) * 1000.0

        if to != old and to not in _ALLOWED.get(old, set()):
            self._logger.debug(f"Unexpected turn transition: {old.value} -> {to.value}")

        self.state = to
        self._since = now
        self._logger.debug(f"[state] {old.value} -> {to.value} ({elapsed_ms:.0f}ms in {old.value})")

        if self._on_change is not None:
            try:
                self._on_change(old, to, elapsed_ms)
            except Exception:
                self._logger.exception("Turn state on_change callback failed.")
        return to
