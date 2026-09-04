import logging

from src.orchestration.turn_state import TurnState

# Which turn phase each state-being-left represents. The state machine reports
# elapsed_ms for the state it is *leaving*, so leaving LISTENING = capture time,
# leaving THINKING = perception+LLM time, leaving SPEAKING = playback time.
_PHASE_FOR_STATE = {
    TurnState.LISTENING: "listen",
    TurnState.THINKING: "think",
    TurnState.SPEAKING: "speak",
}
_PHASES = ("listen", "think", "speak")


class LatencyTracer:
    """
    Per-turn latency tracing, fed by the turn state machine's
    on_change(old, new, elapsed_ms) hook.

    A turn's phases arrive progressively (listen -> think -> speak); the turn is
    flushed when SPEAKING is left (whether it finished or was interrupted). Each
    completed turn is logged as a one-line breakdown and folded into running
    aggregates available via :meth:`summary`.

    Coarse by design: "think" bundles STT + speaker-ID + SER + the LLM call,
    since the state machine models them as one THINKING phase. Finer splits
    would need additional states.
    """

    def __init__(self, logger=None, log_each_turn: bool = True, max_turns: int = 1000):
        self.logger = logger or logging.getLogger("metrics")
        self.log_each_turn = log_each_turn
        self.max_turns = max_turns
        self._current: dict = {}
        self.turns: list[dict] = []
        self._samples: dict[str, list[float]] = {p: [] for p in _PHASES}

    def on_transition(self, old: TurnState, new: TurnState, elapsed_ms: float):
        phase = _PHASE_FOR_STATE.get(old)
        if phase is not None:
            self._current[phase] = round(elapsed_ms, 1)
        # A turn completes when we leave SPEAKING (played fully, or barged in).
        if old == TurnState.SPEAKING:
            self._current["interrupted"] = new == TurnState.INTERRUPTED
            self._flush_turn()

    def _flush_turn(self):
        turn = self._current
        self._current = {}
        turn["total"] = round(sum(turn.get(p, 0.0) for p in _PHASES), 1)

        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        for p in _PHASES:
            if p in turn:
                self._samples[p].append(turn[p])

        if self.log_each_turn:
            self.logger.info(
                "Turn latency (ms): listen=%s think=%s speak=%s total=%s%s",
                turn.get("listen", "-"),
                turn.get("think", "-"),
                turn.get("speak", "-"),
                turn["total"],
                " [interrupted]" if turn.get("interrupted") else "",
            )

    @staticmethod
    def _stats(samples: list[float]) -> dict | None:
        if not samples:
            return None
        ordered = sorted(samples)
        n = len(ordered)
        p95_index = min(n - 1, int(round(0.95 * (n - 1))))
        return {
            "count": n,
            "avg": round(sum(ordered) / n, 1),
            "p50": ordered[n // 2],
            "p95": ordered[p95_index],
            "max": ordered[-1],
        }

    def summary(self) -> dict:
        """Aggregate stats per phase across all completed turns."""
        return {p: self._stats(self._samples[p]) for p in _PHASES}

    def log_summary(self):
        summary = self.summary()
        if not any(summary.values()):
            self.logger.info("Latency summary: no completed turns recorded.")
            return
        for phase in _PHASES:
            s = summary[phase]
            if s:
                self.logger.info(
                    "Latency summary [%s] (ms): count=%d avg=%s p50=%s p95=%s max=%s",
                    phase, s["count"], s["avg"], s["p50"], s["p95"], s["max"],
                )
