from src.orchestration.metrics import LatencyTracer
from src.orchestration.turn_state import TurnState


def _drive_full_turn(tracer, listen, think, speak, interrupted=False):
    """Feed the transitions of one turn; elapsed_ms is time spent in `old`."""
    tracer.on_transition(TurnState.IDLE, TurnState.LISTENING, 0.0)
    tracer.on_transition(TurnState.LISTENING, TurnState.THINKING, listen)
    tracer.on_transition(TurnState.THINKING, TurnState.SPEAKING, think)
    end = TurnState.INTERRUPTED if interrupted else TurnState.LISTENING
    tracer.on_transition(TurnState.SPEAKING, end, speak)


def test_full_turn_records_breakdown():
    t = LatencyTracer(log_each_turn=False)
    _drive_full_turn(t, listen=100.0, think=250.0, speak=400.0)
    assert len(t.turns) == 1
    turn = t.turns[0]
    assert turn["listen"] == 100.0
    assert turn["think"] == 250.0
    assert turn["speak"] == 400.0
    assert turn["total"] == 750.0
    assert turn["interrupted"] is False


def test_interrupted_turn_flagged():
    t = LatencyTracer(log_each_turn=False)
    _drive_full_turn(t, 50.0, 120.0, 30.0, interrupted=True)
    assert t.turns[0]["interrupted"] is True


def test_summary_aggregates_across_turns():
    t = LatencyTracer(log_each_turn=False)
    _drive_full_turn(t, 100.0, 200.0, 300.0)
    _drive_full_turn(t, 200.0, 200.0, 500.0)
    summary = t.summary()
    assert summary["listen"]["count"] == 2
    assert summary["listen"]["avg"] == 150.0
    assert summary["think"]["avg"] == 200.0
    assert summary["speak"]["max"] == 500.0


def test_summary_empty_when_no_turns():
    t = LatencyTracer(log_each_turn=False)
    assert t.summary() == {"listen": None, "think": None, "speak": None}
