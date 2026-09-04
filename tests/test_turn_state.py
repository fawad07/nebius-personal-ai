from src.orchestration.turn_state import TurnState, TurnStateMachine


def test_transitions_invoke_callback_with_elapsed():
    seen = []
    sm = TurnStateMachine(on_change=lambda old, new, ms: seen.append((old, new)))
    sm.transition(TurnState.LISTENING)
    sm.transition(TurnState.THINKING)
    sm.transition(TurnState.SPEAKING)
    assert sm.state == TurnState.SPEAKING
    assert seen == [
        (TurnState.IDLE, TurnState.LISTENING),
        (TurnState.LISTENING, TurnState.THINKING),
        (TurnState.THINKING, TurnState.SPEAKING),
    ]


def test_callback_error_does_not_propagate():
    def boom(old, new, ms):
        raise RuntimeError("observer blew up")

    sm = TurnStateMachine(on_change=boom)
    # Must not raise despite the failing observer.
    assert sm.transition(TurnState.LISTENING) == TurnState.LISTENING


def test_unexpected_transition_is_soft():
    sm = TurnStateMachine()
    # IDLE -> SPEAKING isn't an expected forward move, but it's allowed (soft).
    assert sm.transition(TurnState.SPEAKING) == TurnState.SPEAKING
