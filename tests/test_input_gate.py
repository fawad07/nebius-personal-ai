import asyncio

import pytest

from src.orchestration.input_gate import (
    AlwaysOnGate,
    PushToTalkGate,
    make_input_gate,
)


def test_always_on_gate_never_blocks():
    assert asyncio.run(AlwaysOnGate().wait()) is True


def test_ptt_gate_enter_activates():
    gate = PushToTalkGate(reader=lambda: "\n")
    assert asyncio.run(gate.wait()) is True


def test_ptt_gate_eof_ends_session():
    gate = PushToTalkGate(reader=lambda: "")  # EOF
    assert asyncio.run(gate.wait()) is False


def test_ptt_gate_quit_word_ends_session():
    for word in ("q\n", "quit\n", "EXIT\n"):
        gate = PushToTalkGate(reader=lambda w=word: w)
        assert asyncio.run(gate.wait()) is False


def test_make_input_gate_selects_type():
    assert isinstance(make_input_gate("vad"), AlwaysOnGate)
    assert isinstance(make_input_gate("push_to_talk"), PushToTalkGate)


def test_make_input_gate_rejects_unknown():
    with pytest.raises(ValueError):
        make_input_gate("telepathy")
