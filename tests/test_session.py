import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.orchestration.session_manager import SessionManager


def _make_session(vad_speech_pattern=None, **overrides):
    """
    Build a SessionManager with all dependencies mocked. vad_speech_pattern
    is a list of bools consumed in order by vad.is_speech() across calls.
    """
    audio_in = MagicMock()
    audio_in.config.chunk_size = 4096
    audio_in.config.rate = 16000
    audio_in.capture_chunk.return_value = b"chunk"

    audio_out = MagicMock()
    audio_out.config.chunk_size = 4096

    vad = MagicMock()
    if vad_speech_pattern is not None:
        vad.is_speech.side_effect = vad_speech_pattern

    convo_manager = MagicMock()
    convo_manager.history = []
    convo_manager.respond_with_mood.return_value = {
        "mood": "neutral",
        "intent": "casual_chat",
        "sensitivity": "low",
        "reply": "Hello there.",
    }

    tts = MagicMock()
    tts.synthesize.return_value = b"audio"

    speaker_identifier = MagicMock()
    speaker_identifier.identify.return_value = "self"
    speaker_identifier.identify_detailed.return_value = SimpleNamespace(
        group="self", name="self", score=0.9, embedding=b"emb"
    )

    emotion_classifier = MagicMock()
    emotion_classifier.classify.return_value = {"label": "neutral", "confidence": 0.9}

    kwargs = dict(
        audio_in=audio_in,
        audio_out=audio_out,
        stt_service=MagicMock(),
        speaker_identifier=speaker_identifier,
        emotion_classifier=emotion_classifier,
        conversation_manager=convo_manager,
        tts=tts,
        vad=vad,
        silence_duration_ms=256,  # 1 chunk of silence at 4096/16000
        max_utterance_ms=100000,
    )
    kwargs.update(overrides)
    return SessionManager(**kwargs)


def test_capture_utterance_returns_none_without_speech():
    session = _make_session(vad_speech_pattern=[False])
    result = asyncio.run(session._capture_utterance_async())
    assert result is None


def test_capture_utterance_accumulates_until_silence():
    # speech, speech, silence -> stop after 1 trailing silence chunk
    session = _make_session(vad_speech_pattern=[True, True, False])
    result = asyncio.run(session._capture_utterance_async())
    assert result == b"chunkchunkchunk"
    session.vad.reset.assert_called_once()


def test_handle_turn_skips_when_no_speech():
    session = _make_session(vad_speech_pattern=[False])
    asyncio.run(session.handle_turn())
    session.stt.transcribe.assert_not_called()


def test_handle_turn_full_pipeline_speaks_and_updates_history():
    session = _make_session(vad_speech_pattern=[True, False])
    session.stt.transcribe.return_value = "hello there"

    result = asyncio.run(session.handle_turn())

    assert result is None
    session.speaker_identifier.identify_detailed.assert_called_once()
    session.emotion_classifier.classify.assert_called_once()
    session.convo_manager.respond_with_mood.assert_called_once()
    args, _ = session.convo_manager.respond_with_mood.call_args
    assert args[0] == "hello there"
    assert session.tts.synthesize.called
    assert session.audio_out.play.called
    session.convo_manager.update_history.assert_called_once()
    _, kwargs = session.convo_manager.update_history.call_args
    assert kwargs["user_text"] == "hello there"
    assert kwargs["reply_text"] == "Hello there."


def test_handle_turn_drives_state_machine():
    from src.orchestration.turn_state import TurnState

    transitions = []
    session = _make_session(
        vad_speech_pattern=[True, False],
        on_state_change=lambda old, new, ms: transitions.append(new),
    )
    session.stt.transcribe.return_value = "hello there"

    asyncio.run(session.handle_turn())

    # A full turn should pass through listening -> thinking -> speaking.
    assert TurnState.LISTENING in transitions
    assert TurnState.THINKING in transitions
    assert TurnState.SPEAKING in transitions
    assert transitions.index(TurnState.LISTENING) < transitions.index(TurnState.THINKING)
    assert transitions.index(TurnState.THINKING) < transitions.index(TurnState.SPEAKING)


def test_barge_in_disabled_plays_full_reply_without_listening():
    session = _make_session(vad_speech_pattern=[True, False], barge_in_enabled=False)
    session.stt.transcribe.return_value = "hello there"

    result = asyncio.run(session.handle_turn())

    assert result is None
    assert session.tts.synthesize.called
    assert session.audio_out.play.called
    session.convo_manager.update_history.assert_called_once()


def test_barge_in_cancels_playback_and_returns_leading_chunk():
    session = _make_session(vad_speech_pattern=[True, False])
    session.stt.transcribe.return_value = "hello there"

    # Make TTS synthesis slow so the barge-in listener wins the race.
    def slow_synthesize(*args, **kwargs):
        time.sleep(0.2)
        return b"audio"

    session.tts.synthesize.side_effect = slow_synthesize
    session.convo_manager.respond_with_mood.return_value = {
        "mood": "neutral",
        "intent": "casual_chat",
        "sensitivity": "low",
        "reply": "Hello there, how are you doing today?",
    }

    # Barge-in listener detects speech immediately on its first mic read.
    speech_calls = {"n": 0}

    def is_speech(chunk):
        speech_calls["n"] += 1
        if speech_calls["n"] <= 2:
            return [True, False][speech_calls["n"] - 1]
        return True  # barge-in listener's first read reports speech

    session.vad.is_speech.side_effect = is_speech
    session.audio_in.capture_chunk.return_value = b"bargein"

    result = asyncio.run(session.handle_turn())

    assert result == b"bargein"
    session.convo_manager.update_history.assert_not_called()
