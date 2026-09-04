import os

import pytest

from src.tools.builtins import build_default_registry
from src.tts.voice_cloner import VoiceLibrary


def _ref_wav(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFF....WAVEfake")  # contents don't matter for the library
    return str(p)


def test_library_requires_consent(tmp_path):
    lib = VoiceLibrary(str(tmp_path / "voices"))
    with pytest.raises(PermissionError):
        lib.add("alice", _ref_wav(tmp_path), consent=False)
    assert lib.names() == []


def test_library_add_get_remove(tmp_path):
    lib = VoiceLibrary(str(tmp_path / "voices"))
    lib.add("alice", _ref_wav(tmp_path), consent=True, note="my own voice")
    assert lib.names() == ["alice"]
    assert lib.get("alice") and os.path.isfile(lib.get("alice"))
    # persists across reopen
    lib2 = VoiceLibrary(str(tmp_path / "voices"))
    assert lib2.get("alice") is not None
    assert lib2.remove("alice") is True
    assert lib2.get("alice") is None


def test_missing_reference_rejected(tmp_path):
    lib = VoiceLibrary(str(tmp_path / "voices"))
    with pytest.raises(FileNotFoundError):
        lib.add("x", str(tmp_path / "nope.wav"), consent=True)


class FakeCloner:
    def __init__(self):
        self.calls = []

    def clone_to_file(self, text, speaker_wav, out_path, language=None):
        self.calls.append((text, speaker_wav, out_path))
        return out_path


def test_tool_synthesizes_with_registered_voice(tmp_path):
    lib = VoiceLibrary(str(tmp_path / "voices"))
    lib.add("me", _ref_wav(tmp_path), consent=True)
    fake = FakeCloner()
    reg = build_default_registry(cloner=fake, voice_library=lib,
                                 recordings_path=str(tmp_path / "rec"))
    assert "synthesize_in_voice" in reg.names()
    out = reg.execute("synthesize_in_voice", {"text": "hello world", "voice": "me"})
    assert "Saved a recording" in out and ".wav" in out
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "hello world"


def test_tool_rejects_unregistered_voice(tmp_path):
    lib = VoiceLibrary(str(tmp_path / "voices"))
    reg = build_default_registry(cloner=FakeCloner(), voice_library=lib,
                                 recordings_path=str(tmp_path / "rec"))
    out = reg.execute("synthesize_in_voice", {"text": "hi", "voice": "stranger"})
    assert "unknown voice" in out


def test_tool_absent_without_cloner():
    reg = build_default_registry()  # no cloner/library
    assert "synthesize_in_voice" not in reg.names()
