import os
from unittest.mock import MagicMock

import numpy as np

from src.security.crypto import EmbeddingCipher
from src.speaker.auto_enrol import AutoEnroll
from src.speaker.speaker_database import SpeakerDatabase


def _emb(v=1.0):
    return np.full(192, v, dtype=np.float32)


def test_encrypted_db_roundtrip_and_no_plaintext(tmp_path):
    cipher = EmbeddingCipher(b"k" * 32)
    db = SpeakerDatabase(cipher=cipher)
    db.add_profile("fawad", _emb(0.5), "self", consent={"consented": True})
    db.save_to_disk(str(tmp_path))

    # Encrypted layout: .npy.enc present, no plaintext .npy.
    files = os.listdir(tmp_path)
    assert "fawad.npy.enc" in files
    assert "fawad.npy" not in files

    loaded = SpeakerDatabase(cipher=cipher)
    loaded.load_from_disk(str(tmp_path))
    assert np.allclose(loaded.get_profiles()["fawad"], _emb(0.5))
    assert loaded.get_group("fawad") == "self"
    assert loaded.has_consent("fawad")


def test_encrypted_file_is_not_readable_as_numpy(tmp_path):
    db = SpeakerDatabase(cipher=EmbeddingCipher(b"z" * 32))
    db.add_profile("x", _emb(), "self")
    db.save_to_disk(str(tmp_path))
    raw = open(tmp_path / "x.npy.enc", "rb").read()
    assert raw[:4] == b"VAE1"          # our envelope, not a .npy header
    assert not raw.startswith(b"\x93NUMPY")


def test_plaintext_profiles_still_load_after_enabling_encryption(tmp_path):
    # Old plaintext db written without a cipher...
    plain = SpeakerDatabase()
    plain.add_profile("legacy", _emb(0.25), "friend")
    plain.save_to_disk(str(tmp_path))
    assert (tmp_path / "legacy.npy").exists()

    # ...still loads when encryption is later turned on.
    enc = SpeakerDatabase(cipher=EmbeddingCipher(b"k" * 32))
    enc.load_from_disk(str(tmp_path))
    assert np.allclose(enc.get_profiles()["legacy"], _emb(0.25))


def test_auto_enroll_blocked_without_consent(tmp_path):
    db = SpeakerDatabase()
    ae = AutoEnroll(db, MagicMock(), profiles_path=str(tmp_path),
                    require_consent=True, auto_enroll_consent=False)
    assert ae.should_enroll("stranger") is False
    assert ae.enroll_embedding(_emb()) is None
    assert db.get_profiles() == {}


def test_auto_enroll_allowed_with_blanket_consent(tmp_path):
    db = SpeakerDatabase()
    ae = AutoEnroll(db, MagicMock(), profiles_path=str(tmp_path),
                    require_consent=True, auto_enroll_consent=True)
    assert ae.should_enroll("stranger") is True
    name = ae.enroll_embedding(_emb())
    assert name is not None
    assert db.has_consent(name)
