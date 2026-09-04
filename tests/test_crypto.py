import os
import stat

import pytest

from src.security.crypto import EmbeddingCipher, load_or_create_key


def test_encrypt_decrypt_roundtrip():
    cipher = EmbeddingCipher(b"0" * 32)
    data = b"voice embedding bytes \x00\x01\x02 ..."
    token = cipher.encrypt(data)
    assert token != data
    assert cipher.decrypt(token) == data


def test_ciphertext_is_nondeterministic():
    cipher = EmbeddingCipher(b"k" * 32)
    a = cipher.encrypt(b"same input")
    b = cipher.encrypt(b"same input")
    assert a != b  # random nonce per encryption
    assert cipher.decrypt(a) == cipher.decrypt(b) == b"same input"


def test_tamper_is_detected():
    cipher = EmbeddingCipher(b"k" * 32)
    token = bytearray(cipher.encrypt(b"important"))
    token[-1] ^= 0x01  # flip a bit in the tag
    with pytest.raises(ValueError):
        cipher.decrypt(bytes(token))


def test_wrong_key_fails():
    token = EmbeddingCipher(b"a" * 32).encrypt(b"secret")
    with pytest.raises(ValueError):
        EmbeddingCipher(b"b" * 32).decrypt(token)


def test_env_secret_is_deterministic():
    k1 = load_or_create_key("unused", env_secret="hunter2")
    k2 = load_or_create_key("unused", env_secret="hunter2")
    assert k1 == k2 and len(k1) == 32


def test_keyfile_created_with_owner_only_perms(tmp_path):
    key_path = str(tmp_path / "sub" / "keyfile")
    key = load_or_create_key(key_path)
    assert len(key) == 32
    assert os.path.isfile(key_path)
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600
    # A second call returns the same persisted key.
    assert load_or_create_key(key_path) == key
