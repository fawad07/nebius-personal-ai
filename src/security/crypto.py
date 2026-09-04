"""
Dependency-free authenticated encryption for data at rest.

Voice embeddings are biometric data; this encrypts them before they touch
disk. With no crypto library available in the pinned environment, this uses
only the standard library, composing well-vetted primitives in a standard way:

  - a CTR-style keystream from HMAC-SHA256 (a secure PRF) XORed with the data,
  - encrypt-then-MAC with a separate HMAC-SHA256 tag for integrity/auth.

This is a legitimate construction, not a home-made primitive, but if you can
install `cryptography`, prefer Fernet -- EmbeddingCipher is deliberately a tiny
interface (encrypt/decrypt bytes) so it can be swapped without touching callers.
"""

import hashlib
import hmac
import os
import struct

_MAGIC = b"VAE1"          # voice-agent encrypted, format v1
_NONCE_LEN = 16
_TAG_LEN = 32
_KDF_SALT = b"voice-agent/embeddings/v1"
_KDF_ROUNDS = 200_000


class EmbeddingCipher:
    """Authenticated encryption of arbitrary bytes with a 32-byte master key."""

    def __init__(self, key: bytes):
        if len(key) < 16:
            raise ValueError("key must be at least 16 bytes")
        # Separate sub-keys for the keystream PRF and the authentication tag.
        self._enc_key = hashlib.sha256(b"enc\x00" + key).digest()
        self._mac_key = hashlib.sha256(b"mac\x00" + key).digest()

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            out += hmac.new(
                self._enc_key, nonce + struct.pack(">Q", counter), hashlib.sha256
            ).digest()
            counter += 1
        return bytes(out[:length])

    def encrypt(self, data: bytes) -> bytes:
        nonce = os.urandom(_NONCE_LEN)
        keystream = self._keystream(nonce, len(data))
        ciphertext = bytes(a ^ b for a, b in zip(data, keystream))
        body = _MAGIC + nonce + ciphertext
        tag = hmac.new(self._mac_key, body, hashlib.sha256).digest()
        return body + tag

    def decrypt(self, token: bytes) -> bytes:
        if len(token) < len(_MAGIC) + _NONCE_LEN + _TAG_LEN or not token.startswith(_MAGIC):
            raise ValueError("invalid ciphertext")
        body, tag = token[:-_TAG_LEN], token[-_TAG_LEN:]
        expected = hmac.new(self._mac_key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("authentication failed: wrong key or tampered data")
        nonce = body[len(_MAGIC):len(_MAGIC) + _NONCE_LEN]
        ciphertext = body[len(_MAGIC) + _NONCE_LEN:]
        keystream = self._keystream(nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, keystream))


def load_or_create_key(key_path: str, env_secret: str | None = None) -> bytes:
    """
    Resolve the master key. Priority:
      1. ``env_secret`` (e.g. from $VOICE_AGENT_SECRET) -> derived via PBKDF2.
      2. an existing key file at ``key_path``.
      3. a freshly generated 32-byte key, written to ``key_path`` with 0600
         perms so only the owner can read it.
    """
    if env_secret:
        return hashlib.pbkdf2_hmac("sha256", env_secret.encode("utf-8"), _KDF_SALT, _KDF_ROUNDS)

    if os.path.isfile(key_path):
        with open(key_path, "rb") as f:
            return f.read()

    key = os.urandom(32)
    parent = os.path.dirname(key_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key)
    os.chmod(key_path, 0o600)
    return key
