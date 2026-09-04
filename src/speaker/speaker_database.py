import io
import json
import os

import numpy as np


class SpeakerDatabase:
    """
    Stores speaker profiles and their embeddings.

    Embeddings are biometric data. When a ``cipher`` is supplied they are
    encrypted at rest (``<name>.npy.enc``); otherwise they are stored as plain
    ``<name>.npy`` files. Both layouts load transparently, so enabling
    encryption doesn't strand existing profiles. Per-speaker consent metadata
    is persisted alongside the group mapping.
    """

    def __init__(self, cipher=None):
        self.profiles = {}   # name -> embedding vector
        self.groups = {}     # name -> group ('self', 'friend', 'family')
        self.consent = {}    # name -> {"consented": bool, "ts": str, "method": str}
        self.cipher = cipher

    def add_profile(self, name: str, embedding, group: str, consent: dict | None = None):
        """Add a speaker profile with its group classification and consent record."""
        self.profiles[name] = embedding
        self.groups[name] = group
        if consent is not None:
            self.consent[name] = consent

    def get_profiles(self):
        return self.profiles

    def get_group(self, name: str) -> str:
        return self.groups.get(name, "stranger")

    def has_consent(self, name: str) -> bool:
        return bool(self.consent.get(name, {}).get("consented"))

    # --- (de)serialization of a single embedding -------------------------------

    def _embedding_to_bytes(self, embedding) -> bytes:
        buf = io.BytesIO()
        np.save(buf, np.asarray(embedding))
        return buf.getvalue()

    def _bytes_to_embedding(self, raw: bytes) -> np.ndarray:
        return np.load(io.BytesIO(raw), allow_pickle=False)

    # --- persistence -----------------------------------------------------------

    def load_from_disk(self, path: str):
        """
        Load profiles from disk. Reads encrypted ``<name>.npy.enc`` files when a
        cipher is configured, and plain ``<name>.npy`` files either way (so a
        pre-encryption database still loads). Groups come from ``groups.json``
        and consent from ``consent.json``.
        """
        if not os.path.isdir(path):
            return

        groups = self._read_json(os.path.join(path, "groups.json"), {})
        self.consent.update(self._read_json(os.path.join(path, "consent.json"), {}))

        for filename in os.listdir(path):
            full = os.path.join(path, filename)
            if filename.endswith(".npy.enc"):
                if self.cipher is None:
                    continue  # can't read encrypted profiles without the key
                name = filename[: -len(".npy.enc")]
                with open(full, "rb") as f:
                    raw = self.cipher.decrypt(f.read())
                self.profiles[name] = self._bytes_to_embedding(raw)
            elif filename.endswith(".npy"):
                name = filename[: -len(".npy")]
                self.profiles[name] = np.load(full)
            else:
                continue
            self.groups[name] = groups.get(name, "stranger")

    def save_to_disk(self, path: str):
        """
        Save profiles as encrypted ``<name>.npy.enc`` (when a cipher is set) or
        plain ``<name>.npy``, plus ``groups.json`` and ``consent.json``. When
        encrypting, a stale plaintext ``<name>.npy`` for the same speaker is
        removed so a cleartext copy isn't left behind.
        """
        os.makedirs(path, exist_ok=True)

        for name, embedding in self.profiles.items():
            raw = self._embedding_to_bytes(embedding)
            if self.cipher is not None:
                with open(os.path.join(path, f"{name}.npy.enc"), "wb") as f:
                    f.write(self.cipher.encrypt(raw))
                stale = os.path.join(path, f"{name}.npy")
                if os.path.exists(stale):
                    os.remove(stale)
            else:
                np.save(os.path.join(path, f"{name}.npy"), np.asarray(embedding))

        self._write_json(os.path.join(path, "groups.json"), self.groups)
        self._write_json(os.path.join(path, "consent.json"), self.consent)

    def save_profile(self, path: str, name: str):
        """
        Persist just one speaker's embedding (encrypted when a cipher is set),
        for frequent single-profile writes like drift updates -- avoids
        rewriting the whole database each turn.
        """
        if name not in self.profiles:
            return
        os.makedirs(path, exist_ok=True)
        raw = self._embedding_to_bytes(self.profiles[name])
        if self.cipher is not None:
            with open(os.path.join(path, f"{name}.npy.enc"), "wb") as f:
                f.write(self.cipher.encrypt(raw))
            stale = os.path.join(path, f"{name}.npy")
            if os.path.exists(stale):
                os.remove(stale)
        else:
            np.save(os.path.join(path, f"{name}.npy"), np.asarray(self.profiles[name]))

    @staticmethod
    def _read_json(path: str, default):
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    @staticmethod
    def _write_json(path: str, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
