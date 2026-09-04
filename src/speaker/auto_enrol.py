import logging
import os
from datetime import datetime, timezone

import numpy as np


class AutoEnroll:
    """
    Fully automatic speaker enrollment.
    Detects new speakers, records audio, generates embeddings,
    and saves profiles without user interaction.

    Because auto-enrollment stores biometric data for someone who wasn't asked,
    it is gated by consent policy: when ``require_consent`` is set, nothing is
    stored unless ``auto_enroll_consent`` (a documented blanket consent) is
    also set.
    """

    def __init__(self, speaker_db, embed_model, profiles_path="data/speaker_profiles/",
                 require_consent: bool = False, auto_enroll_consent: bool = False):
        self.speaker_db = speaker_db
        self.embed_model = embed_model
        self.profiles_path = profiles_path
        self.require_consent = require_consent
        self.auto_enroll_consent = auto_enroll_consent

        os.makedirs(self.profiles_path, exist_ok=True)
        self.logger = logging.getLogger("speaker_auto_enroll")

    def should_enroll(self, speaker_group: str) -> bool:
        """
        Decide whether to enroll a speaker automatically. Strangers only, and
        only if consent policy permits storing their biometrics.
        """
        if speaker_group != "stranger":
            return False
        if self.require_consent and not self.auto_enroll_consent:
            self.logger.info("Skipping auto-enroll: consent required but not granted.")
            return False
        return True

    def generate_name(self) -> str:
        """
        Auto-generate a unique speaker name. Counts existing embedding files
        only (``.npy``) so sidecars like groups.json don't skew the index,
        and probes for a free slot to avoid collisions.
        """
        n = sum(1 for f in os.listdir(self.profiles_path) if f.endswith(".npy"))
        candidate = n + 1
        while os.path.exists(os.path.join(self.profiles_path, f"speaker_{candidate}.npy")):
            candidate += 1
        return f"speaker_{candidate}"

    def auto_group(self, embedding) -> str:
        """
        Auto-classify speaker group based on similarity to existing profiles.
        """
        profiles = self.speaker_db.get_profiles()

        if not profiles:
            return "self"

        best_score = 0.0
        best_group = "stranger"

        for name, stored_embedding in profiles.items():
            score = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            if score > best_score:
                best_score = score
                best_group = self.speaker_db.get_group(name)

        return best_group

    def enroll(self, audio_bytes: bytes):
        """
        Fully automatic enrollment pipeline from raw audio.
        """
        embedding = self.embed_model.embed(audio_bytes)
        return self.enroll_embedding(embedding)

    def enroll_embedding(self, embedding) -> str | None:
        """
        Enroll a new speaker from an already-computed embedding, so the caller
        (e.g. the live session, which already embedded the utterance for
        identification) doesn't pay for a second ECAPA pass. Persists via the
        database so encryption + consent metadata are applied consistently.
        Returns the new name, or None if consent policy blocks storage.
        """
        if self.require_consent and not self.auto_enroll_consent:
            self.logger.info("Auto-enroll blocked: consent required but not granted.")
            return None

        name = self.generate_name()
        group = self.auto_group(embedding)
        consent = {
            "consented": True,
            "ts": datetime.now(timezone.utc).isoformat(),
            "method": "auto_enroll_blanket_consent",
        }
        self.speaker_db.add_profile(name=name, embedding=embedding, group=group, consent=consent)
        self.speaker_db.save_to_disk(self.profiles_path)

        self.logger.info(f"Auto-enrolled new speaker: {name} ({group})")
        return name

