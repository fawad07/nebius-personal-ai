import os
import numpy as np
import logging


class ProfileUpdater:
    """
    Automatically updates speaker profiles using new embeddings.
    Runs silently during normal conversation.
    """

    def __init__(
        self,
        speaker_db,
        profiles_path="data/speaker_profiles/",
        update_rate: float = 0.15,   # 15% new embedding, 85% old
        drift_threshold: float = 0.65,  # if similarity < threshold → drift detected
    ):
        self.speaker_db = speaker_db
        self.profiles_path = profiles_path
        os.makedirs(self.profiles_path, exist_ok=True)

        self.logger = logging.getLogger("speaker_profile_update")

        self.update_rate = update_rate
        self.drift_threshold = drift_threshold

    def cosine_similarity(self, a, b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update(self, name: str, new_embedding):
        """
        Update the profile for a known speaker.
        """
        old_embedding = self.speaker_db.profiles.get(name)
        if old_embedding is None:
            return

        similarity = self.cosine_similarity(old_embedding, new_embedding)

        # Detect drift (voice changes, noise, mic differences)
        if similarity < self.drift_threshold:
            self.logger.info(f"Voice drift detected for {name}. Adjusting profile.")

        # Weighted update
        updated = (1 - self.update_rate) * old_embedding + self.update_rate * new_embedding

        # Normalize
        updated = updated / np.linalg.norm(updated)

        # Save back to DB, then persist just this profile (encrypted when the
        # database has a cipher configured).
        self.speaker_db.profiles[name] = updated
        self.speaker_db.save_profile(self.profiles_path, name)

        self.logger.info(f"Updated speaker profile for {name}. Similarity={similarity:.3f}")

