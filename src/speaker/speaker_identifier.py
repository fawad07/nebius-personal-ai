import numpy as np


class SpeakerIdentification:
    """Result of a speaker-identification pass over one utterance."""

    def __init__(self, group: str, name: str | None, score: float, embedding: np.ndarray):
        self.group = group        # 'self' | 'friend' | 'family' | 'stranger'
        self.name = name          # matched profile name, or None if no match
        self.score = score        # cosine similarity of the best match
        self.embedding = embedding  # the utterance embedding (computed once)


class SpeakerIdentifier:
    """
    Identifies speaker by comparing embeddings to known profiles.
    """

    def __init__(self, embed_model, speaker_db, threshold: float = 0.75):
        self.embed_model = embed_model
        self.speaker_db = speaker_db
        self.threshold = threshold

    def _cosine_similarity(self, a, b) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify_detailed(self, audio_bytes: bytes) -> SpeakerIdentification:
        """
        Identify the speaker and return the full result -- group, matched
        profile name, similarity score, and the utterance embedding -- so a
        caller can adapt profiles (drift update / auto-enrollment) without
        recomputing the embedding.
        """
        embedding = self.embed_model.embed(audio_bytes)
        profiles = self.speaker_db.get_profiles()

        best_match = None
        best_score = 0.0
        for name, stored_embedding in profiles.items():
            score = self._cosine_similarity(embedding, stored_embedding)
            if score > best_score:
                best_score = score
                best_match = name

        if best_match is not None and best_score >= self.threshold:
            return SpeakerIdentification(
                group=self.speaker_db.get_group(best_match),
                name=best_match,
                score=float(best_score),
                embedding=embedding,
            )

        return SpeakerIdentification(
            group="stranger", name=None, score=float(best_score), embedding=embedding
        )

    def identify(self, audio_bytes: bytes) -> str:
        """
        Identify speaker group: 'self', 'friend', 'family', or 'stranger'.
        Thin wrapper over :meth:`identify_detailed` for callers that only
        need the group label.
        """
        return self.identify_detailed(audio_bytes).group
