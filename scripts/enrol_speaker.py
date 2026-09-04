import os
import time
import logging
from datetime import datetime, timezone

from utils.config_loader import load_config
from utils.logging_util import setup_logging

from src.audio.input import AudioInput
from src.audio.stream_config import AudioStreamConfig

from src.speaker.embedding_model import SpeakerEmbeddingModel
from src.speaker.speaker_database import SpeakerDatabase
from src.security.crypto import EmbeddingCipher, load_or_create_key


def record_audio_chunk(audio_in, duration_sec: int = 3) -> bytes:
    """
    Record a short audio chunk for enrollment.
    """
    logging.info(f"Recording {duration_sec} seconds of audio...")
    audio_in.start()

    chunks = []
    start = time.time()

    while time.time() - start < duration_sec:
        chunk = audio_in.capture_chunk()
        chunks.append(chunk)

    audio_in.stop()
    logging.info("Recording finished.")

    return b"".join(chunks)


def main():
    setup_logging("config/logging.yaml")
    logger = logging.getLogger("speaker_enroll")

    cfg = load_config("config/settings.yaml")
    audio_cfg = cfg["audio"]
    speaker_cfg = cfg["speaker"]
    privacy_cfg = cfg.get("privacy", {})

    # Init audio input
    stream_cfg = AudioStreamConfig(**audio_cfg)
    audio_in = AudioInput(stream_cfg)

    # Init embedding model + database (encrypted at rest if configured)
    embed_model = SpeakerEmbeddingModel(
        model_path=speaker_cfg["embed_model_path"]
    )
    cipher = None
    if privacy_cfg.get("encrypt_embeddings", True):
        key = load_or_create_key(
            privacy_cfg.get("key_path", "data/keyfile"),
            env_secret=os.getenv("VOICE_AGENT_SECRET"),
        )
        cipher = EmbeddingCipher(key)
    speaker_db = SpeakerDatabase(cipher=cipher)

    profiles_path = speaker_cfg["profiles_path"]
    os.makedirs(profiles_path, exist_ok=True)

    # Load any existing profiles so we extend the database rather than
    # clobbering it (and preserve other speakers' group assignments).
    speaker_db.load_from_disk(profiles_path)

    # Ask user for info
    name = input("Enter speaker name (e.g., Fawad): ").strip()
    group = input("Enter group (self/friend/family): ").strip() or "stranger"

    # Consent: a voiceprint is biometric data; store it only with permission.
    answer = input(
        "Store an encrypted voiceprint for this speaker? Consent required (yes/no): "
    ).strip().lower()
    if answer not in ("y", "yes"):
        print("Consent not given. Nothing was stored.")
        return
    consent = {
        "consented": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "method": "interactive_enrollment",
    }

    logger.info(f"Enrolling speaker: {name} ({group})")

    # Record audio
    audio_bytes = record_audio_chunk(audio_in, duration_sec=5)

    # Compute embedding
    embedding = embed_model.embed(audio_bytes)

    # Add to database
    speaker_db.add_profile(name=name, embedding=embedding, group=group, consent=consent)

    # Persist via the database so both the <name>.npy embedding AND the
    # shared groups.json (name -> group) are written; otherwise the speaker
    # reloads as "stranger" because the group mapping was never saved.
    speaker_db.save_to_disk(profiles_path)
    logger.info(f"Saved profile + group mapping under {profiles_path}")

    print(f"Enrollment complete for {name} ({group}).")


if __name__ == "__main__":
    main()

