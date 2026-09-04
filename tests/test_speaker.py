from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.speaker.embedding_model import SpeakerEmbeddingModel
from src.speaker.speaker_database import SpeakerDatabase
from src.speaker.speaker_identifier import SpeakerIdentifier


@patch("src.speaker.embedding_model.EncoderClassifier")
def test_speaker_identifier_empty(mock_encoder_cls):
    mock_encoder_cls.from_hparams.return_value = MagicMock()

    embed_model = SpeakerEmbeddingModel()
    speaker_db = SpeakerDatabase()
    speaker_id = SpeakerIdentifier(embed_model, speaker_db)

    result = speaker_id.identify(b"")
    assert result == "stranger"


@patch("src.speaker.embedding_model.EncoderClassifier")
def test_speaker_embed_returns_vector(mock_encoder_cls):
    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.zeros((1, 1, 192))
    mock_encoder_cls.from_hparams.return_value = mock_model

    embed_model = SpeakerEmbeddingModel()
    audio_bytes = np.zeros(1600, dtype=np.float32).tobytes()
    embedding = embed_model.embed(audio_bytes)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (192,)


def test_speaker_database_save_and_load(tmp_path):
    speaker_db = SpeakerDatabase()
    speaker_db.add_profile("fawad", np.ones(192, dtype=np.float32), "self")

    speaker_db.save_to_disk(str(tmp_path))

    loaded_db = SpeakerDatabase()
    loaded_db.load_from_disk(str(tmp_path))

    assert "fawad" in loaded_db.get_profiles()
    assert loaded_db.get_group("fawad") == "self"
    assert np.allclose(loaded_db.get_profiles()["fawad"], np.ones(192, dtype=np.float32))
