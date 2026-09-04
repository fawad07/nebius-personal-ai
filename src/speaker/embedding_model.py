import numpy as np
import torch

from utils.torch_compat import apply_torch_amp_shim

# Must run before SpeechBrain's ECAPA forward uses torch.amp.custom_fwd, which
# doesn't exist on the torch 2.2.2 pinned by TTS. See utils/torch_compat.py.
apply_torch_amp_shim()

from speechbrain.inference.speaker import EncoderClassifier

from exceptions.model_err import ModelLoadError, SpeakerEmbeddingError


class SpeakerEmbeddingModel:
    """
    Speaker embedding backend using SpeechBrain's ECAPA-TDNN
    (speechbrain/spkrec-ecapa-voxceleb), pretrained on VoxCeleb.
    Converts audio bytes -> 192-dim embedding vector.
    """

    def __init__(self, model_path: str = "models/speaker/ecapa-tdnn", sample_rate: int = 16000):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Download (if needed) and load the ECAPA-TDNN speaker embedding model.
        """
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.model_path,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load speaker embedding model: {e}") from e

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes into float32 numpy array.
        """
        return np.frombuffer(audio_bytes, dtype=np.float32)

    def embed(self, audio_bytes: bytes) -> np.ndarray:
        """
        Extract speaker embedding vector from audio.
        """
        try:
            audio_array = self._bytes_to_float32(audio_bytes)
            if audio_array.size == 0:
                return np.zeros(192, dtype=np.float32)

            waveform = torch.from_numpy(audio_array.copy()).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform)

            return embedding.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            raise SpeakerEmbeddingError(f"Failed to extract speaker embedding: {e}") from e
