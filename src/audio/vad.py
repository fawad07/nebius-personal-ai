import numpy as np
import torch

from exceptions.model_err import ModelLoadError

# Silero VAD only accepts exactly this many samples per inference call.
_WINDOW_SAMPLES = {16000: 512, 8000: 256}


class VoiceActivityDetector:
    """
    Wraps Silero VAD to detect speech vs. silence in an audio chunk, used by
    SessionManager to accumulate a full utterance before running STT instead
    of treating a single fixed-size audio chunk as a whole turn.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        if sample_rate not in _WINDOW_SAMPLES:
            raise ValueError(f"Silero VAD only supports sample rates {list(_WINDOW_SAMPLES)}")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window_samples = _WINDOW_SAMPLES[sample_rate]
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load Silero VAD: {e}") from e

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Return True if any fixed-size window in the given raw float32 audio
        chunk is classified as speech.
        """
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
        if audio_array.size == 0:
            return False

        n = self.window_samples
        usable_len = (audio_array.size // n) * n
        if usable_len == 0:
            return False

        with torch.no_grad():
            for start in range(0, usable_len, n):
                window = audio_array[start:start + n].copy()
                tensor = torch.from_numpy(window)
                prob = self.model(tensor, self.sample_rate).item()
                if prob >= self.threshold:
                    return True

        return False

    def reset(self):
        """Reset the model's internal streaming state between utterances."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
