import numpy as np
from faster_whisper import WhisperModel

from exceptions.model_err import TranscriptionError


class STTService:
    """
    Real Speech-to-Text service using faster-whisper.
    Converts raw audio bytes into text, streaming-friendly.
    """

    def __init__(self,
                 model_path: str,
                 language: str = "en",
                 use_gpu: bool = True):

        self.model_path = model_path
        self.language = language
        self.use_gpu = use_gpu

        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load faster-whisper model.
        """
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self.model = WhisperModel(self.model_path, device=device)
        except Exception as e:
            raise TranscriptionError(f"Failed to load STT model: {e}")

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes into float32 numpy array.
        """
        return np.frombuffer(audio_bytes, dtype=np.float32)

    def transcribe(self, audio_bytes: bytes, beam_size: int = 5) -> str:
        """
        Transcribe a chunk of audio into text (streaming-friendly).

        ``beam_size`` defaults to 5 for the final, accurate transcription; a
        lower value (e.g. 1) is much cheaper and is used for best-effort
        partial transcripts emitted mid-utterance.
        """
        try:
            audio_array = self._bytes_to_float32(audio_bytes)

            segments, _ = self.model.transcribe(
                audio_array,
                language=self.language,
                beam_size=beam_size
            )

            text = "".join(seg.text for seg in segments).strip()
            return text

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def transcribe_full(self, audio_bytes: bytes) -> str:
        """
        Full-audio transcription for non-streaming use cases.
        """
        return self.transcribe(audio_bytes)

