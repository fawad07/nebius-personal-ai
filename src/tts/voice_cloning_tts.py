import librosa
import numpy as np
from TTS.api import TTS

# XTTS v2 synthesizes at 24 kHz; used as a fallback when the loaded model
# doesn't expose its native output rate (e.g. under test mocks).
_XTTS_NATIVE_RATE = 24000


class VoiceCloningTTS:
    """
    Real voice cloning TTS using Coqui XTTS v2.
    Converts text + mood → audio bytes using a cloned voice profile.

    The model synthesizes at its own native rate (24 kHz for XTTS v2); the
    output is resampled to ``sample_rate`` so it matches the playback stream
    the AudioOutput was opened with. Without this, a 24 kHz waveform played
    through a 16 kHz stream comes out ~33% too slow and pitched down.
    """

    def __init__(self,
                 model_path: str,
                 voice_profile_path: str,
                 sample_rate: int = 16000):

        self.model_path = model_path
        self.voice_profile_path = voice_profile_path
        self.sample_rate = sample_rate

        self.model = None
        self.voice_profile = None
        self.model_sample_rate = _XTTS_NATIVE_RATE

        self._load_model()
        self._load_voice_profile()

    def _load_model(self):
        """
        Load XTTS model.
        """
        self.model = TTS(self.model_path)
        self.model_sample_rate = self._detect_model_sample_rate()

    def _detect_model_sample_rate(self) -> int:
        """
        Read the model's native output sample rate, falling back to the XTTS
        default when it isn't exposed (e.g. under test mocks).
        """
        try:
            rate = self.model.synthesizer.output_sample_rate
            if isinstance(rate, int) and rate > 0:
                return rate
        except (AttributeError, TypeError):
            pass
        return _XTTS_NATIVE_RATE

    def _load_voice_profile(self):
        """
        Load reference audio for voice cloning.
        """
        self.voice_profile = self.voice_profile_path

    def _mood_to_prosody(self, mood: dict) -> dict:
        """
        Map mood → prosody parameters.
        """
        mood_label = mood.get("mood", "neutral")

        mapping = {
            "neutral": {"speed": 1.0, "temperature": 0.7},
            "calm": {"speed": 0.9, "temperature": 0.6},
            "sad": {"speed": 0.85, "temperature": 0.5},
            "excited": {"speed": 1.2, "temperature": 0.9},
            "angry": {"speed": 1.1, "temperature": 0.95},
            "stressed": {"speed": 1.05, "temperature": 0.8}
        }

        return mapping.get(mood_label, mapping["neutral"])

    def synthesize(self, text: str, mood: dict) -> bytes:
        """
        Generate audio bytes using XTTS voice cloning.
        """
        prosody = self._mood_to_prosody(mood)

        # Generate audio (float32 numpy array) at the model's native rate.
        audio_array = np.asarray(
            self.model.tts(
                text=text,
                speaker_wav=self.voice_profile,
                speed=prosody["speed"],
                temperature=prosody["temperature"]
            ),
            dtype=np.float32,
        )

        audio_array = self._resample_to_output(audio_array)

        # Convert numpy array → bytes
        return audio_array.astype(np.float32).tobytes()

    def _resample_to_output(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Resample the synthesized waveform from the model's native rate to the
        configured playback rate. No-op when they already match or the clip
        is empty.
        """
        if audio_array.size == 0 or self.model_sample_rate == self.sample_rate:
            return audio_array
        return librosa.resample(
            audio_array,
            orig_sr=self.model_sample_rate,
            target_sr=self.sample_rate,
        )

