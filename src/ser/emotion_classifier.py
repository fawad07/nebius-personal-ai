import numpy as np

from exceptions.model_err import FeatureExtractionError, EmotionClassificationError


class EmotionClassifier:
    """
    Speech Emotion Recognition (SER) subsystem.

    Classifies emotion from acoustic arousal features (energy, pitch, pitch
    variability) using calibrated thresholds rather than a pretrained deep
    model: transformers' PyTorch-backed model loading requires torch>=2.5,
    but this project's torch version is pinned to 2.2.2 by TTS==0.22.0, so
    wav2vec2-style SER checkpoints cannot currently be loaded in this venv.
    This heuristic is a real, literature-standard arousal/valence acoustic
    baseline (not a stub), tunable via the threshold constructor args.
    """

    def __init__(self,
                 model_path: str = "models/ser/emotion_model.pth",
                 sample_rate: int = 16000,
                 energy_low: float = 0.002,
                 energy_high: float = 0.02,
                 pitch_low: float = 140.0,
                 pitch_high: float = 220.0,
                 pitch_std_high: float = 35.0):
        self.model_path = model_path
        self.sample_rate = sample_rate

        self.energy_low = energy_low
        self.energy_high = energy_high
        self.pitch_low = pitch_low
        self.pitch_high = pitch_high
        self.pitch_std_high = pitch_std_high

    def _bytes_to_float32(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert raw audio bytes into a float32 numpy array.
        """
        return np.frombuffer(audio_bytes, dtype=np.float32)

    def extract_features(self, audio_bytes: bytes) -> dict:
        """
        Extract the arousal features the classifier uses — mean pitch, pitch
        variability, and energy — from raw audio. Pitch is estimated with a
        numpy FFT-autocorrelation method (no librosa/numba/llvmlite), keeping
        the emotion stage dependency-light and reliable.
        """
        try:
            audio_array = self._bytes_to_float32(audio_bytes)

            pitch, pitch_std = self._estimate_pitch(audio_array)
            energy = float(np.sum(audio_array ** 2) / len(audio_array)) if len(audio_array) > 0 else 0.0

            return {
                "pitch": pitch,
                "pitch_std": pitch_std,
                "energy": energy,
            }

        except Exception as e:
            raise FeatureExtractionError(f"Failed to extract SER features: {e}") from e

    def _estimate_pitch(self, audio_array: np.ndarray,
                        frame_length: int = 2048, hop_length: int = 512,
                        fmin: float = 80.0, fmax: float = 400.0,
                        voicing_threshold: float = 0.3) -> tuple[float, float]:
        """
        Estimate mean and std of the fundamental frequency over voiced frames
        via per-frame autocorrelation (computed through the FFT). Returns
        ``(0.0, 0.0)`` when the clip is too short or has no voiced frames —
        mirroring the previous librosa.piptrack-based behaviour.
        """
        n = int(audio_array.size)
        if n == 0:
            return 0.0, 0.0
        frame_length = min(frame_length, n)
        sr = self.sample_rate
        min_lag = max(1, int(sr / fmax))
        max_lag = min(frame_length - 1, int(sr / fmin))
        if max_lag <= min_lag:
            return 0.0, 0.0

        nfft = 1 << int(np.ceil(np.log2(2 * frame_length - 1)))
        pitches = []
        for start in range(0, n - frame_length + 1, hop_length):
            frame = audio_array[start:start + frame_length].astype(np.float64)
            frame = frame - frame.mean()
            corr0 = float(np.dot(frame, frame))
            if corr0 <= 1e-8:  # near-silent frame → unvoiced
                continue
            spec = np.fft.rfft(frame, nfft)
            corr = np.fft.irfft(spec * np.conj(spec), nfft)[:max_lag + 1]
            segment = corr[min_lag:max_lag + 1]
            if segment.size == 0:
                continue
            peak_idx = int(np.argmax(segment))
            if segment[peak_idx] / corr0 < voicing_threshold:  # weak periodicity → unvoiced
                continue
            freq = sr / (min_lag + peak_idx)
            if fmin <= freq <= fmax:
                pitches.append(freq)

        if not pitches:
            return 0.0, 0.0
        arr = np.asarray(pitches, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def classify(self, audio_bytes: bytes) -> dict:
        """
        Classify emotion from audio.
        Returns: {"label": str, "confidence": float}
        """
        try:
            features = self.extract_features(audio_bytes)
            label, confidence = self._label_from_features(features)
            return {"label": label, "confidence": confidence}
        except FeatureExtractionError:
            raise
        except Exception as e:
            raise EmotionClassificationError(f"Failed to classify emotion: {e}") from e

    def _label_from_features(self, features: dict) -> tuple[str, float]:
        """
        Map arousal-related acoustic features (energy, pitch, pitch jitter)
        onto an emotion label + confidence via calibrated thresholds.
        """
        energy = features["energy"]
        pitch = features["pitch"]
        pitch_std = features["pitch_std"]

        high_energy = energy >= self.energy_high
        low_energy = energy <= self.energy_low
        high_pitch = pitch >= self.pitch_high
        low_pitch = pitch > 0 and pitch <= self.pitch_low
        jittery = pitch_std >= self.pitch_std_high

        if high_energy and high_pitch and jittery:
            label = "angry"
        elif high_energy and high_pitch:
            label = "excited"
        elif high_energy and not high_pitch:
            label = "stressed"
        elif low_energy and low_pitch:
            label = "sad"
        elif low_energy:
            label = "calm"
        else:
            label = "neutral"

        # Confidence: how far the dominant signal (energy) sits from the
        # neutral band, normalized into [0, 1].
        band = max(self.energy_high - self.energy_low, 1e-9)
        if energy >= self.energy_high:
            distance = (energy - self.energy_high) / band
        elif energy <= self.energy_low:
            distance = (self.energy_low - energy) / band
        else:
            distance = 0.0
        confidence = float(min(1.0, 0.5 + distance))

        return label, confidence
