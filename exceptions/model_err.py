class ModelError(Exception):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass


class FeatureExtractionError(ModelError):
    """Raised when SER feature extraction fails."""
    pass


class TranscriptionError(ModelError):
    """Raised when STT transcription fails."""
    pass


class EmotionClassificationError(ModelError):
    """Raised when SER classification fails."""
    pass


class SpeakerEmbeddingError(ModelError):
    """Raised when speaker embedding extraction fails."""
    pass


class SpeakerIdentificationError(ModelError):
    """Raised when speaker identification fails."""
    pass


class TTSGenerationError(ModelError):
    """Raised when TTS audio generation fails."""
    pass

