class AudioError(Exception):
    """Base class for all audio-related errors."""
    pass


class AudioInputError(AudioError):
    """Raised when microphone input fails."""
    pass


class AudioOutputError(AudioError):
    """Raised when speaker output fails."""
    pass


class AudioStreamNotStarted(AudioError):
    """Raised when attempting to read/write before stream is started."""
    pass

