import sounddevice as sd
import numpy as np

from exceptions.audio_err import AudioInputError, AudioStreamNotStarted


class AudioInput:
    """
    Handles microphone audio capture in fixed-size chunks.
    """

    def __init__(self, config):
        self.config = config
        self.stream = None

    def start(self):
        """
        Initialize and start the microphone stream.
        """
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                device=self.config.device_index
            )
            self.stream.start()
        except Exception as e:
            raise AudioInputError(f"Failed to start microphone input: {e}") from e

    def capture_chunk(self) -> bytes:
        """
        Capture a chunk of audio and return raw bytes.
        """
        if self.stream is None:
            raise AudioStreamNotStarted("AudioInput stream not started.")

        audio_frame, overflow = self.stream.read(self.config.chunk_size)

        # Convert float32 numpy array → bytes
        return audio_frame.astype(np.float32).tobytes()

    def stop(self):
        """
        Stop the microphone stream.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()

