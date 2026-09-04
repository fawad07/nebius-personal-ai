import sounddevice as sd
import numpy as np

from exceptions.audio_err import AudioOutputError, AudioStreamNotStarted


class AudioOutput:
    """
    Handles playback of synthesized audio chunks.
    """

    def __init__(self, config):
        self.config = config
        self.stream = None

    def start(self):
        """
        Initialize and start the speaker output stream.
        """
        try:
            self.stream = sd.OutputStream(
                samplerate=self.config.rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                device=self.config.device_index
            )
            self.stream.start()
        except Exception as e:
            raise AudioOutputError(f"Failed to start speaker output: {e}") from e

    def play(self, audio_bytes: bytes):
        """
        Play raw audio bytes through the speaker.
        """
        if self.stream is None:
            raise AudioStreamNotStarted("AudioOutput stream not started.")

        # Convert bytes → numpy array
        audio_frame = np.frombuffer(audio_bytes, dtype=np.float32)

        self.stream.write(audio_frame)

    def stop(self):
        """
        Stop the speaker stream.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()

