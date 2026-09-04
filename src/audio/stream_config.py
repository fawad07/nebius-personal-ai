class AudioStreamConfig:
    """
    Configuration for audio streaming parameters.
    Shared by AudioInput and AudioOutput.
    """

    def __init__(self,
                 rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 4096,
                 device_index: int | None = None):
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

