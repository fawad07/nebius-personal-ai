import numpy as np
from utils.audio_util import bytes_to_float32, float32_to_bytes, normalize_audio

def test_audio_conversion():
    arr = bytes_to_float32(b"\x00\x00\x00\x00")
    assert isinstance(arr, np.ndarray)

def test_audio_normalization():
    arr = normalize_audio(np.array([0.5, -0.5], dtype=np.float32))
    assert max(abs(arr)) == 1.0

