import numpy as np

def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw audio bytes into a float32 numpy array.
    """
    return np.frombuffer(audio_bytes, dtype=np.float32)


def float32_to_bytes(array: np.ndarray) -> bytes:
    """
    Convert float32 numpy array into raw audio bytes.
    """
    return array.astype(np.float32).tobytes()


def normalize_audio(array: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to prevent clipping.
    """
    if array.size == 0:
        return array
    max_val = np.max(np.abs(array))
    return array / max_val if max_val > 0 else array

