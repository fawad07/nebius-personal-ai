class TokenBuffer:
    """
    Buffers streaming LLM tokens and emits clean text chunks
    based on punctuation, sentence boundaries, and length heuristics.
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 80):
        self.tokens = []
        self.last_emitted = ""

        # Tunable parameters
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.sentence_endings = {".", "!", "?"}
        self.soft_endings = {",", ";", ":"}

    def add(self, token: str):
        self.tokens.append(token)

    def get_text(self):
        return "".join(self.tokens)

    def should_emit(self) -> bool:
        text = self.get_text()

        # Too short → don't speak yet
        if len(text) < self.min_chars:
            return False

        # Sentence boundary → speak immediately
        if any(text.endswith(p) for p in self.sentence_endings):
            return True

        # Soft boundary + enough length
        if any(text.endswith(p) for p in self.soft_endings) and len(text) > self.min_chars * 2:
            return True

        # Overflow → force emit
        if len(text) >= self.max_chars:
            return True

        return False

    def emit(self) -> str:
        text = self.get_text()
        self.tokens = []
        self.last_emitted = text
        return text


def split_into_speech_segments(text: str, min_chars: int = 12, max_chars: int = 80) -> list[str]:
    """
    Split a complete reply string into speech-friendly segments using the
    same sentence/length heuristics TokenBuffer uses for live token
    streaming. Used when the full reply text is already available (e.g. a
    single structured LLM call) but we still want progressive TTS playback
    -- and barge-in interruption points -- instead of synthesizing the
    whole reply as one block.
    """
    buffer = TokenBuffer(min_chars=min_chars, max_chars=max_chars)
    segments = []
    for ch in text:
        buffer.add(ch)
        if buffer.should_emit():
            segments.append(buffer.emit())
    if buffer.get_text():
        segments.append(buffer.emit())
    return segments
