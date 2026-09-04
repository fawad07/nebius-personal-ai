import asyncio
import logging
import sys

logger = logging.getLogger("session_manager")

_QUIT_WORDS = {"q", "quit", "exit"}


class AlwaysOnGate:
    """
    Default gate: the agent is always listening. Turn start is never blocked,
    so VAD-based endpointing drives everything (the original behavior).
    """

    async def wait(self) -> bool:
        return True


class PushToTalkGate:
    """
    Push-to-talk: the agent waits for the user to press Enter before it starts
    listening for the next utterance. Typing 'q' (or quit/exit) ends the
    session; EOF on stdin (Ctrl-D) does the same.

    ``reader`` is injectable so this is testable without a real terminal; it
    must return one input line (or "" at EOF), mirroring ``stdin.readline``.
    """

    def __init__(self, prompt: str = "[press Enter to talk] ", reader=None):
        self.prompt = prompt
        self._reader = reader or self._default_reader

    @staticmethod
    def _default_reader() -> str:
        sys.stdout.write("")  # ensure prompt (written separately) is flushed
        return sys.stdin.readline()

    async def wait(self) -> bool:
        # Print prompt on the main thread, block for a line on a worker thread
        # so the event loop stays responsive.
        sys.stdout.write(self.prompt)
        sys.stdout.flush()
        line = await asyncio.to_thread(self._reader)
        if line == "":  # EOF (Ctrl-D)
            logger.info("Input stream closed; ending session.")
            return False
        if line.strip().lower() in _QUIT_WORDS:
            logger.info("Quit requested; ending session.")
            return False
        return True


def make_input_gate(mode: str = "vad", ptt_prompt: str = "[press Enter to talk] "):
    """Build an input gate from a config mode ('vad' or 'push_to_talk')."""
    normalized = (mode or "vad").lower()
    if normalized in ("push_to_talk", "ptt"):
        return PushToTalkGate(prompt=ptt_prompt)
    if normalized in ("vad", "always_on", "always-on"):
        return AlwaysOnGate()
    raise ValueError(f"Unknown input mode: {mode!r} (expected 'vad' or 'push_to_talk')")
