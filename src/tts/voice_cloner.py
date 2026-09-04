"""
Voice cloning: register a consented reference voice, then synthesize arbitrary
text in that voice and save it as a WAV recording.

Ethics/consent: cloning a voice is dual-use. A voice is only usable here after
it's been *registered* with an explicit consent acknowledgment (see
scripts/clone_voice.py and VoiceLibrary), and the conversational tool can only
use already-registered voices -- never an arbitrary path supplied at runtime.
Generated audio is synthetic and stays local.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone

logger = logging.getLogger("voice_clone")


class VoiceLibrary:
    """
    A registry of named, consented reference voices under ``voices_path``:
    each voice is a copied ``<name>.wav`` plus a ``registry.json`` entry with
    consent metadata. Registration is a deliberate, consent-gated action; the
    conversational tool only ever *reads* from here.
    """

    def __init__(self, voices_path: str = "data/voices"):
        self.voices_path = voices_path
        os.makedirs(self.voices_path, exist_ok=True)
        self._registry_file = os.path.join(voices_path, "registry.json")
        self._registry = self._load()

    def _load(self) -> dict:
        if os.path.isfile(self._registry_file):
            with open(self._registry_file) as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self._registry_file, "w") as f:
            json.dump(self._registry, f, indent=2)

    def add(self, name: str, ref_wav: str, consent: bool, note: str = "") -> None:
        """Register a voice from a reference WAV. Requires explicit consent."""
        if not consent:
            raise PermissionError("Registering a cloned voice requires consent.")
        if not os.path.isfile(ref_wav):
            raise FileNotFoundError(ref_wav)
        dest = os.path.join(self.voices_path, f"{name}.wav")
        shutil.copyfile(ref_wav, dest)
        self._registry[name] = {
            "wav": dest,
            "consent": {
                "consented": True,
                "ts": datetime.now(timezone.utc).isoformat(),
                "note": note,
            },
        }
        self._save()
        logger.info("Registered cloned voice '%s'.", name)

    def get(self, name: str) -> str | None:
        entry = self._registry.get(name)
        if entry and os.path.isfile(entry["wav"]):
            return entry["wav"]
        return None

    def names(self) -> list[str]:
        return list(self._registry)

    def remove(self, name: str) -> bool:
        entry = self._registry.pop(name, None)
        if entry is None:
            return False
        try:
            if os.path.isfile(entry["wav"]):
                os.remove(entry["wav"])
        finally:
            self._save()
        logger.info("Removed cloned voice '%s'.", name)
        return True


class VoiceCloner:
    """
    Synthesizes text in a cloned voice using Coqui XTTS v2. Writes a real WAV
    (native model rate) via the TTS engine's own file writer.

    The underlying model can be injected directly, or resolved lazily via
    ``model_getter`` so the agent can reuse the TTS engine it has already
    loaded instead of loading a second ~2 GB copy.
    """

    def __init__(self, model=None, model_getter=None, language: str = "en"):
        self._model = model
        self._model_getter = model_getter
        self.language = language

    @classmethod
    def load(cls, model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2",
             language: str = "en") -> "VoiceCloner":
        from TTS.api import TTS  # deferred: heavy import
        return cls(model=TTS(model_path), language=language)

    def _resolve_model(self):
        if self._model is None:
            if self._model_getter is None:
                raise RuntimeError("No TTS model available for cloning.")
            self._model = self._model_getter()
        return self._model

    def clone_to_file(self, text: str, speaker_wav: str, out_path: str,
                      language: str | None = None) -> str:
        if not text.strip():
            raise ValueError("text is empty")
        if not os.path.isfile(speaker_wav):
            raise FileNotFoundError(speaker_wav)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._resolve_model().tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            file_path=out_path,
            language=language or self.language,
        )
        logger.info("Synthesized %d chars -> %s", len(text), out_path)
        return out_path
