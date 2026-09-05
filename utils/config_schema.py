"""
Typed, validated configuration schema.

Turns the raw YAML dict into typed section objects and — crucially — makes an
**unrecognized key an error** instead of silent dead weight. This is the guard
against "config that lies": a setting that no component reads (e.g. a renamed
or misspelled key, or a feature that was never wired up) fails fast at startup
rather than appearing to work while doing nothing.

Kept dependency-free (dataclasses + a tiny validator) so it doesn't drag in
pydantic; the shape is the contract, and `AppConfig.from_dict` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


class ConfigError(ValueError):
    """Raised when the configuration is structurally invalid."""


def _check_keys(section: str, raw: dict, allowed: set[str]) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"Config section '{section}' must be a mapping, got {type(raw).__name__}.")
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in config section '{section}'. "
            f"Allowed: {sorted(allowed)}."
        )


def _build(cls, section: str, raw: dict | None):
    raw = raw or {}
    allowed = {f.name for f in fields(cls)}
    _check_keys(section, raw, allowed)
    return cls(**raw)


@dataclass
class AudioCfg:
    rate: int = 16000
    channels: int = 1
    chunk_size: int = 4096
    device_index: int | None = None


@dataclass
class STTCfg:
    model_path: str = "medium"
    language: str = "en"
    use_gpu: bool = False


@dataclass
class SERCfg:
    model_path: str = "models/ser/emotion_model.pth"
    sample_rate: int = 16000
    energy_low: float | None = None
    energy_high: float | None = None
    pitch_low: float | None = None
    pitch_high: float | None = None
    pitch_std_high: float | None = None


@dataclass
class SpeakerCfg:
    embed_model_path: str = "models/speaker/ecapa-tdnn"
    threshold: float = 0.75
    profiles_path: str = "data/speaker_profiles/"
    profile_update: bool = True
    profile_update_rate: float = 0.15
    drift_threshold: float = 0.65
    auto_enroll: bool = False


@dataclass
class VADCfg:
    sample_rate: int = 16000
    threshold: float = 0.5


@dataclass
class TTSCfg:
    engine: str = "system"              # "system" (macOS say, default) | "xtts" (Coqui voice cloning)
    voice: str | None = None            # system engine: named OS voice (e.g. "Samantha"); None = default
    base_wpm: int = 180                 # system engine: base speaking rate (words/min)
    model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # xtts only
    voice_profile_path: str = "data/samples/agent_voice.wav"           # xtts only
    sample_rate: int = 16000


@dataclass
class LLMCfg:
    provider: str = "openai"     # "openai" | "ollama" | "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str | None = None  # e.g. http://localhost:11434/v1 for ollama
    max_calls_per_minute: int = 30
    max_tokens_budget: int = 200_000
    max_tokens: int = 1024              # per-response output cap (must fit full structured JSON)
    system_preamble: str = "detailed thinking off"  # Nemotron: disable chain-of-thought for low latency


@dataclass
class ConversationCfg:
    persona_style: str | None = None
    boundaries: str | None = None
    tone_map: dict = field(default_factory=dict)


@dataclass
class MemoryCfg:
    enabled: bool = True
    db_path: str = "data/memory.db"
    history_limit: int = 20


@dataclass
class MetricsCfg:
    latency_tracing: bool = True   # log per-turn listen/think/speak latency
    log_each_turn: bool = True     # per-turn line (off = only the shutdown summary)


@dataclass
class ToolsCfg:
    enabled: bool = True                    # allow the model to call built-in tools
    facts_db_path: str = "data/facts.db"    # persistence for remember/recall
    automation: bool = True                 # graft the offline_assistant automation suite (files/notes/system/etc.)


@dataclass
class VoiceCloneCfg:
    enabled: bool = True                       # expose the synthesize_in_voice tool
    voices_path: str = "data/voices"           # registered, consented reference voices
    recordings_path: str = "data/recordings"   # where generated WAVs are saved


@dataclass
class PrivacyCfg:
    encrypt_embeddings: bool = True         # encrypt voice biometrics at rest
    key_path: str = "data/keyfile"          # generated (chmod 600) if missing
    require_consent: bool = True            # gate biometric storage on consent
    auto_enroll_consent: bool = False       # blanket consent for auto-enrolled speakers


@dataclass
class InputCfg:
    mode: str = "vad"                               # "vad" (always-on) | "push_to_talk"
    ptt_prompt: str = "[press Enter to talk] "


@dataclass
class SessionCfg:
    loop_delay_ms: int = 10
    token_buffer_min_chars: int = 12
    token_buffer_max_chars: int = 80
    silence_duration_ms: int = 800
    max_utterance_ms: int = 15000
    barge_in_speech_chunks: int = 2
    barge_in_enabled: bool | None = None            # None = auto (off for push_to_talk)


_SECTIONS = {
    "audio": AudioCfg,
    "stt": STTCfg,
    "ser": SERCfg,
    "speaker": SpeakerCfg,
    "vad": VADCfg,
    "tts": TTSCfg,
    "llm": LLMCfg,
    "conversation": ConversationCfg,
    "memory": MemoryCfg,
    "metrics": MetricsCfg,
    "tools": ToolsCfg,
    "voice_clone": VoiceCloneCfg,
    "privacy": PrivacyCfg,
    "input": InputCfg,
    "session": SessionCfg,
}


@dataclass
class AppConfig:
    audio: AudioCfg
    stt: STTCfg
    ser: SERCfg
    speaker: SpeakerCfg
    vad: VADCfg
    tts: TTSCfg
    llm: LLMCfg
    conversation: ConversationCfg
    memory: MemoryCfg
    metrics: MetricsCfg
    tools: ToolsCfg
    voice_clone: VoiceCloneCfg
    privacy: PrivacyCfg
    input: InputCfg
    session: SessionCfg

    @classmethod
    def from_dict(cls, raw: dict) -> "AppConfig":
        if not isinstance(raw, dict):
            raise ConfigError(f"Top-level config must be a mapping, got {type(raw).__name__}.")
        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ConfigError(
                f"Unknown top-level config section(s) {sorted(unknown)}. "
                f"Allowed: {sorted(_SECTIONS)}."
            )
        return cls(**{name: _build(model, name, raw.get(name)) for name, model in _SECTIONS.items()})


def validate_config(raw: dict) -> AppConfig:
    """Parse and validate a raw config dict, raising ConfigError on any problem."""
    return AppConfig.from_dict(raw)
