import logging
import os

from utils.config_loader import load_config
from utils.config_schema import validate_config
from utils.logging_util import setup_logging

from src.llm.llm_client import LLMClient
from src.conversation.conversation_manager import ConversationManager
from src.memory.conversation_store import ConversationStore
from src.memory.fact_store import FactStore
from src.tools.builtins import build_default_registry
from src.tools.offline_automation import extend_registry_with_automation
# NOTE: XTTS voice cloning (src.tts.voice_cloner / voice_cloning_tts) pulls in
# Coqui-TTS + numba + llvmlite. It is imported lazily in the `xtts` branches
# below so the default `system` TTS engine needs none of that toolchain.

from src.audio.stream_config import AudioStreamConfig
from src.audio.input import AudioInput
from src.audio.output import AudioOutput

from src.stt.stt_service import STTService

from src.speaker.embedding_model import SpeakerEmbeddingModel
from src.speaker.speaker_database import SpeakerDatabase
from src.speaker.speaker_identifier import SpeakerIdentifier
from src.speaker.profile_updater import ProfileUpdater
from src.speaker.auto_enrol import AutoEnroll
from src.security.crypto import EmbeddingCipher, load_or_create_key

from src.ser.emotion_classifier import EmotionClassifier
from src.audio.vad import VoiceActivityDetector

from src.orchestration.session_manager import SessionManager
from src.orchestration.metrics import LatencyTracer
from src.orchestration.input_gate import make_input_gate


class Agent:
    """
    Top-level entry point: loads config, wires up every pipeline component,
    and hands them to a SessionManager to run the conversation loop.
    """

    def __init__(self, config_path: str = "config/settings.yaml",
                 audio_in=None, audio_out=None,
                 on_state_change=None, on_partial_transcript=None, on_reply=None):
        setup_logging("config/logging.yaml")
        self.logger = logging.getLogger("agent")
        self.config = load_config(config_path)
        # Fail fast on a malformed or lying config (unknown/misspelled keys)
        # before we start loading multi-hundred-MB models.
        validate_config(self.config)

        # Audio source/sink can be overridden (e.g. a browser bridge instead of
        # the local mic/speakers); default to the local sounddevice streams.
        self._audio_in_override = audio_in
        self._audio_out_override = audio_out
        # Optional external observers (e.g. a browser UI).
        self._ext_on_state_change = on_state_change
        self._ext_on_partial_transcript = on_partial_transcript
        self._ext_on_reply = on_reply

        self._init_llm()
        self._init_conversation()
        self._init_audio()
        self._init_stt()
        self._init_speaker()
        self._init_ser()
        self._init_vad()
        self._init_tts()
        self._init_session_manager()

    _OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"

    @staticmethod
    def _is_local_url(url: str | None) -> bool:
        """True for a loopback endpoint (no API key needed), False otherwise.

        A null base_url means hosted OpenAI, which is not local.
        """
        if not url:
            return False
        return any(host in url for host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))

    def _init_llm(self):
        llm_cfg = self.config.get("llm", {})
        provider = (llm_cfg.get("provider") or "openai").lower()
        # An explicit `run.sh --model <name>` (exported as LLM_MODEL_OVERRIDE)
        # takes precedence over the configured model.
        model_name = os.getenv("LLM_MODEL_OVERRIDE") or llm_cfg.get("model", "gpt-4o-mini")

        base_url = llm_cfg.get("base_url")
        if provider == "ollama" and not base_url:
            base_url = self._OLLAMA_DEFAULT_URL

        # A *remote* OpenAI-compatible endpoint (e.g. Nebius Token Factory) still
        # needs an API key; only a truly local server (Ollama / LM Studio on
        # localhost) is keyless. Distinguish by whether the endpoint is local.
        is_local_endpoint = provider == "ollama" or self._is_local_url(base_url)
        require_key = not is_local_endpoint
        # Nebius issues its own key; accept it under NEBIUS_API_KEY and fall back
        # to OPENAI_API_KEY so a hosted-OpenAI setup still works unchanged.
        api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")

        self.llm_client = LLMClient(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            require_key=require_key,
            provider=provider,
            max_calls_per_minute=llm_cfg.get("max_calls_per_minute", 30),
            max_tokens_total=llm_cfg.get("max_tokens_budget", 200_000),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            # For Nemotron reasoning models this disables chain-of-thought,
            # cutting per-turn latency from ~50s to ~1s (see config note).
            system_preamble=llm_cfg.get("system_preamble", "detailed thinking off"),
        )
        # Fail fast: for hosted OpenAI this validates the key; for a local
        # endpoint it confirms the server is reachable -- rather than surfacing
        # the error deep into a live conversation on the first real LLM call.
        self.llm_client.validate_api_key()
        target = base_url or "OpenAI"
        self.logger.info(
            f"LLM client initialized: provider={provider} model={model_name} endpoint={target}."
        )
        # Warm the (serverless) model during startup so its cold-start doesn't
        # land on the user's first spoken turn. Non-blocking; on by default.
        if llm_cfg.get("warmup", True):
            self.llm_client.warmup_async()
            self.logger.info("Warming up the model in the background (cold-start mitigation).")

    def _init_conversation(self):
        mem_cfg = self.config.get("memory", {})
        self.conversation_store = None
        if mem_cfg.get("enabled", True):
            self.conversation_store = ConversationStore(
                db_path=mem_cfg.get("db_path", "data/memory.db")
            )
            self.logger.info("Conversation memory (SQLite) enabled.")

        tools_cfg = self.config.get("tools", {})
        self.fact_store = None
        tools = None
        if tools_cfg.get("enabled", True):
            self.fact_store = FactStore(db_path=tools_cfg.get("facts_db_path", "data/facts.db"))

            # Voice-cloning tool (registered/consented voices only). It reuses
            # the XTTS engine's model, so it's only available when tts.engine is
            # "xtts"; with the default system voice there is no clone model and
            # the tool (and its heavy Coqui/numba imports) are skipped.
            vc_cfg = self.config.get("voice_clone", {})
            cloner = voice_library = None
            tts_engine = (self.config.get("tts", {}).get("engine") or "system").lower()
            if vc_cfg.get("enabled", True) and tts_engine == "xtts":
                from src.tts.voice_cloner import VoiceCloner, VoiceLibrary
                voice_library = VoiceLibrary(vc_cfg.get("voices_path", "data/voices"))
                cloner = VoiceCloner(model_getter=lambda: self.tts.model)

            tools = build_default_registry(
                fact_store=self.fact_store,
                cloner=cloner,
                voice_library=voice_library,
                recordings_path=vc_cfg.get("recordings_path", "data/recordings"),
            )

            # Graft the automation suite carried over from offline_assistant
            # (files, notes, system info, activity report, code check) onto the
            # same registry. On by default; disable with tools.automation=false.
            if tools_cfg.get("automation", True):
                extend_registry_with_automation(tools)

            self.logger.info(f"Tools enabled: {tools.names()}")

        self.conversation_manager = ConversationManager(
            self.llm_client,
            persona_config=self.config.get("conversation", {}),
            store=self.conversation_store,
            history_limit=mem_cfg.get("history_limit", 20),
            tools=tools,
        )
        self.logger.info("Conversation manager initialized.")

    def _init_audio(self):
        audio_cfg = self.config.get("audio", {})
        stream_cfg = AudioStreamConfig(**audio_cfg)
        self.audio_in = self._audio_in_override or AudioInput(stream_cfg)
        self.audio_out = self._audio_out_override or AudioOutput(stream_cfg)
        kind = "browser bridge" if self._audio_in_override else "local devices"
        self.logger.info(f"Audio input/output initialized ({kind}).")

    def _init_stt(self):
        stt_cfg = self.config.get("stt", {})
        self.stt_service = STTService(**stt_cfg)
        self.logger.info("STT service initialized.")

    def _init_speaker(self):
        speaker_cfg = self.config.get("speaker", {})
        privacy_cfg = self.config.get("privacy", {})
        self.embed_model = SpeakerEmbeddingModel(
            model_path=speaker_cfg.get("embed_model_path")
        )

        # Encrypt voice biometrics at rest when enabled.
        cipher = None
        if privacy_cfg.get("encrypt_embeddings", True):
            key = load_or_create_key(
                privacy_cfg.get("key_path", "data/keyfile"),
                env_secret=os.getenv("VOICE_AGENT_SECRET"),
            )
            cipher = EmbeddingCipher(key)
            self.logger.info("Voice biometric encryption at rest enabled.")

        self.speaker_db = SpeakerDatabase(cipher=cipher)
        self.speaker_db.load_from_disk(speaker_cfg.get("profiles_path", "data/speaker_profiles/"))
        self.speaker_identifier = SpeakerIdentifier(
            self.embed_model,
            self.speaker_db,
            threshold=speaker_cfg.get("threshold", 0.75),
        )

        profiles_path = speaker_cfg.get("profiles_path", "data/speaker_profiles/")
        # Drift adaptation for known speakers (safe: bounded set of names).
        self.profile_updater = ProfileUpdater(
            self.speaker_db,
            profiles_path=profiles_path,
            update_rate=speaker_cfg.get("profile_update_rate", 0.15),
            drift_threshold=speaker_cfg.get("drift_threshold", 0.65),
        ) if speaker_cfg.get("profile_update", True) else None
        # Auto-enrollment of strangers is opt-in (off by default) since it
        # creates a new profile per unrecognized speaker.
        self.auto_enroll = AutoEnroll(
            self.speaker_db,
            self.embed_model,
            profiles_path=profiles_path,
            require_consent=privacy_cfg.get("require_consent", True),
            auto_enroll_consent=privacy_cfg.get("auto_enroll_consent", False),
        ) if speaker_cfg.get("auto_enroll", False) else None

        self.logger.info("Speaker identification initialized.")

    def _init_ser(self):
        ser_cfg = self.config.get("ser", {})
        self.emotion_classifier = EmotionClassifier(**ser_cfg)
        self.logger.info("Emotion classifier initialized.")

    def _init_vad(self):
        vad_cfg = self.config.get("vad", {})
        self.vad = VoiceActivityDetector(
            sample_rate=vad_cfg.get("sample_rate", 16000),
            threshold=vad_cfg.get("threshold", 0.5),
        )
        self.logger.info("Voice activity detector initialized.")

    def _init_tts(self):
        tts_cfg = dict(self.config.get("tts", {}))
        # Default to the dependency-light system voice; `xtts` opts into the
        # heavy Coqui voice-cloning engine (imported lazily so it isn't required
        # unless selected).
        self.tts_engine = (tts_cfg.pop("engine", None) or "system").lower()
        if self.tts_engine == "xtts":
            from src.tts.voice_cloning_tts import VoiceCloningTTS
            self.tts = VoiceCloningTTS(**tts_cfg)
            self.logger.info("TTS engine initialized: XTTS voice cloning.")
        else:
            from src.tts.system_tts import SystemTTS
            self.tts = SystemTTS(**tts_cfg)
            self.logger.info("TTS engine initialized: system voice (say).")

    def _init_session_manager(self):
        session_cfg = self.config.get("session", {})

        metrics_cfg = self.config.get("metrics", {})
        self.latency_tracer = None
        tracer_cb = None
        if metrics_cfg.get("latency_tracing", True):
            self.latency_tracer = LatencyTracer(
                log_each_turn=metrics_cfg.get("log_each_turn", True)
            )
            tracer_cb = self.latency_tracer.on_transition
            self.logger.info("Latency tracing enabled.")

        # Fan the state-change signal to both the latency tracer and any
        # external observer (e.g. a browser UI).
        ext_state = self._ext_on_state_change
        if tracer_cb and ext_state:
            def on_state_change(old, new, ms):
                tracer_cb(old, new, ms)
                ext_state(old, new, ms)
        else:
            on_state_change = tracer_cb or ext_state

        input_cfg = self.config.get("input", {})
        input_mode = input_cfg.get("mode", "vad")
        input_gate = make_input_gate(
            mode=input_mode,
            ptt_prompt=input_cfg.get("ptt_prompt", "[press Enter to talk] "),
        )
        # Barge-in defaults on for always-on VAD, off for push-to-talk, unless
        # explicitly forced in config.
        barge_in_enabled = session_cfg.get("barge_in_enabled")
        if barge_in_enabled is None:
            barge_in_enabled = input_mode.lower() not in ("push_to_talk", "ptt")
        self.logger.info(f"Input mode: {input_mode} (barge-in {'on' if barge_in_enabled else 'off'}).")

        self.session_manager = SessionManager(
            audio_in=self.audio_in,
            audio_out=self.audio_out,
            stt_service=self.stt_service,
            speaker_identifier=self.speaker_identifier,
            emotion_classifier=self.emotion_classifier,
            conversation_manager=self.conversation_manager,
            tts=self.tts,
            vad=self.vad,
            loop_delay_ms=session_cfg.get("loop_delay_ms", 10),
            token_buffer_min_chars=session_cfg.get("token_buffer_min_chars", 12),
            token_buffer_max_chars=session_cfg.get("token_buffer_max_chars", 80),
            silence_duration_ms=session_cfg.get("silence_duration_ms", 800),
            max_utterance_ms=session_cfg.get("max_utterance_ms", 15000),
            barge_in_speech_chunks=session_cfg.get("barge_in_speech_chunks", 2),
            barge_in_enabled=barge_in_enabled,
            input_gate=input_gate,
            on_state_change=on_state_change,
            on_partial_transcript=self._ext_on_partial_transcript,
            on_reply=self._ext_on_reply,
            profile_updater=self.profile_updater,
            auto_enroll=self.auto_enroll,
        )
        self.logger.info("Session manager initialized.")

    def run(self):
        """Start the conversation loop."""
        self.logger.info("Agent starting.")
        try:
            self.session_manager.run_forever()
        finally:
            speaker_cfg = self.config.get("speaker", {})
            self.speaker_db.save_to_disk(speaker_cfg.get("profiles_path", "data/speaker_profiles/"))
            if self.conversation_store is not None:
                self.conversation_store.close()
            if self.fact_store is not None:
                self.fact_store.close()
            if self.latency_tracer is not None:
                self.latency_tracer.log_summary()
