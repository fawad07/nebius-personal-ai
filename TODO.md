# voice-agent — Engineering TODO

Working checklist derived from the code review. Order of execution:
**Bad → Ugly → Key structural changes → Phases → New features.**

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[>]` deferred (needs runtime/hardware or is multi-week)

---

## 🟠 The Bad (fix first)

- [x] **B1 — TTS sample-rate bug.** XTTS v2 synthesizes at 24 kHz but the output stream runs at 16 kHz and `VoiceCloningTTS.sample_rate` is never used → playback ~33% too slow / pitched down. Resample TTS output to the playback rate.
- [x] **B2 — Config that lies.** The `conversation:` block (persona/boundaries/tone_map) is ignored; `ConversationManager` hardcodes its own persona. `run.sh --model` exports `LLM_MODEL_OVERRIDE` that nothing reads. Wire both through.
- [x] **B3 — Dead subsystems.** `AutoEnroll` + `ProfileUpdater` are written but never wired into the running agent → no runtime enrollment or profile drift adaptation. Wire into the turn loop (or delete).
- [x] **B4 — Enrollment can't set groups.** `enrol_speaker.py` writes `<name>.npy` but never `groups.json`, so every enrolled speaker reloads as `stranger`. Persist groups.
- [x] **B5 — Partial-transcript hazard.** Untracked `create_task` firing full beam-size-5 Whisper passes that stack up and swallow exceptions. Add single-flight + lighter decode.
- [x] **B6 — Legacy/empty artifacts.** Empty `README.md`, empty `docs/`, empty `tests/test_audio.py` (0 tests); unused `respond()` / `stream_reply()` streaming path. Document + add audio tests; keep or retire legacy methods deliberately.

## 🧪 Found by the actual offline pipeline test (scripts/smoke_test.py)

- [x] **T1 — torch + CTranslate2 segfault/hang.** Loading faster-whisper and torch models in one process crashes (segfault) or deadlocks on macOS depending on load order. Fixed with `OMP_NUM_THREADS=1` in `src/main.py` (+ smoke_test). Verified 6/6 stages green.
- [x] **T2 — Speaker embedding broken (`torch.amp` has no `custom_fwd`).** SpeechBrain 1.1.0 expects torch>=2.4 autocast API; TTS pins torch 2.2.2. Bridged by `utils/torch_compat.py`; requirements note added. Speaker embed now returns a (192,) vector.

## 🔴 The Ugly

- [x] **U1 — Self-barge-in / no AEC.** Barge-in listens on an open mic during playback → hears the agent's own cloned voice → interrupts itself off headphones. Add a mitigation now (require sustained speech + startup grace); real AEC in Phase 1.
- [x] **U2 — No version control.** Not a git repo; `venv/` sits in the tree. `git init` + `.gitignore`.
- [x] **U3 — Voice biometrics with no privacy scaffolding.** Embeddings `np.save`'d plaintext, no consent/retention/`.gitignore`. Add gitignore + docs now; encryption-at-rest in Phase 1.
- [x] **U4 — Exceptions swallowed.** `raise e` no-op in `AudioInput.start()`; broad `except Exception` collapses auth/mic/bug into the calm fallback line. Use the real error taxonomy and log.

## 🏗️ Key structural changes

- [x] **S1 — Provider interfaces (Protocols).** `STTProvider`, `LLMProvider`, `TTSProvider`, `EmotionProvider`, `SpeakerProvider`; `TTSProvider.synthesize` returns `(audio, sample_rate)` so output owns resampling.
- [x] **S2 — Typed, validated config.** Replace `dict.get()` sprawl with a validated schema; an unconsumed field becomes an error, not silent dead weight.
- [x] **S3 — Perception true parallel fan-out.** Start STT ‖ SpeakerID ‖ SER the instant the utterance closes (ID/SER don't need the transcript).
- [x] **S4 — Turn state machine.** `LISTENING → THINKING → SPEAKING → INTERRUPTED` are now first-class, observable transitions (`src/orchestration/turn_state.py`) with an `on_state_change(old, new, elapsed_ms)` hook (UI/metrics/latency-tracing ready), driven from the existing async loop. (A full decoupled EventBus/pub-sub is a further step if multiple independent subscribers are needed.)
- [x] **S5 — Persistent memory store (SQLite).** Conversation turns persist across restarts (`src/memory/conversation_store.py`), loaded back into context on startup; config-gated via `memory:`. (Speaker profiles still persist via SpeakerDatabase; consolidating profiles + consent into the same store is a follow-up.)

---

## 🗺️ Phases

### Phase 0 — Stabilize (days)
- [x] git init + `.gitignore` (venv/, data/, .env, models)  → see U2/U3
- [x] Fix TTS sample-rate  → B1
- [x] Wire AutoEnroll + ProfileUpdater  → B3
- [x] Fix `enrol_speaker.py` groups.json  → B4
- [x] README + `.env.example`  → B6
- [x] Bound partial-transcript concurrency  → B5

### Phase 1 — Correctness & trust (1–2 wks)
- [>] Acoustic echo cancellation (WebRTC APM / reference subtraction)  → real fix for U1
- [x] Typed config (schema + startup validation)  → S2
- [x] Provider Protocols  → S1
- [x] Real error taxonomy across the audio path  → U4
- [x] Consent flow + encryption-at-rest for voice embeddings  → U3. Embeddings encrypted at rest (`src/security/crypto.py`, stdlib HMAC-based AEAD; Fernet-swappable); per-speaker consent recorded; enrollment + auto-enroll gated on consent; `scripts/manage_data.py` for list/forget/wipe (erasure = consent withdrawal). Config: `privacy:`.

### Phase 2 — Capability (2–4 wks)
- [>] Replace SER heuristic with wav2vec2 / emotion2vec (resolve torch pin via TTS-as-service)
- [x] Local-LLM provider (Ollama / OpenAI-compatible) → fully offline LLM leg
- [>] Streaming STT (partial hypotheses) + streaming TTS

### Phase 3 — Product (ongoing)
- [>] Multi-session / multi-user
- [x] Web / WebSocket UI — browser voice client (`src/server/`, `web/index.html`): streams mic audio over WS into the same pipeline via an audio bridge; live turn state + transcript + reply + audio playback; mock mode (no models) validated in-browser, real mode wired. Needs `websockets`.
- [>] Metrics dashboard (per-stage turn latency)
- [>] Wellbeing mode with configurable escalation

---

## ✨ New features backlog
- [>] Acoustic echo cancellation (also U1)
- [>] Local LLM + fully-local mode
- [x] Push-to-talk gating (`input.mode: push_to_talk` → press Enter to talk; auto-disables voice barge-in). Wake-word still deferred (needs a model).
- [x] Function / tool calling — provider-agnostic tool layer (`src/tools/`): the model requests a tool via an `action` in its structured reply; registry dispatches it and the model rephrases the result. Built-ins: get_current_time, remember_fact, recall_facts (SQLite FactStore). Config: `tools:`. (Timers/home-control are further tools to add.)
- [x] Runtime auto-enrollment + drift adaptation (code already exists → B3)
- [x] Voice cloning tool — register a consented reference voice (`VoiceLibrary`) and synthesize any text to a WAV in that voice (`src/tts/voice_cloner.py`, reuses the loaded XTTS). CLI `scripts/clone_voice.py` (add/list/say/remove) + agent tool `synthesize_in_voice` (registered voices only). Consent-gated. Config: `voice_clone:`.
- [>] Emotion trend memory across sessions (opt-in)
- [x] Latency tracing per stage — `LatencyTracer` (`src/orchestration/metrics.py`) consumes the turn-state hook; logs per-turn listen/think/speak/total (+interrupted) and an aggregate count/avg/p50/p95/max summary at shutdown. Config: `metrics:`. (Finer split of "think" into STT vs LLM would need extra states.)
