# Submission — Nebius Personal AI

Copy each section into the matching field on the Devpost submission form.
Track: **Personal AI**.

---

## Project name
Nebius Personal AI

## Tagline
A private, voice-driven personal assistant that reasons with NVIDIA Nemotron on Nebius, remembers you, and gets things done.

---

## What it is (project description)

**Nebius Personal AI** is an always-on, private personal assistant. You talk to it;
it recognizes *how* you sound, reasons with an **NVIDIA Nemotron open model served
on Nebius Token Factory**, acts through a set of tools you control, remembers you
across sessions, and replies in a mood-matched voice.

It runs as two surfaces from one core:

- **Hosted web app** — a public URL where anyone can type or speak (browser voice
  via the Web Speech API). This is the reliable, judge-testable demo: reasoning +
  tools + memory + web search, no install, nothing to break.
- **Local voice pipeline** — the full experience shown in the video: everything
  except the language model runs **on-device** — speech-to-text (Whisper), speaker
  identification (ECAPA-TDNN), acoustic emotion detection, barge-in (you can
  interrupt it mid-sentence), and an optional cloned voice. Only the reasoning
  prompt leaves the machine, going to the open model you could self-host.

**What it can do**
- **Persistent memory** — remembers facts you tell it and recalls them in later
  turns and later sessions.
- **Notes** — "take a note…", "what are my notes?", search your notes.
- **Web search** — looks things up online through the **Tavily** API when you ask.
- **Emotion-aware replies** — infers mood/intent and adapts tone.
- **Speaker-aware** — identifies who is speaking (on the local pipeline).

**Why it fits Personal AI:** it's a private assistant with persistent memory,
reusable skills (tools), access to the information you choose, and the ability to
carry out tasks — with your biometrics and data staying on your own machine while
the reasoning runs on an open model.

---

## How we built it (tech + where Nebius/NVIDIA/Tavily are used)

- **NVIDIA Nemotron on Nebius Token Factory** is the reasoning brain. The app's
  LLM client speaks the OpenAI-compatible Chat Completions API, and Token Factory
  *is* OpenAI-compatible, so pointing the whole system at Nemotron was
  configuration, not new code: `provider: openai_compatible`,
  `base_url: https://api.tokenfactory.nebius.com/v1/`, `model:
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`. Nano handles fast everyday turns; Super
  and Ultra are a one-line swap for heavier reasoning.
- **One structured call per turn** returns mood, intent, sensitivity, the reply,
  and an optional tool action — so perception, reasoning, and tool-routing happen
  in a single Nemotron round trip.
- **Tavily** powers the `web_search` tool — a real runtime call to the Tavily
  search API returning a synthesized answer plus sources for the assistant to
  speak.
- **On-device perception** (local pipeline): faster-whisper (STT), SpeechBrain
  ECAPA-TDNN (speaker ID), an acoustic emotion classifier, Silero VAD, and
  system/Coqui TTS. Memory is SQLite.
- **Hosted surface** is deliberately torch-free (browser does the voice), so it
  hosts for free and wakes in seconds.

---

## What was significantly updated during the Submission Period (disclosure)

This project **composes two of my own pre-existing projects** — `voice-agent`
(a real-time voice pipeline) and `offline_assistant` (a task-automation tool
suite). Both predate the hackathon. **Everything that makes this a submission was
built during the Submission Period**, in a fresh repository (the originals are
untouched):

- **The merge itself** — a new adapter that bridges `offline_assistant`'s tool
  suite (notes, files, system info, activity report, code check) into
  `voice-agent`'s provider-agnostic tool registry, so one assistant has both the
  voice front-end and the task tools.
- **The entire Nebius + Nemotron integration** — pointing the OpenAI-compatible
  client at Token Factory, model selection, remote-key handling, and a critical
  latency fix (`detailed thinking off` + a token cap) that cut Nemotron voice
  turns from ~51s to ~6s.
- **The Tavily `web_search` tool** — a new runtime integration.
- **The lightweight hosted web app** — a new FastAPI surface with browser-based
  voice (Web Speech API), a curated public-safe tool set, per-session memory, and
  a Docker/Hugging Face Spaces deployment.
- **Reliability + correctness work** — a system-voice TTS backend (so the app
  doesn't depend on the hard-to-install voice-cloning stack), a SQLite notes
  store, a push-to-talk capture fix, dependency pinning (numpy<2 for torch),
  removing the numba/llvmlite/librosa dependency chain from the core, and tests.

Full history is in the public repository's commit log and `BUILD_TRACKER.md`.

---

## Feedback on Nebius, NVIDIA, and Tavily tools

Honest notes from actually building on these during the hackathon.

**Nebius Token Factory**
- The **OpenAI-compatible API is the standout** — an existing OpenAI-client app
  pointed at Token Factory with only config changes, no new SDK. This alone saved
  hours and is the single best thing about the platform for developers migrating.
- **Model catalog** is broad and current (Nemotron Nano/Super/Ultra, plus others);
  `GET /v1/models` made discovering exact model ids easy.
- **Friction:** (1) A **cold start** of ~30–60s on the first request to an idle
  model landed on the first user turn until we added a startup warmup — a documented
  "keep-warm"/provisioned option, or a faster cold path, would help real-time apps.
  (2) Billing is **threshold-based auto-charge** with no hard spending-limit
  toggle we could find; an explicit per-key/per-project spend cap would make it
  much safer to expose a public demo where each visitor consumes tokens.
- Pricing is transparent and, for Nano, very affordable for a voice loop.

**NVIDIA Nemotron**
- Nemotron 3 is a **reasoning model**, and by default it emits a long
  chain-of-thought that made structured voice turns extremely slow. The
  **`detailed thinking off` system directive works well** and is essential for
  low-latency use — but this is not obvious; it deserves prominent documentation
  in the Token Factory model card, ideally with a first-class parameter rather than
  a magic system string.
- With `response_format: json_object`, Nemotron produced reliable structured JSON
  (mood/intent/reply/tool-action) once we gave it enough `max_tokens` headroom to
  finish the object — worth flagging that a too-small cap silently truncates JSON.
- The Nano/Super/Ultra tiering is genuinely useful: Nano for fast turns, Super/Ultra
  when a query needs deeper reasoning.

**Tavily**
- Exactly the right shape for an LLM app: a single REST call returns a synthesized
  **answer plus ranked sources**, so we could speak a clean result instead of
  post-processing raw links. Integration was ~20 lines. `include_answer` is the
  killer feature.

---

## Try it
- **Demo URL:** _(paste your Hugging Face Space URL here)_
- **Repository:** https://github.com/fawad07/nebius-personal-ai
- **Video:** https://youtu.be/3PZBpnp4G68

Setup and architecture are in the repository `README.md`.
