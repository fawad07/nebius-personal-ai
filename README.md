---
title: Nebius Personal AI
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Nebius Personal AI

A **private, voice-driven personal assistant** that listens, recognizes *how*
you sound, reasons with an **NVIDIA Nemotron open model served on Nebius**, acts
on your machine through a suite of local tools, remembers you across sessions,
and replies in a **mood-matched voice** you can interrupt mid-sentence. The
default voice is the reliable system voice; optional XTTS **voice-cloning** is a
one-flag upgrade (`tts.engine: xtts`).

Built for the **Nebius x NVIDIA Global AI Hackathon** — *Personal AI* track.

> **Provenance / disclosure:** this project composes two of the author's
> pre-existing projects — `voice-agent` (real-time voice pipeline) and
> `offline_assistant` (task-automation tool suite) — into a new application. The
> Nebius/Nemotron integration and the merge are new work for this hackathon. The
> originals remain untouched; this repo is a copy-and-integrate, not a move.

---

## What it does

```
mic → VAD → ┌─ Whisper STT ───────────┐
            ├─ Speaker ID (ECAPA) ─────┤→  Nemotron on Nebius  →  reply + optional tool call
            └─ Emotion (acoustic SER) ─┘        (mood + intent + action, one call)
                                                       │
        speaker ◄── system voice / XTTS clone (mood-matched) ◄──┘
                        ▲
                        └── barge-in: your speech cancels playback
```

Everything except the LLM runs **on-device** (speech recognition, speaker
identification, emotion, text-to-speech). Only the reasoning prompt goes to the
Nebius-hosted open model — an open model you can self-host, not a closed API.

### Tools the assistant can call
Grafted from `offline_assistant`, dispatched through the voice agent's own
provider-agnostic tool layer (see `src/tools/offline_automation.py`):

| Tool | What it does |
|---|---|
| `list_files` / `tree` / `read_file` | Read anywhere in the project |
| `write_file` / `delete_file` | Write/delete — sandboxed to `data/workspace/` |
| `save_note` / `list_notes` / `search_notes` | Personal notes |
| `web_search` | Search the web via the **Tavily** API (runtime call) |
| `system_info` | Machine status (read-only) |
| `activity_report` | Markdown summary of the assistant's recent activity |
| `check_code` | Validate a Python snippet |

Plus the voice agent's built-ins: `get_current_time`, `remember_fact` /
`recall_facts`, and `synthesize_in_voice` (registered/consented voices only).

**Other services:** [Tavily](https://tavily.com) powers `web_search` (a real
runtime call to the Tavily API, enabled when `TAVILY_API_KEY` is set). The
hosted demo (`webapp/`) exposes a curated, public-safe subset of these tools
(time, memory, notes, web search).

---

## The Nebius / Nemotron integration

The voice agent's `LLMClient` is a wrapper around **any OpenAI-compatible Chat
Completions API**. Nebius Token Factory *is* OpenAI-compatible, so the entire
mandatory hackathon requirement is satisfied by configuration — no new client
code:

```yaml
# config/settings.yaml
llm:
  provider: "openai_compatible"
  model: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"       # Nemotron 3 Nano — fast/cheap for a voice loop
  base_url: "https://api.tokenfactory.nebius.com/v1/"  # Nebius Token Factory OpenAI-compatible endpoint
  system_preamble: "detailed thinking off"             # Nemotron is a reasoning model; this keeps turns ~1s not ~50s
```

The API key is read from `NEBIUS_API_KEY` (see `.env.example`). Remote endpoints
require a key; a local Ollama endpoint stays keyless for offline development.
Swap `model` to `nvidia/nemotron-3-super-120b-a12b` (Super) or
`nvidia/Nemotron-3-Ultra-550b-a55b` (Ultra) for deeper reasoning; use `--list`
in `scripts/nebius_spike.py` to see your account's catalog.

> **Cold start:** the first request to a serverless model can take ~30-60s while
> it spins up. A background warmup at startup (`llm.warmup`) absorbs this so it
> doesn't land on your first spoken turn.

---

## Setup

```bash
cd ~/Desktop/nebius-personal-ai
python3.11 -m venv .venv && source .venv/bin/activate   # TTS/torch need Python ≤3.11
pip install -r requirements.txt
cp .env.example .env        # paste your NEBIUS_API_KEY
./run.sh
```

A browser demo surface is served for the required demo URL (see `web/` and the
server module). See **BUILD_TRACKER.md** for status and the remaining work.

---

## Where things live

| Path | Role | From |
|---|---|---|
| `src/audio`, `src/stt`, `src/ser`, `src/speaker`, `src/tts` | Real-time voice pipeline | voice-agent |
| `src/llm/llm_client.py` | OpenAI-compatible client → **Nebius** | voice-agent |
| `src/conversation` | Persona + tool-routing dialogue | voice-agent |
| `src/tools/offline_automation.py` | **Adapter** exposing the automation suite as tools | new (the merge seam) |
| `automation/` | The task-automation tool modules | offline_assistant |
| `core/{paths,safety,behavior}.py` | Support the automation tools | offline_assistant |
| `web/` | Browser demo surface (→ demo URL) | voice-agent |

## Stretch goals (not in v1)
- **Biometric identity gating** (face-auth + liveness from `offline_assistant`) —
  makes it a "private assistant that knows its owner." Pair with SQLCipher-
  encrypted memory when added.
- Per-tool consent prompts surfaced in the voice/web UI.

## License
MIT — see [LICENSE](LICENSE). (Hackathon requires MIT / Apache-2.0 / MPL-2.0.)
