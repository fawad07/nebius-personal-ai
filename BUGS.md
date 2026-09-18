# Bug log — Nebius Personal AI

Bugs found while building and demo-testing the merged app, with root cause, fix,
and the commit that fixed each. Newest first within each group.

---

## Reasoning / Nebius integration

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 1 | Every voice turn took **~51s** | Nemotron 3 is a reasoning model — it emitted a long chain-of-thought before answering | `system_preamble: "detailed thinking off"` + a per-response `max_tokens` cap (~51s → ~6s) | `f969eec` |
| 2 | Tool calls silently didn't fire (fell back to plain text) | `max_tokens` too low **truncated the structured JSON** mid-object, so the tool action was lost | Raised `max_tokens` to 1024 so the full JSON (reply + action) always completes | `f969eec` |
| 3 | First turn after startup was very slow | Serverless model **cold start** (~30–60s) landed on the user's first turn | Background warmup call at startup absorbs the cold start | `33d36cf` |
| 4 | README pointed at the wrong endpoint/model | Placeholder values from scaffolding | Corrected to real Token Factory endpoint + Nemotron model id | `965141a` |

## Voice pipeline

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 5 | "think" latency ~50s even after the reasoning fix | STT was **Whisper `medium`** on CPU (float32, single-thread) — the real bottleneck, not the LLM | Switched STT to `base` (~12s → ~1.3s per clip) | `0e8324b` |
| 6 | Transcripts were invented phrases ("thank you", "he's here") on silence | Whisper **hallucinates on non-speech/echo** | Enabled `vad_filter` + `condition_on_previous_text=False` + `temperature=0` (silence/noise → empty) | `18964bc` |
| 7 | Assistant answered things the user never said | Open mic picked up **ambient room speech** (a nearby phone/TV) and the assistant's own speaker output | Defaulted to **push-to-talk**; documented headphones for the local pipeline | `f03811a` |
| 8 | Push-to-talk captured **nothing** after pressing Enter | Capture sampled a single 256ms chunk and bailed if it wasn't already speech — impossible to hit | Gates declare `requires_onset_wait`; push-to-talk **waits (bounded) for speech to begin** | `6d6c197` |
| 9 | Robotic reply voice in the browser demo | The webapp used the browser's **default** speechSynthesis voice | Pick the most natural available voice (prefers Chrome "Google US English", then macOS Enhanced/Premium) + warmer prosody | `0054b72` |

## Tool routing / conversation quality

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 10 | `save_note`/`list_notes` said "no store configured" | The stateful notes backend was never injected into the carried-over tool | Added `SQLiteNotesStore`; auto-wire on graft | `f969eec` |
| 11 | Memory was flaky — "remember X" then "I don't know your X" | Model sometimes replied conversationally **without calling the tool**; `save_note`'s description said "use when the user says to *remember*", colliding with `remember_fact` | Disambiguated tool descriptions + hardened routing rules in the prompt | `28db635` |
| 12 | Personal facts showed up under **"what are my notes"** | "remember my favourite colour" was routed to `save_note` instead of `remember_fact`, polluting the notes store | **Deterministic pre-router** in the webapp: unambiguous memory/notes intents pick the tool directly; everything else falls through to the LLM | `1a68e91` |
| 13 | Reply contradicted itself: "Your colour is blue. I don't have anything remembered." | Empty `recall_facts` result was echoed alongside the answer the model knew from context | Tool-result phrasing now sees the recent conversation and is told never to contradict itself | `dc77b89` |
| 14 | "take a note" (no content) saved the literal words "take note" | No content-handling rule; trigger phrase saved as the note | Prompt: save the actual content only; empty "take a note" asks what to note | `28db635` |
| 15 | Invented **reminders** it can't set ("I've set a reminder for 3:15") | No reminder/scheduling tool exists; model filled the gap | Prompt: no scheduling ability — save a note instead, never claim a reminder | `28db635` |
| 16 | Redundant "**How can I help you today?**" tacked on mid-chat | Model added generic openers/closers | Prompt: conversation is ongoing — answer directly, no filler | `fb8b38a` |

## Environment / build

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| 17 | `run.sh` used system Python (no deps) | Launcher looked for `venv/`, project uses `.venv/`; also mis-parsed a commented `base_url`, and treated the remote Nebius endpoint as keyless | Detect `.venv`; strip inline comments; require a key for remote endpoints; gate XTTS checks on `tts.engine` | `d32186e` |
| 18 | `torch` warned "Failed to initialize NumPy" | torch 2.2.2 built against NumPy 1.x, but NumPy 2.x was installed | Pinned `numpy<2` | `d32186e` |
| 19 | XTTS won't install (llvmlite toolchain fails on Intel macOS) | Coqui `TTS` → `numba` → `llvmlite` needs an LLVM build | Made TTS pluggable; default **system voice**, XTTS optional; dropped librosa from the core | `d25d7c4` |
| 20 | `pytest` errored at collection | Two tests imported the optional XTTS module (needs librosa) | `importorskip` the optional deps; verify the default SystemTTS conforms instead | `9dd19ae` |
