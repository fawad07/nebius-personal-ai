# Build Tracker — Nebius Personal AI

> Live status board. Plan & rationale live in `ACTION_PLAN.md`.
> Personal AI track · **Deadline: 2026-10-30, 10:00 PT** · Legend: 🔴 not started · 🟡 in progress · 🟢 done

Last updated: 2026-09-01

---

## Snapshot / Health
| Area | State | Notes |
|---|---|---|
| Folder scaffolded (copy of voice-agent) | 🟢 | `~/Desktop/nebius-personal-ai`; originals untouched |
| Automation tools grafted (adapter) | 🟢 | `src/tools/offline_automation.py`; 11 tools build + dispatch verified (py3.14, no torch) |
| Nebius/Nemotron wired (config-only) | 🟢 | Key works; endpoint confirmed; spike passed. Default model = Nemotron 3 Nano |
| Config schema updated | 🟢 | `tools.automation` added; validator accepts new keys, rejects typos |
| Remote-endpoint key handling | 🟢 | `NEBIUS_API_KEY` → falls back to `OPENAI_API_KEY` |
| Runs end-to-end (voice + Nebius) | 🟢 | Full pipeline boots + a live turn logged; time/files/notes tools fire; ~6s/turn. Voice re-test with fixes pending user mic |
| Demo URL (browser surface) | 🟢 built | Two surfaces: heavy local voice (`voice_server --real`) + lightweight hosted webapp (torch-free, browser voice). Hosted app verified locally; public deploy = user's HF account |
| Tests green (merged suites) | 🔴 | Not yet run together |
| Demo video | 🔴 | Phase 4 |
| Repo public + license | 🟢 MIT / 🔴 push | Push before submission |

## Submission gates (must all be 🟢 to submit)
| Gate | Status |
|---|---|
| Runtime call to Nebius Token Factory | 🟢 (spike passed) |
| Uses NVIDIA Nemotron / open model | 🟢 (Nemotron on Nebius) |
| Working public demo URL (free, unrestricted) | 🔴 |
| Public repo + MIT license visible in About | 🟢 / 🔴 push |
| README w/ setup + NVIDIA/Token Factory usage | 🟢 |
| YouTube video < 3 min showing it run | 🔴 |
| Track identified (Personal AI) + tools feedback | 🟡 / 🔴 |
| "Significantly updated" explanation | 🟡 (started in README) |

---

## Task board

### Phase 0 — Access & Spike ✅ DONE
- [x] Nebius account + Token Factory API key → `.env` (gitignored)
- [ ] (opt) Join Nebius Builder Program for $25 credits
- [x] Confirmed Nemotron model ids + base_url → `config/settings.yaml` (default: Nano)
- [x] `.venv` created (openai installed; full `requirements.txt` in Phase 1)
- [x] Spike passed: Nemotron call via Nebius returns a reply

### Phase 1 — Prototype (MVP)
- [x] `./run.sh` full voice loop with Nebius as the brain (boots, one live turn logged)
- [x] Tool routing verified via Nebius (`get_current_time`, `list_files`, `save_note`/`list_notes`)
- [x] Notes persist (SQLite notes store wired)
- [x] Latency fixed (reasoning off + max_tokens): ~51s → ~6s/turn
- [x] `data/workspace/.gitkeep` sandbox exists
- [x] Merged pieces smoke-tested together (34 logic tests pass)
- [ ] **Voice re-test with fixes** (user mic): confirm ~6s reply + tool by voice
- [ ] Memory persists across restart (facts + conversation) — verify

### Phase 2 — Product & Design
- [x] Browser demo surface boots & is presentable (state, transcript, speaker/emotion/mood, tool, reply)
- [x] Text input wired to the real Agent (judge-testable without a mic)
- [x] Tools + memory verified in-browser (get_current_time, save_note→list_notes across turns)
- [ ] Persona / reusable-skills tuning
- [x] **Tavily** wired into `web_search` (real runtime call verified in-browser → $3k bonus)
- [ ] Impact narrative written + shown in demo
- [ ] Voice-in via browser mic verified (quiet env / headset)

### Phase 3 — Hardening / Production
- [x] Lightweight hosted webapp built (torch-free; FastAPI + browser voice)
- [x] Hosted app verified in-browser (time tool + Tavily + per-session memory)
- [x] Dockerfile + .dockerignore (keeps .env/heavy files out) + DEPLOY.md
- [ ] **Deploy to Hugging Face Spaces** (user's HF account: create Space, add 2 secrets, push)
- [ ] Push public code repo (GitHub) for judging
- [ ] `pytest` green across merged suites
- [ ] Graceful handling of Nebius timeout / rate-limit / bad key
- [ ] Set Nebius account spend limit before sharing the URL

### Phase 4 — Submission Assets
- [ ] Demo video (<3 min, public YouTube, no copyrighted music)
- [ ] Project text description
- [ ] NVIDIA/Nemotron + Token Factory usage section
- [ ] "Significantly updated" explanation (the merge + Nebius work)
- [ ] Tools feedback (own bonus prize)
- [ ] Repo pushed public; Devpost form submitted (**a day early**)

### Stretch (post-MVP only)
- [ ] Face-auth + liveness identity gating (from offline_assistant)
- [ ] SQLCipher-encrypted memory (with face-auth)
- [ ] Per-tool consent in the UI

---

## Changelog
| Date | Change |
|---|---|
| 2026-09-01 | Scaffolded from voice-agent; grafted offline_assistant automation via adapter; wired Nebius/Nemotron via config + `NEBIUS_API_KEY`; added `tools.automation` to schema; MIT license; README, ACTION_PLAN, this tracker. Graft verified (11 tools) + config validated (py3.14). |
| 2026-09-01 | Set `max_tokens_budget: 50000` + account spend-limit note in config. |
| 2026-09-01 | **Phase 1 prep (pre-emptive):** hardened `extract_json` to strip reasoning `<think>` blocks and scan for the real structured object — fixes the risk that Nemotron's chain-of-thought would silently drop tool calls. Added 3 regression tests. Verified against existing + new cases. |
| 2026-09-01 | **Install unblock (reliability):** XTTS voice-clone wouldn't install (llvmlite toolchain fails on Intel macOS). Made TTS pluggable: new `SystemTTS` (macOS `say`, 16 kHz float32, zero heavy deps) is now the default `tts.engine`; XTTS kept behind `engine: xtts` + `requirements-voiceclone.txt`, imported lazily. Also removed the last `librosa`/`numba` dependency by reimplementing SER pitch estimation in numpy (FFT-autocorrelation; recovers a 150 Hz tone at 150.1 Hz). Core `requirements.txt` now has no numba/llvmlite. 13 SER+LLM tests green. Voice-clone → stretch goal. |
