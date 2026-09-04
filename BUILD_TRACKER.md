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
| Runs end-to-end (voice + Nebius) | 🔴 | Needs py3.11 venv + real key |
| Demo URL (browser surface) | 🟡 | Inherited web bridge; needs boot-check + polish |
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
- [ ] `./run.sh` full voice loop with Nebius as the brain
- [ ] Voice tool routing verified (`save_note`, `list_files`, …)
- [ ] Memory persists across restart (facts + conversation)
- [x] `data/workspace/.gitkeep` sandbox exists
- [ ] Merged pieces smoke-tested together

### Phase 2 — Product & Design
- [ ] Browser demo surface boots & is presentable (state, transcript, tool, reply)
- [ ] Persona / reusable-skills tuning
- [ ] **Tavily** wired into `web_search` (real runtime call → $3k bonus)
- [ ] Impact narrative written + shown in demo

### Phase 3 — Hardening / Production
- [ ] `pytest` green across merged suites
- [ ] Graceful handling of Nebius timeout / rate-limit / bad key
- [ ] Demo URL deployed publicly (Nebius Serverless Endpoint on-theme)
- [ ] Clean-machine install works from README alone
- [ ] No secrets committed; `.env.example` complete; MIT visible in About

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
