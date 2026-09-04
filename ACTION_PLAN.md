# Action Plan — Nebius Personal AI

> Prototype → Production plan for the **Nebius x NVIDIA Global AI Hackathon**,
> **Personal AI** track. Companion to `BUILD_TRACKER.md` (live status board).
> **Deadline: 2026-10-30, 10:00 PT.** Today: 2026-09-01 (~8 weeks of runway).

---

## The goal, in one sentence
A private, voice-driven personal assistant that reasons with an **NVIDIA Nemotron
model served on Nebius Token Factory**, remembers you, and carries out tasks on
your machine through a local tool suite — shipped with a public demo URL, a
<3-minute video, and a public MIT repo.

## Definition of Done (the submission gates)
A build is "done" only when every one of these is true — they map 1:1 to the rules:

- ✅ Makes a **runtime call to Nebius Token Factory** (gate #1)
- ✅ Uses an **NVIDIA open model / Nemotron** (gate #2)
- ✅ **Public demo URL** that works, free & unrestricted
- ✅ **Public repo** with MIT license visible in the About section
- ✅ **README** with setup + how NVIDIA/Nemotron/Token Factory are used
- ✅ **YouTube video < 3 min** showing it running
- ✅ **Track identified** (Personal AI) + **tools feedback** written
- ✅ **"Significantly updated" explanation** (this is a pre-existing-project merge)

## How the plan targets the 4 judging criteria (each 25%)
| Criterion | Where it's won | Phase |
|---|---|---|
| Technological Implementation | Clean Nebius/Nemotron use + working tools | 1, 3 |
| Design | Polished browser demo, coherent product feel | 2 |
| Potential Impact | Clear "private assistant that acts for you" narrative | 2, 4 |
| Quality of Idea | Voice + emotion + on-device biometrics + open cloud brain | 2, 4 (stretch) |

---

## Phase 0 — Access & Spike  ·  *goal: prove Nebius works*  ·  ~½ day
**Exit criteria:** one Nemotron reply comes back through the existing client.

1. Create Nebius account → get **Token Factory API key** → `.env` (`NEBIUS_API_KEY`).
2. (Optional) Join the Nebius Builder Program for credits.
3. Confirm the **exact Nemotron model id** + **base_url** from the catalog; set in `config/settings.yaml`.
4. `python3.11 -m venv .venv && pip install -r requirements.txt`.
5. **Spike script** — one `LLMClient` call to Nebius, print the reply. Prove auth + connectivity before touching audio.

## Phase 1 — Prototype (MVP)  ·  *goal: pass Stage 1 (viable + on-theme)*  ·  ~2–4 days
**Exit criteria:** speak → Nemotron reasons → a tool fires → cloned voice replies, locally.

1. `./run.sh` — full voice loop with Nebius as the brain.
2. **Verify tool routing by voice**: "take a note: buy milk" → `save_note`; "what files are here?" → `list_files`. (Nemotron's JSON/tool formatting may differ from GPT — tune the system prompt / lean on `extract_json()`.)
3. Confirm memory persists across restarts (`remember_fact` / `recall_facts`, conversation store).
4. Create `data/workspace/.gitkeep` so file tools have their sandbox. *(done)*
5. Smoke-test the merged pieces together; fix import/path issues.

## Phase 2 — Product & Design  ·  *goal: win Design + Impact*  ·  ~1 week
**Exit criteria:** a stranger can open the demo URL and "get it" in 30 seconds.

1. **Browser demo surface** — boot voice-agent's web bridge; make it presentable (state indicator, transcript, what tool ran, the reply). This *is* the demo URL and 25% of the score.
2. **Persona & skills** — tune the assistant's personality and the "reusable skills" framing the Personal AI track asks for.
3. **Tavily bonus ($3k)** — wire the `web_search` tool to a real **Tavily API** runtime call (currently agent-handled/disabled). Low effort, separate prize, strengthens "tools you choose."
4. **Impact narrative** — write the "who is this for / what problem" story; make the demo show it.

## Phase 3 — Hardening / Production  ·  *goal: reliable under a judge's hands*  ·  ~3–5 days
**Exit criteria:** fresh clone → setup → runs first try; tests green.

1. Merge & green the two test suites (`pytest`); fix path/import drift.
2. Error handling: Nebius timeout / rate-limit / bad-key surfaces gracefully, not a crash mid-demo.
3. **Deploy the demo URL** somewhere public (Nebius Serverless Endpoint is on-theme and scores well; a simple hosted box also works). Confirm it's reachable & unrestricted.
4. Verify a clean-machine install from the README alone.
5. Config sanity: no secrets committed; `.env.example` complete; MIT license shows in About.

## Phase 4 — Submission Assets  ·  *goal: submit*  ·  ~2–3 days
**Exit criteria:** Devpost form complete before Oct 30, 10:00 PT.

1. **Demo video (<3 min, YouTube public):** voice in → mood detected → Nemotron reasons → tool acts → cloned voice out. No copyrighted music.
2. **Project text description** — features + functionality.
3. **NVIDIA/Nebius usage section** — exactly where Nemotron + Token Factory are used.
4. **"Significantly updated" explanation** — the merge + Nebius integration is the new work; be specific.
5. **Tools feedback** — honest notes on Token Factory / AI Cloud / NVIDIA (own bonus prize).
6. Push public repo; fill and submit the Devpost form. **Submit a day early.**

## Stretch (only if Phases 0–4 land with time to spare)
- **Biometric identity gating** — face-auth + liveness from `offline_assistant` (`core/auth.py`, `face_embedding.py`, `anti_spoof_model.py`, `liveness.py`). Heavy deps (PyQt6, insightface). This is the "knows its owner" hook and a strong Quality-of-Idea boost.
- **SQLCipher-encrypted memory** — swap in alongside face-auth (they belong together).
- **Per-tool consent** surfaced in the UI.

---

## Suggested timeline (buffer built in; real work ≈ 3 focused weeks)
| Window | Focus |
|---|---|
| Wk of Sep 1 | Phase 0 + start Phase 1 (get it talking to Nebius) |
| Wk of Sep 8 | Finish Phase 1 (MVP works end to end) |
| Wk of Sep 15 | Phase 2 (demo surface + Tavily) |
| Wk of Sep 22 | Phase 3 (hardening + deploy demo URL) |
| Wk of Sep 29 | Buffer / stretch (face-auth) |
| Wk of Oct 6–20 | Polish, re-test, record draft video |
| Wk of Oct 27 | Phase 4 assets, **submit by Oct 30** |

## Risk register
| Risk | Impact | Mitigation |
|---|---|---|
| Nemotron formats tool JSON differently than GPT | Tools never fire | Test routing in Phase 1; harden prompt; `extract_json()` is lenient |
| Model id / base_url drift from placeholders | Auth fails at first run | Confirm from live catalog in Phase 0 |
| Demo URL flaky under judging | Lost points / DQ risk | Deploy early (Phase 3), keep it free & unrestricted till judging ends |
| Voice deps (torch 2.2.2 pin) on wrong Python | Won't install | Python ≤3.11 venv, per README |
| Scope creep into face-auth too early | MVP slips | Face-auth is stretch, gated behind Phases 0–4 |
| Copyrighted music in video | Video rejected | Use licensed/no music |
