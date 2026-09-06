# Demo video script (target: under 3:00)

Record in a quiet room. Show the product working on screen the whole time.
No copyrighted music. Screen-record the hosted web app and/or the local voice app.

---

**0:00–0:20 — Hook + what it is**
> "This is Nebius Personal AI — a private voice assistant that reasons with an
> NVIDIA Nemotron model on Nebius, remembers me, and gets things done. Everything
> except the language model runs on my own machine."

(Show the app open, the mic button, the badges: Nemotron · Nebius.)

**0:20–0:50 — Core loop (voice or type)**
> "I can just talk to it."

Ask: *"What's the tallest building in the world?"* → it searches the web (Tavily)
and answers with sources, out loud.

(If demoing the local pipeline, mention: "it heard me, figured out my mood, and
reasoned with Nemotron — one call.")

**0:50–1:25 — Memory + tasks (the Personal AI story)**
- *"Remember that my favourite colour is teal."* → it confirms.
- *"Take a note: pick up milk on the way home."* → confirms.
- *"What are my notes?"* → reads the note back.
- *"What's my favourite colour?"* → recalls "teal."

> "It remembers across turns and across sessions — that's the persistent memory
> the Personal AI track is about."

**1:25–2:00 — How it's built (the tech, briefly)**
> "The reasoning runs on NVIDIA Nemotron through Nebius Token Factory. Because
> Token Factory is OpenAI-compatible, wiring it up was pure configuration — no new
> code. Nano keeps voice turns fast; I can swap to Super or Ultra for heavier
> reasoning. Web search is a real Tavily API call."

(Show `config/settings.yaml` llm block for a beat, or the tools list.)

**2:00–2:30 — Privacy / the differentiator**
> "On the full local version, speech recognition, speaker identification, and
> emotion detection all run on-device. Only the reasoning prompt goes to an open
> model I could self-host. Your data stays yours; the brain is open."

(Optional: show speaker/emotion/mood updating on the local app.)

**2:30–2:55 — Close**
> "It's live — you can try the hosted version in your browser, type or talk. Built
> for the Nebius x NVIDIA hackathon, Personal AI track. Thanks for watching."

(Show the demo URL on screen.)

---

## Shot list / checklist
- [ ] Quiet room (or type, to avoid mic noise)
- [ ] Hosted web app open in a clean browser window
- [ ] Do a warm-up turn first so there's no cold-start pause on camera
- [ ] Web search turn (Tavily) — shows current info + sources
- [ ] Memory turn (remember → recall) and a note (save → list)
- [ ] A glance at the config/tools to make the Nebius/Nemotron point concrete
- [ ] Demo URL visible at the end
- [ ] Export < 3:00, upload to YouTube as Public, no copyrighted audio
