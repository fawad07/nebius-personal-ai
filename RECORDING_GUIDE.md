# Recording guide — the demo video (how & what)

Companion to `VIDEO_SCRIPT.md` (the spoken beats). This covers the mechanics.

## The one macOS gotcha: capturing the assistant's voice
The assistant speaks through **system audio** (browser TTS on the hosted app,
macOS `say` locally). The built-in recorder (Cmd+Shift+5) and QuickTime capture
**only the mic, not system audio** — so by default the assistant would be silent
on tape. Fixes:
- **OBS Studio (free):** captures screen + desktop audio + mic together. Best.
- **Loom (desktop app):** screen + system audio in one click.
- **No-system-audio trick:** turn *off* "Speak replies," record screen + mic
  narration, and narrate — the on-screen text proves the replies.

## Recommended setup (reliable + shows voice)
Record the **hosted web app**, **type the queries**, keep **"Speak replies" ON**:
- No mic input → **zero room-noise risk**.
- The assistant still **speaks** (browser TTS → system audio → captured).
- Narrate lightly over the top.

### OBS quick setup
1. Install OBS (obsproject.com), choose "optimize for recording."
2. Sources → + → **macOS Screen Capture** (display or browser window).
3. Settings → Audio: enable **Desktop Audio** (assistant voice) + **Mic** (you).
4. Allow the macOS Screen Recording + Microphone prompts.
5. Output: **MP4**, High. Video: **1920×1080**, **30 fps**.
6. Start Recording → run the demo → Stop.

## Before you hit record
- Quiet moment (only matters if using the mic; typing avoids it).
- **Warm up the model** with one throwaway query — no cold-start pause on camera.
- Clean, full-screen browser window; other tabs closed.
- Demo URL visible for the closing shot.
- Do one full **dry run** first.

## What to record (~2:30, see VIDEO_SCRIPT.md)
1. Hook (~15s): private voice assistant, Nemotron on Nebius.
2. **Web search (Tavily)** — "tallest building in the world?" → answer + sources. *(shows the Tavily bonus — keep it on camera)*
3. Memory — "remember my favourite colour is teal" → later recall it.
4. Notes — "take a note: buy milk" → "what are my notes?"
5. Tech point (~20s): show `config/settings.yaml` — Nemotron on Nebius, config-only.
6. Privacy/differentiator (~20s): on-device speech/speaker/emotion.
7. Close: demo URL on screen.

## Optional intro clip
Run `python scripts/intro.py` to have the assistant **introduce itself out loud
in its own voice** — a strong opening shot. Add `--save intro.wav` to capture a
clean audio file to drop into the edit. See that script's `--help`.

## Editing + export
- iMovie (free) or CapCut (free): trim dead air, cut to **under 3:00**
  (judges stop at 3:00 — front-load the best parts).
- **No copyrighted music.** Export 1080p MP4.

## Upload
- YouTube → Upload → **Public** (rules require "publicly visible").
- Title: "Nebius Personal AI — Nebius x NVIDIA Hackathon (Personal AI)".
- Paste the link into Devpost.

## Final checklist
- [ ] Under 3:00, 1080p, Public on YouTube
- [ ] Assistant's **voice audible** (or clear narration)
- [ ] **Tavily web search** shown working
- [ ] Memory + notes shown
- [ ] Nebius/Nemotron made concrete (config or spoken)
- [ ] Demo URL visible at the end
- [ ] No copyrighted music
