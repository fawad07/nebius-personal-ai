# Deploy the hosted demo (near-$0)

The public demo URL is the **lightweight web app** in `webapp/` — Nemotron on
Nebius + tools + memory + Tavily, with voice in the browser (Web Speech API).
It's torch-free, so it hosts free and wakes in seconds.

Recommended host: **Hugging Face Spaces (free, Docker)**. Free CPU, public URL,
secrets management, no card, runs until judging. (A free Space sleeps after ~48h
idle and wakes on the next visit — fine, and fast because the app is light.)

## Run it locally first
```bash
cd ~/Desktop/nebius-personal-ai
source .venv/bin/activate
set -a; source .env; set +a          # NEBIUS_API_KEY (+ optional TAVILY_API_KEY)
pip install -r requirements-web.txt
uvicorn webapp.app:app --host 127.0.0.1 --port 7860
# open http://127.0.0.1:7860
```

## Deploy to Hugging Face Spaces
1. Create a free account at https://huggingface.co, then **New → Space**.
   - **SDK: Docker** (blank template), name e.g. `nebius-personal-ai`, **Public**.
2. Add the HF frontmatter to the **top of `README.md`** (HF needs it on a Docker
   Space; it's harmless on GitHub):
   ```
   ---
   title: Nebius Personal AI
   emoji: 🎙️
   colorFrom: indigo
   colorTo: teal
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
3. **Add your secrets** in the Space: Settings → *Variables and secrets* →
   - `NEBIUS_API_KEY` = your Nebius key
   - `TAVILY_API_KEY` = your Tavily key (optional; enables web search)
   > Secrets are injected as env vars at runtime. **Never commit `.env`** — the
   > `.dockerignore` already excludes it.
4. Push the code to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/nebius-personal-ai
   git push space main
   ```
   The Space builds the `Dockerfile` and starts `uvicorn` on 7860.
5. Your demo URL: `https://<your-username>-nebius-personal-ai.hf.space`

## Cost & safety notes
- **$0** on the free CPU Space. Only your Nebius/Tavily token usage costs anything,
  bounded by `llm.max_tokens_budget` per session (config).
- The hosted surface exposes only safe tools (time, memory, notes, web search) —
  no file/system access (see `webapp/agent_core.py`).
- Set a spending limit in your Nebius console before sharing the URL widely; each
  visitor is a fresh session (fresh token budget).
