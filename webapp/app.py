"""
webapp/app.py

Lightweight FastAPI surface for the hosted demo. Serves a browser chat page
(voice via the Web Speech API, so no server-side audio) and a /chat endpoint
backed by the torch-free AgentCore (Nemotron + tools + memory + Tavily).

Run locally:  uvicorn webapp.app:app --host 0.0.0.0 --port 7860
On Hugging Face Spaces (Docker), the container runs the same command on 7860.
"""

from __future__ import annotations

import os
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from webapp.agent_core import AgentCore

app = FastAPI(title="Nebius Personal AI")
core = AgentCore()

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "static", "index.html"), encoding="utf-8") as _f:
    _INDEX_HTML = _f.read()


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


@app.post("/chat")
def chat(inp: ChatIn) -> dict:
    session_id = inp.session_id or uuid.uuid4().hex
    message = (inp.message or "").strip()
    if not message:
        return {"session_id": session_id, "reply": "(say something!)", "mood": "neutral"}
    result = core.respond(session_id, message)
    return {
        "session_id": session_id,
        "reply": result.get("reply", ""),
        "mood": result.get("mood", "neutral"),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "tools": core.tool_names()}
