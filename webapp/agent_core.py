"""
webapp/agent_core.py

The torch-free reasoning core for the hosted demo: Nemotron (on Nebius) +
tools + per-session memory + Tavily. Reuses the exact modules the full voice
app uses (LLMClient, ConversationManager, the tool registry), minus the audio
pipeline — so the public demo is tiny, cheap to host, and can't choke.

Only a curated, safe subset of tools is exposed publicly (time, personal
memory, notes, web search). The file/system/code tools from the automation
suite are intentionally left out of the hosted surface — they belong to the
private, on-your-own-machine experience, not a public URL.
"""

from __future__ import annotations

import os
import threading

from utils.config_loader import load_config
from src.llm.llm_client import LLMClient
from src.tools.registry import ToolRegistry
from src.tools.builtins import build_default_registry
from src.tools.offline_automation import extend_registry_with_automation
from src.tools.web_search import build_web_search_tool
from src.memory.fact_store import FactStore
from src.conversation.conversation_manager import ConversationManager

# Tools safe to expose on a public URL.
_PUBLIC_TOOLS = {
    "get_current_time",
    "remember_fact",
    "recall_facts",
    "save_note",
    "list_notes",
    "search_notes",
    "web_search",
}


def _is_local(url: str | None) -> bool:
    return bool(url) and any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0"))


class AgentCore:
    def __init__(self, config_path: str = "config/settings.yaml"):
        cfg = load_config(config_path)
        llm_cfg = cfg.get("llm", {})
        base_url = llm_cfg.get("base_url")
        api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")

        self.llm = LLMClient(
            model=llm_cfg.get("model"),
            api_key=api_key,
            base_url=base_url,
            require_key=not _is_local(base_url),
            provider=llm_cfg.get("provider", "openai_compatible"),
            max_calls_per_minute=llm_cfg.get("max_calls_per_minute", 30),
            max_tokens_total=llm_cfg.get("max_tokens_budget", 50_000),
            max_tokens=llm_cfg.get("max_tokens", 1024),
            system_preamble=llm_cfg.get("system_preamble", "detailed thinking off"),
        )

        # Build the full registry, then keep only the public-safe subset.
        full = build_default_registry(fact_store=FactStore(db_path="data/facts.db"))
        extend_registry_with_automation(full, notes_db_path="data/notes.db")
        search_cfg = cfg.get("search", {})
        tavily_key = os.getenv("TAVILY_API_KEY")
        if search_cfg.get("enabled", True) and tavily_key:
            full.register(build_web_search_tool(
                tavily_key, max_results=search_cfg.get("max_results", 5)))
        self.registry = ToolRegistry(
            [full.get(n) for n in full.names() if n in _PUBLIC_TOOLS]
        )

        self._persona = cfg.get("conversation", {})
        self._history_limit = cfg.get("memory", {}).get("history_limit", 20)
        self._sessions: dict[str, ConversationManager] = {}
        self._lock = threading.Lock()

    def _manager(self, session_id: str) -> ConversationManager:
        with self._lock:
            cm = self._sessions.get(session_id)
            if cm is None:
                cm = ConversationManager(
                    self.llm,
                    persona_config=self._persona,
                    tools=self.registry,
                    history_limit=self._history_limit,
                )
                self._sessions[session_id] = cm
            return cm

    def respond(self, session_id: str, text: str) -> dict:
        cm = self._manager(session_id)
        result = cm.respond_with_mood(text, {"label": "neutral"}, "primary")
        cm.update_history(
            text,
            result.get("reply", ""),
            {
                "mood": result.get("mood"),
                "intent": result.get("intent"),
                "sensitivity": result.get("sensitivity"),
            },
        )
        return result

    def tool_names(self) -> list[str]:
        return self.registry.names()
