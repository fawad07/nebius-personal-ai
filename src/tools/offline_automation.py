"""
src/tools/offline_automation.py

Adapter that exposes the automation suite carried over from the
`offline_assistant` project as voice-agent ``Tool`` objects.

The two projects use different tool contracts:

  * offline_assistant declares tools in ``automation.TOOL_SPECS`` and
    dispatches them through ``AutomationController().execute(name, args)``,
    with each module's ``execute`` returning a short human-readable string.
  * voice-agent's ``ConversationManager`` consumes a ``ToolRegistry`` of
    ``Tool(name, description, params, func)`` objects, where
    ``func(args: dict) -> str``.

Both sides already speak "dict in, string out", so the bridge is thin: for
every dispatchable spec we wrap ``AutomationController.execute`` in a closure
of the right shape. The ``web_search`` spec (module ``None``) is intentionally
skipped — it is the one tool that leaves the machine and, in the original
design, is handled by the agent with per-call user approval rather than
dispatched blindly.
"""

from __future__ import annotations

import logging

from automation import TOOL_SPECS, AutomationController
from src.tools.registry import Tool

logger = logging.getLogger("tools")


def _format_params(args_spec: dict) -> str:
    """Render an automation spec's ``args`` dict as a voice-agent params hint."""
    if not args_spec:
        return "()"
    parts = [f'"{key}": <{desc}>' for key, desc in args_spec.items()]
    return "{" + ", ".join(parts) + "}"


def _make_dispatch(controller: AutomationController, name: str):
    def dispatch(args: dict) -> str:
        # AutomationController tools return a human-readable string; the
        # registry's own execute() already guards against exceptions, but we
        # keep the call site simple and let broken tools surface there.
        return controller.execute(name, args or {})
    return dispatch


def build_automation_tools(controller: AutomationController | None = None) -> list[Tool]:
    """
    Build voice-agent ``Tool`` objects for every dispatchable automation spec.

    A spec is dispatchable when it names a real module (``module is not None``);
    ``web_search`` is excluded on purpose (see module docstring).
    """
    controller = controller or AutomationController()
    tools: list[Tool] = []
    for name, spec in TOOL_SPECS.items():
        if spec.get("module") is None:
            continue
        tools.append(
            Tool(
                name=name,
                description=spec["summary"],
                params=_format_params(spec.get("args", {})),
                func=_make_dispatch(controller, name),
            )
        )
    logger.info("Loaded %d automation tools: %s", len(tools), [t.name for t in tools])
    return tools


def _wire_notes_store(notes_db_path: str) -> None:
    """Inject a SQLite notes store into automation.notes if not already set.

    The note-taking tools (save_note/list_notes/search_notes) are stateful and
    need a backend injected at startup; without it they report "no store
    configured". We use plain SQLite to match this project's other memory.
    """
    if not notes_db_path:
        return
    from automation import notes as notes_module
    if getattr(notes_module, "_STORE", None) is None:
        from src.memory.notes_store import SQLiteNotesStore
        notes_module.set_store(SQLiteNotesStore(notes_db_path))


def extend_registry_with_automation(registry, controller: AutomationController | None = None,
                                    notes_db_path: str = "data/notes.db"):
    """
    Register every automation tool onto an existing ``ToolRegistry`` (as built
    by ``build_default_registry``), so the voice assistant's built-in tools and
    the carried-over automation suite share one dispatch surface. Also wires the
    stateful note-taking store so save_note/list_notes/search_notes work.
    """
    _wire_notes_store(notes_db_path)
    for tool in build_automation_tools(controller):
        registry.register(tool)
    return registry
