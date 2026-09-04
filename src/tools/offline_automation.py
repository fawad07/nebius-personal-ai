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


def extend_registry_with_automation(registry, controller: AutomationController | None = None):
    """
    Register every automation tool onto an existing ``ToolRegistry`` (as built
    by ``build_default_registry``), so the voice assistant's built-in tools and
    the carried-over automation suite share one dispatch surface.
    """
    for tool in build_automation_tools(controller):
        registry.register(tool)
    return registry
