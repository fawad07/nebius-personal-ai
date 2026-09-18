"""
automation/__init__.py

Dispatches tool calls from the agent to the right automation module, and
is the single source of truth for what tools exist and what arguments
they take.

TOOL_SPECS drives three things that must not drift apart:
  - dispatch (TOOL_MODULE_MAP, derived below),
  - the tool list shown to the LLM (describe_tools()),
  - what a reader/maintainer sees as the available surface.

Tool names are namespaced implicitly rather than requiring the LLM to
know module paths — it says "list_files" and this routes to
automation.files.execute("list_files", args).
"""

from __future__ import annotations

import logging

from automation import files, system, apps, reports, workflows, notes, code_check

logger = logging.getLogger("automation")

TOOL_SPECS = {
    "list_files": {
        "module": files,
        "summary": "List files in one project directory (not recursive). Path is "
                   "relative to the project root; '.' is the whole project.",
        "args": {"path": "directory relative to the project root (optional, defaults to '.')"},
    },
    "tree": {
        "module": files,
        "summary": "Show a directory as an indented recursive tree of files and folders. "
                   "Path is relative to the project root.",
        "args": {"path": "directory to start from (optional, defaults to '.')"},
    },
    "read_file": {
        "module": files,
        "summary": "Read a text file anywhere in the project (e.g. core/agent.py).",
        "args": {"path": "file path relative to the project root"},
    },
    "write_file": {
        "module": files,
        "summary": "Create or overwrite a text file in the workspace (data/workspace/) only.",
        "args": {
            "path": "file path relative to the workspace",
            "content": "text to write",
        },
    },
    "delete_file": {
        "module": files,
        "summary": "Delete a file from the workspace (data/workspace/) only.",
        "args": {"path": "file path relative to the workspace"},
    },
    "system_info": {
        "module": system,
        "summary": "Report the machine's status: platform, CPU/load, memory, "
                   "disk space, battery, uptime. Read-only.",
        "args": {},
    },
    "activity_report": {
        "module": reports,
        "summary": "Summarise the assistant's own recent activity (tools used, "
                   "how busy, when) as a markdown report.",
        "args": {"days": "how many days back to cover (optional, default 7)"},
    },
    "save_note": {
        "module": notes,
        "summary": "Save a to-do / note the user dictates. Use ONLY when they "
                   'explicitly say "take a note", "note down", or "jot down" AND '
                   "give the thing to note (e.g. \"take a note: buy milk\" -> save "
                   '"buy milk"). Save the actual content only, never the words '
                   '"take a note". For personal facts/preferences use remember_fact '
                   "instead, not this.",
        "args": {"content": "the exact thing to note, e.g. 'buy milk' (never the phrase 'take a note')"},
    },
    "list_notes": {
        "module": notes,
        "summary": "List the user's saved notes, most recent first.",
        "args": {"limit": "how many to show (optional, default 20)"},
    },
    "search_notes": {
        "module": notes,
        "summary": "Find saved notes matching a word or phrase (e.g. what the "
                   "user noted about a topic).",
        "args": {"query": "word or phrase to look for"},
    },
    "check_code": {
        "module": code_check,
        "summary": "Check whether a snippet of Python is syntactically valid, "
                   "using Python's own parser. Use for 'is this code correct?'.",
        "args": {"code": "the Python code to check"},
    },
    "web_search": {
        # module None: this is the ONE tool that leaves the machine, so the
        # agent handles it directly (it needs the search conduit + the user's
        # per-search approval) and never dispatches it here. Only advertised
        # to the model when web search is enabled (see run_assistant.build_agent).
        "module": None,
        "summary": "Search the web. Use ONLY when the user explicitly asks to "
                   "search online or look something up on the internet — never "
                   "on your own. Each search needs the user's approval.",
        "args": {"query": "what to search for"},
    },
    # apps.py / workflows.py tools: add entries here as those modules grow
    # past their current stubs.
}

# Only real, dispatchable modules go in the map; web_search is agent-handled.
TOOL_MODULE_MAP = {name: spec["module"] for name, spec in TOOL_SPECS.items()
                   if spec["module"] is not None}


def describe_tools(allowed: set | None = None) -> str:
    """
    Render the tool list for the LLM's system prompt.

    Without this the prompt only named tool *categories*, so the model had
    no way to learn the real names and would invent plausible ones
    ("file_operations"), which the safety allowlist then refused — the
    agent could never successfully call a tool.

    allowed: restrict to this set (pass the safety allowlist) so the model
    is never advertised a tool it would be refused for calling.
    """
    lines = []
    for name, spec in TOOL_SPECS.items():
        if allowed is not None and name not in allowed:
            continue
        args = ", ".join(f'"{k}": <{v}>' for k, v in spec["args"].items())
        lines.append(f'- {name}: {spec["summary"]}\n    args: {{{args}}}')
    return "\n".join(lines)


class AutomationController:
    def execute(self, tool: str, args: dict, grant=None):
        """
        grant: a single path the human approved for this one call. Only
        the file tools understand it; modules that take no grant are
        called unchanged so they cannot be handed one by accident.
        """
        module = TOOL_MODULE_MAP.get(tool)
        if module is None:
            raise NotImplementedError(f"No automation module registered for tool '{tool}'")
        if grant is None:
            return module.execute(tool, args)
        return module.execute(tool, args, grant=grant)
