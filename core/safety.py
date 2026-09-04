"""
core/safety.py — gates tool execution behind safety/sensitivity checks.

Policy lives in config/permissions.yaml:
  - allowed_tools: allowlist — anything not listed is refused before it
    reaches the automation layer (defense in depth on top of the
    dispatcher's TOOL_MODULE_MAP, which would raise anyway).
  - sensitive_tools: tools that additionally require a fresh
    admin-presence check (enforced by core/agent.py via require_admin()).

Fail-closed: a missing or unreadable permissions file means an empty
allowlist — every tool call is refused, none are silently allowed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.paths import CONFIG_DIR

logger = logging.getLogger("safety")

DEFAULT_PERMISSIONS_PATH = CONFIG_DIR / "permissions.yaml"

# Used only if the YAML loads but omits sensitive_tools — sensitivity
# classification should never silently become "nothing is sensitive".
FALLBACK_SENSITIVE_TOOLS = {"delete_file", "system_shutdown", "install_app", "modify_permissions"}


class SafetySystem:
    def __init__(self, permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH):
        self.allowed_tools: set = set()
        self.sensitive_tools: set = set(FALLBACK_SENSITIVE_TOOLS)

        try:
            with open(permissions_path) as f:
                cfg = yaml.safe_load(f) or {}
        except OSError as e:
            logger.error(
                "Could not read permissions config %s (%s) — failing closed: "
                "all tool calls will be refused.", permissions_path, e,
            )
            return

        self.allowed_tools = set(cfg.get("allowed_tools", []))
        self.sensitive_tools = set(cfg.get("sensitive_tools", FALLBACK_SENSITIVE_TOOLS))

    def check_action(self, tool: str, args: dict) -> bool:
        if tool not in self.allowed_tools:
            logger.warning("Tool '%s' refused: not in allowed_tools allowlist.", tool)
            return False
        if not isinstance(args, dict):
            logger.warning("Tool '%s' refused: args must be a dict, got %s.", tool, type(args).__name__)
            return False
        return True

    def is_sensitive(self, tool: str) -> bool:
        return tool in self.sensitive_tools
