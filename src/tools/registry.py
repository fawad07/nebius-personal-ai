import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("tools")


@dataclass
class Tool:
    """
    One callable capability the assistant can invoke.

    ``func`` takes a dict of arguments and returns a short human-readable
    result string (which is then woven into the spoken reply). ``params`` is a
    brief hint of the expected args, shown to the model in the prompt.
    """

    name: str
    description: str
    params: str
    func: Callable[[dict], str]


class ToolRegistry:
    """
    Provider-agnostic tool layer. The model requests a tool by returning an
    ``action`` object in its structured JSON (see ConversationManager); the
    registry dispatches it. This avoids depending on any vendor's native
    tool/function-calling API, so it works identically with hosted OpenAI and
    local models (Ollama, etc.).
    """

    def __init__(self, tools=None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def has(self, name: str) -> bool:
        return name in self._tools

    def describe(self) -> str:
        """Render the tool list for the system prompt."""
        return "\n".join(
            f"- {t.name}{t.params}: {t.description}" for t in self._tools.values()
        )

    def execute(self, name: str, args: dict | None) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"(unknown tool: {name})"
        try:
            return tool.func(args or {})
        except Exception as e:  # a broken tool must not break the conversation
            logger.exception("Tool %s failed.", name)
            return f"(tool {name} failed: {e})"

    def __len__(self) -> int:
        return len(self._tools)
