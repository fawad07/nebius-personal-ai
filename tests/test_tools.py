from src.conversation.conversation_manager import ConversationManager
from src.memory.fact_store import FactStore
from src.tools.builtins import build_default_registry
from src.tools.registry import Tool, ToolRegistry


def test_registry_dispatch_and_errors():
    reg = ToolRegistry([Tool("echo", "echo back", '{"x": str}', lambda a: f"got {a.get('x')}")])
    assert reg.has("echo")
    assert reg.execute("echo", {"x": "hi"}) == "got hi"
    assert "unknown tool" in reg.execute("missing", {})


def test_registry_execute_swallows_tool_error():
    def boom(args):
        raise RuntimeError("kaboom")

    reg = ToolRegistry([Tool("boom", "always fails", "()", boom)])
    result = reg.execute("boom", {})
    assert "failed" in result  # error surfaced as text, not raised


def test_fact_store_roundtrip(tmp_path):
    store = FactStore(str(tmp_path / "facts.db"))
    store.add("daughter", "her name is Sara")
    store.add("daughter", "she likes drawing")
    assert store.get("daughter") == ["she likes drawing", "her name is Sara"]  # newest first
    assert "her name is Sara" in store.get()  # unfiltered
    store.close()


def test_builtin_time_tool():
    reg = build_default_registry(fact_store=None)
    assert "get_current_time" in reg.names()
    assert "remember_fact" not in reg.names()  # no store -> no memory tools
    out = reg.execute("get_current_time", {})
    assert isinstance(out, str) and out


def test_builtin_remember_and_recall(tmp_path):
    store = FactStore(str(tmp_path / "f.db"))
    reg = build_default_registry(fact_store=store)
    assert set(reg.names()) >= {"get_current_time", "remember_fact", "recall_facts"}
    reg.execute("remember_fact", {"subject": "pet", "fact": "cat named Milo"})
    assert "Milo" in reg.execute("recall_facts", {"subject": "pet"})
    store.close()


class ToolUseLLM:
    """First (json) call requests a tool; the follow-up text call phrases it."""

    def generate_json(self, system_prompt, user_prompt):
        return {
            "mood": "neutral",
            "intent": "task_request",
            "sensitivity": "low",
            "reply": "Sure, let me check.",
            "action": {"name": "get_current_time", "args": {}},
        }

    def generate_text(self, system_prompt, user_prompt):
        return "It's just past three in the afternoon."


def test_conversation_runs_requested_tool():
    reg = build_default_registry(fact_store=None)
    cm = ConversationManager(ToolUseLLM(), tools=reg)
    result = cm.respond_with_mood("what time is it?", {"label": "neutral"}, "self")
    # The rephrased, tool-informed reply is used instead of the pre-tool reply.
    assert result["reply"] == "It's just past three in the afternoon."
    assert result["intent"] == "task_request"


class NoActionLLM:
    def generate_json(self, system_prompt, user_prompt):
        return {"mood": "calm", "intent": "casual_chat", "sensitivity": "low",
                "reply": "Nice to chat.", "action": None}

    def generate_text(self, system_prompt, user_prompt):
        raise AssertionError("should not make a second call when no tool is requested")


def test_conversation_no_tool_when_action_null():
    reg = build_default_registry(fact_store=None)
    cm = ConversationManager(NoActionLLM(), tools=reg)
    result = cm.respond_with_mood("hi", {"label": "neutral"}, "self")
    assert result["reply"] == "Nice to chat."
