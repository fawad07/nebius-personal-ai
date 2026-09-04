from src.conversation.conversation_manager import ConversationManager

class DummyLLM:
    def generate_text(self, system_prompt, user_prompt):
        return "Hello, I'm here with you."

    def generate_json(self, system_prompt, user_prompt):
        return {
            "mood": "calm",
            "intent": "venting",
            "sensitivity": "high",
            "reply": "I hear you, and I'm here with you.",
        }

def test_conversation_basic():
    cm = ConversationManager(DummyLLM())
    mood = {"mood": "neutral", "intent": "casual_chat", "sensitivity": "medium"}
    reply = cm.respond("Hi", mood, "self")
    assert reply == "Hello, I'm here with you."

def test_conversation_history():
    cm = ConversationManager(DummyLLM())
    cm.update_history("Hi", "Hello", {"mood": "neutral"})
    assert len(cm.history) == 1

def test_respond_with_mood_merges_reply_and_mood_in_one_call():
    cm = ConversationManager(DummyLLM())
    result = cm.respond_with_mood("I've had a rough day", {"label": "sad"}, "self")
    assert result == {
        "mood": "calm",
        "intent": "venting",
        "sensitivity": "high",
        "reply": "I hear you, and I'm here with you.",
    }

def test_respond_with_mood_falls_back_on_missing_reply():
    class BadLLM:
        def generate_json(self, system_prompt, user_prompt):
            return {"mood": "calm"}  # no "reply" key

    cm = ConversationManager(BadLLM())
    result = cm.respond_with_mood("Hi", {"label": "neutral"}, "self")
    assert result["mood"] == "neutral"
    assert result["reply"]

def test_respond_with_mood_falls_back_on_llm_error():
    class BrokenLLM:
        def generate_json(self, system_prompt, user_prompt):
            raise RuntimeError("boom")

    cm = ConversationManager(BrokenLLM())
    result = cm.respond_with_mood("Hi", {"label": "neutral"}, "self")
    assert result["mood"] == "neutral"
    assert result["intent"] == "casual_chat"
    assert result["reply"]


def test_respond_with_mood_uses_plain_text_when_json_fails():
    """When JSON generation fails but plain text works (typical of small local
    models), the user should still get the model's REAL reply, not a canned line."""

    class NoJsonLLM:
        def generate_json(self, system_prompt, user_prompt):
            raise ValueError("model did not return JSON")

        def generate_text(self, system_prompt, user_prompt):
            return "Sure — here's a real answer from the model."

    cm = ConversationManager(NoJsonLLM())
    result = cm.respond_with_mood("Tell me something", {"label": "neutral"}, "self")
    assert result["reply"] == "Sure — here's a real answer from the model."
    assert result["mood"] == "neutral"  # analysis defaults applied in fallback
