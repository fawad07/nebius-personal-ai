from src.conversation.conversation_manager import ConversationManager
from src.memory.conversation_store import ConversationStore


class DummyLLM:
    def generate_json(self, system_prompt, user_prompt):
        return {"mood": "calm", "intent": "venting", "sensitivity": "low", "reply": "ok"}

    def generate_text(self, system_prompt, user_prompt):
        return "ok"


def test_store_append_and_recent_roundtrip(tmp_path):
    db = str(tmp_path / "memory.db")
    store = ConversationStore(db)
    store.append("s1", "hello", "hi there", {"mood": "calm"})
    store.append("s1", "how are you", "doing well", {"mood": "neutral"})

    recent = store.recent(10)
    assert [h["user"] for h in recent] == ["hello", "how are you"]  # chronological
    assert recent[0]["reply"] == "hi there"
    assert recent[0]["mood"] == {"mood": "calm"}
    store.close()


def test_recent_respects_limit_and_order(tmp_path):
    store = ConversationStore(str(tmp_path / "m.db"))
    for i in range(5):
        store.append("s", f"u{i}", f"r{i}", {})
    recent = store.recent(2)
    assert [h["user"] for h in recent] == ["u3", "u4"]  # last two, oldest->newest
    store.close()


def test_history_persists_across_manager_instances(tmp_path):
    db = str(tmp_path / "memory.db")

    store1 = ConversationStore(db)
    cm1 = ConversationManager(DummyLLM(), store=store1)
    cm1.update_history("hi", "hello", {"mood": "neutral"})
    store1.close()

    # A fresh manager on the same DB should see the earlier turn.
    store2 = ConversationStore(db)
    cm2 = ConversationManager(DummyLLM(), store=store2)
    assert len(cm2.history) == 1
    assert cm2.history[0]["user"] == "hi"
    assert cm2.history[0]["reply"] == "hello"
    store2.close()


def test_manager_without_store_still_works():
    cm = ConversationManager(DummyLLM())
    cm.update_history("hi", "hello", {"mood": "neutral"})
    assert len(cm.history) == 1
